#!/usr/bin/env python3
"""Frozen author-style dual-task TCR-BERT CDR3beta AA-only mean embedding."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(_release_path('path_1304218e86d2e596'))
OUT = ROOT / "embeddings" / "tcrbert_dualtask_aa_mean"
SOURCE = Path(_release_path('path_b0b906586de9611b'))
INPUT = ROOT / "data" / "aa_union.parquet"
BASE_PATH = ROOT / "src" / "02_embed.py"
spec = importlib.util.spec_from_file_location("embedding_base", BASE_PATH)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

MODEL_NAME = "tcrbert_dualtask_aa_mean"
FILES = ("pytorch_model.bin", "config.json", "vocab.txt",
         "tokenizer_config.json", "special_tokens_map.json")
HF_REVISION = "ef65ddcb4e549990e584680e27f9ae2618c884ff"


def build_model_spec() -> dict:
    import torch
    base.require(all((SOURCE / f).is_file() for f in FILES),
                 "Author dual-task checkpoint missing required files")
    cfg = json.loads((SOURCE / "config.json").read_text())
    base.require(cfg["model_type"] == "bert", "Checkpoint is not BERT")
    base.require(cfg["architectures"] == ["BertForSequenceClassification"],
                 "Checkpoint is not dual-task sequence-classification model")
    base.require(cfg["num_hidden_layers"] == 12 and cfg["hidden_size"] == 768,
                 "Unexpected author BERT architecture")
    base.require(cfg["max_position_embeddings"] == 64,
                 "Unexpected author BERT positional limit")
    return {
        "model": MODEL_NAME,
        "model_source": "https://huggingface.co/wukevin/tcr-bert",
        "source_revision": HF_REVISION,
        "paper": "Wu et al., PMLR 240:194-229 (2024), https://proceedings.mlr.press/v240/wu24b.html",
        "paper_pooling": "Methods p.15: average each input amino acid's final-layer embedding",
        "model_path": str(SOURCE),
        "model_files_sha256": {f: base.sha256(SOURCE / f) for f in FILES},
        "weight_file_used": "pytorch_model.bin",
        "safetensors": "not loaded; local copy lacks compatible format metadata",
        "input_fields": ["seq_idx", "aa"],
        "tokenization": "uppercase CDR3 amino acids separated by spaces; BertTokenizerFast; no truncation",
        "pooling": "final layer, arithmetic mean of AA tokens only; CLS, SEP, PAD excluded",
        "hidden_dim": 768,
        "max_model_tokens": 64,
        "bank_dtype": "float16",
        "inference_precision": "fp16 CUDA; fp32 CPU reference",
        "script_sha256": base.sha256(Path(__file__)),
        "shared_io_script_sha256": base.sha256(BASE_PATH),
        "torch": torch.__version__,
        "transformers": base.version("transformers"),
        "numpy": base.version("numpy"),
        "pyarrow": base.version("pyarrow"),
    }


def residue_mean(hidden, attention_mask, lengths):
    """Mean over original AA positions; reject tokenization that alters any AA."""
    import torch
    counts = attention_mask.sum(dim=1)
    expected = torch.tensor([len(aa) + 2 for aa in lengths],
                            device=counts.device, dtype=counts.dtype)
    base.require(torch.equal(counts, expected),
                 "Tokenizer did not map each raw AA to one token plus CLS/SEP")
    mask = attention_mask.clone()
    mask[:, 0] = 0
    mask[torch.arange(mask.shape[0], device=mask.device), counts - 1] = 0
    base.require(torch.equal(mask.sum(dim=1), expected - 2),
                 "AA-only pooling mask has wrong residue count")
    weight = mask.unsqueeze(-1).to(hidden.dtype)
    return (hidden * weight).sum(dim=1) / weight.sum(dim=1)


class AuthorEncoder:
    def __init__(self, device: str, precision: str, batch_size: int, peak_limit: int):
        import torch
        from transformers import BertModel, BertTokenizerFast

        self.torch = torch
        self.device = device
        self.precision = precision
        self.batch_size = batch_size
        self.peak_limit = peak_limit
        self.tokenizer = BertTokenizerFast.from_pretrained(
            str(SOURCE), do_lower_case=False, local_files_only=True)
        model, loading = BertModel.from_pretrained(
            str(SOURCE), add_pooling_layer=False, local_files_only=True,
            use_safetensors=False, output_loading_info=True)
        base.require(not loading["missing_keys"] and not loading["mismatched_keys"]
                     and not loading["error_msgs"],
                     f"Encoder weights missing/mismatched: {loading}")
        allowed = {"bert.pooler.dense.bias", "bert.pooler.dense.weight",
                   "classifier.bias", "classifier.weight"}
        base.require(set(loading["unexpected_keys"]) == allowed,
                     f"Unexpected checkpoint keys differ: {loading['unexpected_keys']}")
        self.load_diagnostics = loading
        self.model = model.eval().requires_grad_(False).to(device)
        if precision == "fp16":
            base.require(device == "cuda", "fp16 encoder requires CUDA")
            self.model.half()

    def embed(self, sequences: list[str]):
        import numpy as np
        torch = self.torch
        result = []
        offset = 0
        while offset < len(sequences):
            size = min(self.batch_size, len(sequences) - offset)
            try:
                subset = sequences[offset:offset + size]
                encoded = self.tokenizer(
                    [" ".join(s) for s in subset], return_tensors="pt",
                    padding=True, truncation=False)
                base.require(encoded["input_ids"].shape[1] <= 64,
                             "Tokenizer output exceeds 64 tokens")
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                with torch.inference_mode():
                    hidden = self.model(**encoded).last_hidden_state
                    pooled = residue_mean(hidden, encoded["attention_mask"], subset)
                result.append(pooled.float().cpu().numpy())
                offset += size
                if self.device == "cuda":
                    torch.cuda.synchronize()
                    peak = torch.cuda.max_memory_reserved() / (1024 ** 2)
                    base.require(peak <= self.peak_limit,
                                 f"GPU reserved {peak:.0f} MiB exceeds {self.peak_limit}")
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if size <= 1:
                    raise
                self.batch_size = max(1, size // 2)
                print(f"CUDA OOM: reducing author BERT batch to {self.batch_size}",
                      flush=True)
        return np.concatenate(result, axis=0)


def smoke(model_spec: dict) -> None:
    import numpy as np
    import torch
    from transformers import BertForSequenceClassification

    samples = [_private_literal('private_7cee5a7ae88192c9'), _private_literal('private_48769fcd2cfa432e'), _private_literal('private_afe396ab2995c4e1')]
    encoder = AuthorEncoder("cpu", "fp32", 3, 12288)
    full = BertForSequenceClassification.from_pretrained(
        str(SOURCE), local_files_only=True, use_safetensors=False)
    full.eval().requires_grad_(False)
    inputs = encoder.tokenizer(
        [" ".join(s) for s in samples], return_tensors="pt",
        padding=True, truncation=False)
    with torch.inference_mode():
        backbone = full.bert(**inputs).last_hidden_state
        direct = residue_mean(backbone, inputs["attention_mask"], samples)
    vectors = encoder.embed(samples)
    difference = float(np.max(np.abs(direct.numpy() - vectors)))
    base.require(vectors.shape == (3, 768)
                 and np.isfinite(vectors).all() and difference < 1e-6,
                 f"Direct dual-task backbone comparison failed: {difference}")
    result = {
        "status": "PASS", "model_fingerprint": base.json_hash(model_spec),
        "load_diagnostics": encoder.load_diagnostics,
        "direct_backbone_max_abs_difference": difference,
        "sample_vector_sha256": hashlib.sha256(vectors.tobytes()).hexdigest(),
        "sample_aa": samples,
        "shape": list(vectors.shape),
        "nonfinite": 0,
    }
    base.atomic_json(OUT / "cpu_smoke.json", result)
    print(json.dumps(result, sort_keys=True), flush=True)


def check_freeze(n: int, input_sha: str, model_spec: dict, model_sha: str) -> None:
    frozen = json.loads((OUT / "model_freeze.json").read_text())
    base.require(frozen["status"] == "frozen"
                 and frozen["n_total"] == n
                 and frozen["input_sha256"] == input_sha
                 and frozen["model_fingerprint"] == model_sha
                 and frozen["model_spec"] == model_spec,
                 "Author BERT frozen provenance mismatch")
    smoke_receipt = json.loads((OUT / "cpu_smoke.json").read_text())
    base.require(smoke_receipt["status"] == "PASS"
                 and smoke_receipt["model_fingerprint"] == model_sha,
                 "Author BERT CPU smoke mismatch")


def precision_check(sequences, vectors, peak_limit):
    import numpy as np
    rng = np.random.default_rng(20260930)
    positions = np.sort(rng.choice(len(sequences), size=min(64, len(sequences)),
                                   replace=False))
    cpu = AuthorEncoder("cpu", "fp32", 32, peak_limit)
    reference = cpu.embed([sequences[i] for i in positions])
    observed = np.asarray(vectors[positions], dtype=np.float32)
    den = np.linalg.norm(reference, axis=1) * np.linalg.norm(observed, axis=1)
    base.require(np.all(den > 0), "Zero vector in CPU precision check")
    cos = np.einsum("ij,ij->i", reference, observed) / den
    rel = np.linalg.norm(observed - reference, axis=1) / np.linalg.norm(
        reference, axis=1)
    base.require(np.isfinite(cos).all() and np.isfinite(rel).all(),
                 "Nonfinite CPU precision comparison")
    base.require(float(cos.min()) >= 0.999 and float(rel.max()) <= 0.02,
                 f"FP16/FP32 precision outside gate: min cosine={cos.min()}, max relative L2={rel.max()}")
    return {"rows": len(positions), "cosine_min": float(cos.min()),
            "cosine_median": float(np.median(cos)),
            "relative_l2_max": float(rel.max()),
            "relative_l2_median": float(np.median(rel))}


def pilot(args, n, input_sha, model_spec, model_sha, peak_limit):
    import numpy as np
    pilot_dir = OUT / "pilot"
    pilot_dir.mkdir(exist_ok=True)
    started = time.time()
    indices, sequences, strata = base.stratified_pilot(
        INPUT, n, args.pilot_rows, seed=20260929)
    base.preflight_gpu(args.device)
    backend = AuthorEncoder(args.device, args.precision, args.batch_size, peak_limit)
    if args.device == "cuda":
        backend.torch.cuda.reset_peak_memory_stats()
    start = time.time()
    vectors = backend.embed(sequences)
    sec = time.time() - start
    base.require(vectors.shape == (len(indices), 768)
                 and np.isfinite(vectors).all(), "Pilot shape or finite failure")
    compare = (precision_check(sequences, vectors, peak_limit)
               if args.device == "cuda" else None)
    fp16 = vectors.astype(np.float16)
    base.require(np.isfinite(fp16).all(), "FP16 pilot nonfinite")
    idx_sha = base.atomic_npy(pilot_dir / "seq_idx.npy", indices)
    vec_sha = base.atomic_npy(pilot_dir / "vectors_fp16.npy", fp16)
    peak = (backend.torch.cuda.max_memory_reserved() / 1024 ** 2
            if args.device == "cuda" else 0.0)
    receipt = {
        "status": "complete", "model_fingerprint": model_sha,
        "input_sha256": input_sha, "n_total": n,
        "requested_rows": args.pilot_rows, "selected_rows": len(indices),
        "selection": strata, "seq_idx_sha256": idx_sha,
        "vectors_sha256": vec_sha, "shape": list(fp16.shape),
        "dtype": "float16", "nonfinite": 0,
        "initial_batch_size": args.batch_size,
        "final_batch_size": backend.batch_size,
        "tokenizer_workers": args.tokenizer_workers,
        "gpu_peak_reserved_mib": round(peak, 1),
        "gpu_peak_limit_mib": peak_limit,
        "encode_sec": round(sec, 2),
        "seq_per_sec": round(len(indices) / sec, 2),
        "elapsed_sec": round(time.time() - started, 2),
        "precision_comparison": compare,
        "model_spec": model_spec,
    }
    base.atomic_json(pilot_dir / "meta.json", receipt)
    print(f"author BERT pilot complete: {len(indices)} rows; "
          f"{receipt['seq_per_sec']:.1f} seq/s; peak {peak:.0f} MiB; "
          f"CPU comparison {compare}", flush=True)


def full(args, n, input_sha, input_stat, model_spec, model_sha, peak_limit):
    import numpy as np
    meta_path = OUT / "meta.json"
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        base.require(old.get("input_sha256") == input_sha
                     and old.get("model_fingerprint") == model_sha
                     and old.get("shard_rows") == args.shard_rows
                     and old.get("n_total") == n,
                     "Existing author BERT provenance differs")
        if old.get("status") == "complete":
            bank = OUT / "bank_fp16.npy"
            base.require(bank.is_file() and base.sha256(bank) == old["bank_sha256"],
                         "Published author BERT bank checksum differs")
            print(f"Already complete: {bank}", flush=True)
            return
    started = time.time()
    meta = {
        "status": "running", "n_total": n, "row_limit": n,
        "input_path": str(INPUT), "input_sha256": input_sha,
        "input_stat": list(input_stat), "model_fingerprint": model_sha,
        "model_spec": model_spec, "shard_rows": args.shard_rows,
        "bank_shape": [n, 768], "bank_dtype": "float16",
        "gpu_peak_limit_mib": peak_limit,
        "initial_batch_size": args.batch_size,
        "tokenizer_workers": args.tokenizer_workers,
    }
    base.atomic_json(meta_path, meta)
    backend = None
    done = 0
    skipped = 0
    for start, end, sequences in base.iter_shards(INPUT, args.shard_rows, n):
        if base.verify_shard(OUT, start, end, 768, input_sha, model_sha):
            skipped += 1
            done = end
            print(f"verified existing shard {start}:{end}", flush=True)
            continue
        if backend is None:
            base.preflight_gpu(args.device)
            backend = AuthorEncoder(args.device, args.precision,
                                    args.batch_size, peak_limit)
            if args.device == "cuda":
                backend.torch.cuda.reset_peak_memory_stats()
        vectors = backend.embed(sequences)
        base.require(vectors.shape == (end - start, 768)
                     and np.isfinite(vectors).all(),
                     f"Full shard shape/nonfinite at {start}:{end}")
        peak = (backend.torch.cuda.max_memory_reserved() / 1024 ** 2
                if args.device == "cuda" else 0.0)
        base.write_shard(OUT, start, end, vectors, input_sha, model_sha,
                         round(peak, 1), 0)
        done = end
        print(f"completed shard {start}:{end}; peak_reserved={peak:.0f} MiB; "
              f"elapsed={time.time()-started:.1f}s", flush=True)
    base.require((INPUT.stat().st_size, INPUT.stat().st_mtime_ns) == input_stat,
                 "Union input changed during encoding")
    meta.update({
        "status": "shards_complete", "n_done": done,
        "shards_verified_on_resume": skipped,
        "nonstandard_junction_warnings_total": 0,
        "elapsed_sec": round(time.time() - started, 2),
        "final_batch_size": backend.batch_size if backend else args.batch_size,
        "gpu_peak_reserved_mib": round(
            backend.torch.cuda.max_memory_reserved() / 1024 ** 2, 1)
            if backend and args.device == "cuda" else 0.0,
    })
    base.atomic_json(meta_path, meta)
    digest = base.publish_bank(OUT, n, 768, args.shard_rows, input_sha, model_sha)
    stat = (OUT / "bank_fp16.npy").stat()
    meta.update({"status": "complete", "bank_sha256": digest,
                 "bank_size_bytes": stat.st_size, "bank_mtime_ns": stat.st_mtime_ns})
    base.atomic_json(meta_path, meta)
    print(f"published {OUT / 'bank_fp16.npy'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--smoke", action="store_true")
    group.add_argument("--pilot", action="store_true")
    group.add_argument("--full", action="store_true")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--tokenizer-workers", type=int, default=4)
    parser.add_argument("--shard-rows", type=int, default=50_000)
    parser.add_argument("--pilot-rows", type=int, default=50_000)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--precision", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()
    policy = json.loads((ROOT / "config" / "resource_policy.json").read_text())
    base.require(policy["embedding_shard_rows_min"] <= args.shard_rows
                 <= policy["embedding_shard_rows_max"], "shard size outside policy")
    base.require(1 <= args.tokenizer_workers
                 <= policy["embedding_tokenizer_workers_max"],
                 "tokenizer worker count outside policy")
    base.require(args.batch_size > 0 and args.pilot_rows > 0,
                 "batch and pilot size must be positive")
    os.environ["RAYON_NUM_THREADS"] = str(args.tokenizer_workers)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["OMP_NUM_THREADS"] = str(min(args.tokenizer_workers, 8))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "shards").mkdir(exist_ok=True)
    n, input_sha, input_stat = base.input_info(INPUT)
    model_spec = build_model_spec()
    model_sha = base.json_hash(model_spec)
    if args.freeze:
        base.require(not (OUT / "meta.json").exists()
                     and not (OUT / "pilot" / "meta.json").exists(),
                     "Freeze must precede pilot/full")
        receipt = {"status": "frozen", "n_total": n, "input_sha256": input_sha,
                   "input_stat": list(input_stat), "model_fingerprint": model_sha,
                   "model_spec": model_spec,
                   "pooling_source": "Wu et al. 2024 Methods p.15",
                   "scientific_role": "primary TCR-BERT comparator, set before pN outcome"}
        frozen_path = OUT / "model_freeze.json"
        if frozen_path.exists():
            base.require(json.loads(frozen_path.read_text()) == receipt,
                         "Existing model freeze differs; never overwrite it")
        else:
            base.atomic_json(frozen_path, receipt)
        print(f"frozen {MODEL_NAME}: {model_sha}", flush=True)
        return
    if args.smoke:
        frozen = json.loads((OUT / "model_freeze.json").read_text())
        base.require(frozen["status"] == "frozen"
                     and frozen["input_sha256"] == input_sha
                     and frozen["model_fingerprint"] == model_sha
                     and frozen["model_spec"] == model_spec,
                     "CPU smoke model freeze mismatch")
        smoke(model_spec)
        return
    check_freeze(n, input_sha, model_spec, model_sha)
    peak_limit = int(policy["gpu_peak_mib_target"])
    with base.gpu_lock(ROOT / ".gpu.lock", MODEL_NAME,
                       "full" if args.full else "pilot"):
        if args.pilot:
            pilot(args, n, input_sha, model_spec, model_sha, peak_limit)
        else:
            full(args, n, input_sha, input_stat, model_spec, model_sha, peak_limit)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"AUTHOR BERT EMBEDDING FAILED: {type(exc).__name__}: {exc}",
              file=sys.stderr)
        raise
