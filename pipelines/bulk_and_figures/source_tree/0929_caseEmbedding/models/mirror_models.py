#!/usr/bin/env python3
"""Byte-identical, read-only local mirror of the three actually used checkpoints."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil

ROOT = Path(_release_path('path_1304218e86d2e596'))
MODELS = ("tcrbert_mlm", "sceptr_cdr3_only", "tcrbert_dualtask_aa_mean")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, content):
    temp = path.with_name(path.name + f".partial.{os.getpid()}")
    with temp.open("w") as stream:
        json.dump(content, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temp, path)


def mirror(model):
    embedding = ROOT / "embeddings" / model
    meta = json.loads((embedding / "meta.json").read_text())
    verified = json.loads((embedding / "verification.json").read_text())
    require(meta["status"] == "complete" and verified["status"] == "PASS",
            f"{model}: bank not independently complete")
    for key, receipt_key in (("model_fingerprint", "model_fingerprint"),
                             ("input_sha256", "input_sha256"),
                             ("bank_sha256", "bank_sha256_verified")):
        require(meta[key] == verified[receipt_key],
                f"{model}: {key} verification mismatch")
    spec = meta["model_spec"]
    source = Path(spec["model_path"])
    require(source.is_dir(), f"{model}: checkpoint source missing")
    destination = ROOT / "models" / model
    destination.mkdir(parents=True, exist_ok=True)
    items = []
    for filename, expected_sha in sorted(spec["model_files_sha256"].items()):
        origin = source / filename
        target = destination / filename
        require(origin.is_file(), f"{model}: missing source {filename}")
        actual_source_sha = sha256(origin)
        require(actual_source_sha == expected_sha,
                f"{model}: source checksum changed for {filename}")
        if target.exists():
            require(target.is_file() and sha256(target) == expected_sha,
                    f"{model}: existing mirror checksum differs for {filename}")
        else:
            temp = target.with_name(target.name + f".partial.{os.getpid()}")
            try:
                with origin.open("rb") as incoming, temp.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                    outgoing.flush()
                    os.fsync(outgoing.fileno())
                require(sha256(temp) == expected_sha,
                        f"{model}: copied bytes differ for {filename}")
                os.chmod(temp, 0o444)
                os.replace(temp, target)
            finally:
                if temp.exists():
                    temp.unlink()
        require(sha256(target) == expected_sha,
                f"{model}: final mirror checksum differs for {filename}")
        items.append({
            "filename": filename,
            "source": str(origin),
            "mirror": str(target),
            "sha256": expected_sha,
            "bytes": target.stat().st_size,
            "mode": oct(target.stat().st_mode & 0o777),
        })
    receipt = {
        "status": "PASS",
        "model": model,
        "scientific_role": ("primary TCR-BERT comparator"
                            if model == "tcrbert_dualtask_aa_mean"
                            else "primary SCEPTR comparator"
                            if model == "sceptr_cdr3_only"
                            else "legacy technical sensitivity"),
        "bank_sha256": meta["bank_sha256"],
        "model_fingerprint": meta["model_fingerprint"],
        "input_sha256": meta["input_sha256"],
        "source_revision": spec.get("source_revision"),
        "source_model_path_used_during_encoding": str(source),
        "mirror_model_path": str(destination),
        "mirror_timing": "post-encoding; original source was used at run time, mirror is byte-identical",
        "mirror_tool_sha256": sha256(Path(__file__)),
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "files": items,
    }
    atomic_json(destination / "mirror_receipt.json", receipt)
    print(json.dumps({"model": model, "status": "PASS",
                      "files": len(items),
                      "bytes": sum(x["bytes"] for x in items),
                      "receipt": str(destination / "mirror_receipt.json")}),
          flush=True)
    return receipt


def main():
    receipts = [mirror(model) for model in MODELS]
    manifest = {
        "status": "PASS",
        "models": {item["model"]: {
            "receipt": f"models/{item['model']}/mirror_receipt.json",
            "bank_sha256": item["bank_sha256"],
            "model_fingerprint": item["model_fingerprint"],
            "files": len(item["files"]),
            "bytes": sum(row["bytes"] for row in item["files"]),
        } for item in receipts},
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    atomic_json(ROOT / "models" / "MANIFEST.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
