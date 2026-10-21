#!/usr/bin/env bash
set -uo pipefail
ROOT="${CRC_TCR_ATLAS_ROOT:?Set CRC_TCR_ATLAS_ROOT to an isolated atlas runtime}"
PY_SCRIPT="$ROOT/src/08_compare_mil.py"
RUNS="$ROOT/representation_comparison/mil/runs"
mkdir -p "$RUNS"
exec 9>"$RUNS/.queue.lock"
flock -n 9 || { echo "Comparison queue already running"; exit 1; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=0
printf 'queue_started_utc=%s pid=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$"
for model in cdr3vec_small kmer3_cosine_svd tcrbert_dualtask_aa_mean sceptr_cdr3_only; do
  if [[ "$model" != kmer3_cosine_svd ]]; then
    waits=0
    while ! python3 - "$ROOT" "$model" <<'PY' >/dev/null 2>&1
import json, pathlib, sys
root=pathlib.Path(sys.argv[1])
model=sys.argv[2]
p=root/'representation_comparison'/'vectors'/model
r=json.loads((p/'receipt.json').read_text())
if r.get('status')!='complete' or not (p/'bank_fp16.npy').is_file():
    raise RuntimeError('aligned bank not complete')
PY
    do
      if (( waits % 10 == 0 )); then
        printf '%s waiting_for_aligned_bank=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model"
      fi
      sleep 60
      waits=$((waits+1))
    done
  fi
  started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  printf '%s starting_model=%s\n' "$started" "$model"
  python3 -u "$PY_SCRIPT" --model "$model" --phase all --device cuda --threads 8 >"$RUNS/$model.log" 2>&1
  rc=$?
  python3 - "$ROOT" "$model" "$started" "$rc" <<'PY'
import datetime, hashlib, json, pathlib, sys
import pandas as pd
root=pathlib.Path(sys.argv[1]); model=sys.argv[2]; started=sys.argv[3]; rc=int(sys.argv[4])
base=root/'representation_comparison'/'mil'; runs=base/'runs'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''): h.update(block)
    return h.hexdigest()
issues=[]; receipt_paths=[]; final_rows=[]
for outer in range(5):
    for stage in ('inner_0','inner_1','inner_2','final'):
        for seed in (17,29,43):
            d=base/model/f'outer_{outer}'/stage/f'seed_{seed}'
            rp=d/'receipt.json'; pp=d/'prediction.tsv'
            if not rp.exists() or not pp.exists():
                issues.append(str(rp)); continue
            try:
                receipt=json.loads(rp.read_text())
                if (receipt.get('status')!='complete' or
                    receipt.get('prediction_sha256')!=sha(pp) or
                    receipt.get('contract',{}).get('model')!=model):
                    issues.append(str(rp)); continue
                receipt_paths.append(rp)
                if stage=='final':
                    final_rows.append(pd.read_csv(pp,sep='\t',dtype={'patient_key':str}))
            except Exception as exc:
                issues.append(f'{rp}: {exc}')
if final_rows:
    p=pd.concat(final_rows,ignore_index=True)
    for seed in (17,29,43):
        s=p.loc[p.seed.eq(seed)]
        if len(s)!=126 or s.patient_key.nunique()!=126 or sorted(s.recorded_n_binary.value_counts().to_dict().items())!=[(0,70),(1,56)]:
            issues.append(f'final patient coverage invalid seed {seed}')
status='complete' if rc==0 and not issues and len(receipt_paths)==60 else 'failed'
result={
    'status':status, 'model':model, 'started_utc':started,
    'finished_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'exit_code':rc, 'n_expected_fits':60, 'n_verified_fits':len(receipt_paths),
    'n_final_patient_seed_rows':sum(len(x) for x in final_rows),
    'issues':issues[:25],
    'script_sha256':sha(root/'src/08_compare_mil.py'),
    'log_sha256':sha(runs/f'{model}.log'),
    'folds_sha256':sha(root/'folds/folds.tsv'),
    'draw_sha256':sha(root/'representation_comparison/home_sample/patient_aa_draw.parquet'),
}
target=runs/f'{model}.json'; tmp=target.with_suffix('.json.partial')
tmp.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
tmp.replace(target)
print(json.dumps({k:result[k] for k in ('model','status','exit_code','n_verified_fits','n_final_patient_seed_rows')},sort_keys=True),flush=True)
PY
  if (( rc != 0 )) || ! python3 - "$RUNS/$model.json" <<'PY' >/dev/null
import json,sys
assert json.load(open(sys.argv[1]))['status']=='complete'
PY
  then
    printf '%s queue_stopped_after_failure=%s exit_code=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$model" "$rc"
    exit 1
  fi
done
printf 'queue_complete_utc=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
