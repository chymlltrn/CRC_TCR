#!/usr/bin/env python3
"""Run a separate SCEPTR graph after the existing TCR-BERT run is accepted."""
from release_runtime import private_literal as _private_literal, release_path as _release_path
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(_release_path('path_fd724b65e07400d1'))
CURRENT = None
STATE = {}


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, obj):
    path = Path(path)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def running_scripts(root, proc_root=Path('/proc')):
    """Detect the previous runner, its stages and finalizer without importing BLAS."""
    found = []
    prefix = str(root / 'src') + '/'
    for p in proc_root.iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / 'cmdline').read_bytes().split(b'\0')
            if any(a.decode(errors='replace').startswith(prefix) for a in args):
                found.append(int(p.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return found


def predecessor_status(base):
    state_file = base / 'audit/runner_state.json'
    if not state_file.exists():
        return False, 'WAITING_PREDECESSOR_STATE'
    state = read(state_file)
    if state.get('status') != 'COMPLETE':
        return False, 'WAITING_PREDECESSOR_' + str(state.get('status', 'UNKNOWN'))
    report = base / 'report/full_cluster_counts_by_resolution.json'
    if not report.exists() or read(report).get('status') != 'PASS':
        return False, 'WAITING_PREDECESSOR_FINAL_ACCEPTANCE'
    if not (base / 'report/FINAL_COUNTS.md').is_file():
        return False, 'WAITING_PREDECESSOR_FINAL_REPORT'
    if running_scripts(base):
        return False, 'WAITING_PREDECESSOR_PROCESSES'
    return True, 'READY'


def update():
    dump(ROOT / 'audit/runner_state.json', STATE)
    lines = ['# SCEPTR 全量 Leiden 运行状态', '', '更新：' + stamp(), '',
             '**状态：' + STATE['status'] + '**', '',
             '范围：28,380,380 条 unique CDR3β AA；独立索引、独立图。',
             'k=30；主分辨率1、种子17/29；尺度敏感性0.5/2、种子17；Leiden迭代2次。',
             '512个相同的长度/来源分层查询，对全库精确top30验收平均严格ID召回≥0.95。', '',
             '| 阶段 | 状态 | 秒 |', '|---|---|---:|']
    for row in STATE.get('stages', []):
        lines.append('|{}|{}|{:.1f}|'.format(row['stage'], row['status'], row.get('seconds', 0)))
    if STATE.get('wait_reason'):
        lines += ['', '等待原因：' + STATE['wait_reason']]
    for rel in ['report/FINAL_COUNTS.md', 'report/MODEL_COMPARISON.md']:
        if (ROOT / rel).exists():
            lines += ['', '[' + Path(rel).name + '](' + Path(rel).name + ')']
    lines += ['', '本轮仅无标签图与聚类；未估计N+贡献或抗原结合。联合Home/External图谱用于探索比较。']
    (ROOT / 'report/CURRENT_STATUS.md').write_text('\n'.join(lines) + '\n')


def stop_child():
    if CURRENT is None or CURRENT.poll() is not None:
        return
    try:
        os.killpg(CURRENT.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        CURRENT.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(CURRENT.pid, signal.SIGKILL)
        CURRENT.wait(timeout=30)


def stop(signum, frame):
    STATE.update(status='STOPPED', ended_utc=stamp(), signal=signum)
    update()
    stop_child()
    raise SystemExit(128 + signum)


def source_gate(contract):
    for name, expected in contract['execution_source_sha256'].items():
        if sha(ROOT / 'src' / name) != expected:
            raise RuntimeError('Frozen execution source hash changed: ' + name)


def environment(threads):
    env = os.environ.copy()
    for key in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS']:
        env[key] = str(threads)
    env.update(ATLAS_CPU_THREADS=str(threads), OMP_WAIT_POLICY='PASSIVE',
               PYTHONPATH=str(ROOT / 'env/vendor'), PYTHONDONTWRITEBYTECODE='1',
               TMPDIR=str(ROOT / 'env/tmp'), XDG_CACHE_HOME=str(ROOT / 'env/cache'),
               MPLCONFIGDIR=str(ROOT / 'env/matplotlib'))
    return env


def run_stage(name, args, threads, contract):
    global CURRENT
    import psutil  # Imports only once the predecessor has finished.
    source_gate(contract)
    affinity = sorted(os.sched_getaffinity(0))[:threads]
    if len(affinity) != threads:
        raise RuntimeError('Insufficient allowed CPUs for stage')
    row = dict(stage=name, status='RUNNING', started_utc=stamp(), threads_limit=threads,
               argv=args, peak_total_rss_bytes=0)
    STATE['stages'].append(row)
    update()
    began = time.monotonic()
    def preexec():
        os.sched_setaffinity(0, affinity)
    with (ROOT / 'logs' / ('serial_' + name + '.log')).open('a') as log:
        CURRENT = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=environment(threads), start_new_session=True,
                                   preexec_fn=preexec)
        row['child_pid'] = CURRENT.pid
        while CURRENT.poll() is None:
            # Do not overlap a newly started TCR-BERT repair with our computation.
            if running_scripts(Path(contract['reference_root'])):
                raise RuntimeError('Predecessor restarted; stopping own child to preserve resource limits')
            try:
                parent = psutil.Process(CURRENT.pid)
                rss = 0
                for p in [parent] + parent.children(recursive=True):
                    try:
                        rss += p.memory_info().rss
                    except psutil.NoSuchProcess:
                        continue
                rss += psutil.Process().memory_info().rss
                row['peak_total_rss_bytes'] = max(row['peak_total_rss_bytes'], rss)
                if rss > contract['max_rss_bytes']:
                    raise RuntimeError('Total runner/child RSS exceeded approved 512GB ceiling')
            except psutil.NoSuchProcess:
                pass
            row['seconds'] = time.monotonic() - began
            update()
            time.sleep(10)
        row.update(seconds=time.monotonic() - began, exit_code=CURRENT.returncode,
                   ended_utc=stamp(), status='PASS' if CURRENT.returncode == 0 else 'FAIL')
        update()
        if CURRENT.returncode != 0:
            raise RuntimeError('Stage failed: ' + name)
    source_gate(contract)


def main():
    global STATE
    lock = (ROOT / 'audit/runner.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = read(ROOT / 'config/comparison_contract.json')
    old = ROOT / 'audit/runner_state.json'
    if old.exists() and read(old).get('status') == 'COMPLETE':
        print('Already complete; no duplicate run.', flush=True)
        return
    if old.exists():
        snapshot = ROOT / 'audit' / ('runner_state_before_resume_' + str(time.time_ns()) + '.json')
        snapshot.write_bytes(old.read_bytes())
    STATE = dict(status='QUEUED', started_utc=stamp(), pid=os.getpid(),
                 model='sceptr_cdr3_only', n_vertices=28380380, stages=[],
                 reference_root=contract['reference_root'], gpu_used=False)
    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, stop)
    try:
        source_gate(contract)
        while True:
            ready, reason = predecessor_status(Path(contract['reference_root']))
            if ready:
                break
            if STATE.get('wait_reason') != reason:
                STATE.update(status='QUEUED', wait_reason=reason)
                update()
                print(stamp(), reason, flush=True)
            time.sleep(30)
        STATE.update(status='RUNNING', activated_utc=stamp())
        STATE.pop('wait_reason', None)
        update()
        graph = [sys.executable, str(ROOT / 'src/full_leiden_graph.py'), '--threads', '48']
        check = [sys.executable, str(ROOT / 'src/validate_outputs.py')]
        stages = [
            ('source_verification', check + ['source'], 4),
            ('smoke', graph[:-1] + ['4', '--stage', 'smoke'], 4),
            ('index_full', graph + ['--stage', 'index'], 48),
            ('knn_full', graph + ['--stage', 'knn'], 48),
            ('graph_full', graph + ['--stage', 'graph'], 48),
            ('leiden_primary', graph + ['--stage', 'leiden', '--resolutions', '1', '--repeat-primary'], 48),
            ('leiden_scale_counts', graph + ['--stage', 'leiden', '--resolutions', '0.5,2'], 48),
            ('final_acceptance_and_comparison', check + ['finalize'], 4),
        ]
        for name, argv, threads in stages:
            run_stage(name, argv, threads, contract)
        STATE.update(status='COMPLETE', completed_utc=stamp())
        update()
        print(stamp(), 'COMPLETE', flush=True)
    except SystemExit:
        stop_child()
        raise
    except BaseException as exc:
        stop_child()
        STATE.update(status='FAILED', failed_utc=stamp(), error=repr(exc), traceback=traceback.format_exc())
        update()
        print(STATE['traceback'], flush=True)
        raise


if __name__ == '__main__':
    main()
