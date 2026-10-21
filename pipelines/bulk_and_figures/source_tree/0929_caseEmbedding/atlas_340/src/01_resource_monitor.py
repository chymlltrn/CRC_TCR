#!/usr/bin/env python3
from release_runtime import private_literal as _private_literal, release_path as _release_path
import csv
import json
import os
import sys
import time
from pathlib import Path
import psutil

pid=int(sys.argv[1])
root=Path(_release_path('path_99d8c1c6fde2a6b9'))
samples=root/'audit/01_resource_samples.tsv'
receipt=root/'audit/01_resource_receipt.json'
process=psutil.Process(pid)
start=time.time()
peak_rss=0
peak_processes=0
n=0
with samples.open('w',newline='') as fh:
    writer=csv.writer(fh,delimiter='\t')
    writer.writerow(['unix_time','n_processes','aggregate_rss_bytes','aggregate_cpu_percent'])
    while process.is_running() and process.status()!=psutil.STATUS_ZOMBIE:
        procs=[process]+process.children(recursive=True)
        rss=0
        cpu=0.0
        alive=0
        for proc in procs:
            try:
                with proc.oneshot():
                    rss+=proc.memory_info().rss
                    cpu+=proc.cpu_percent(interval=None)
                    alive+=1
            except (psutil.NoSuchProcess,psutil.AccessDenied):
                continue
        peak_rss=max(peak_rss,rss)
        peak_processes=max(peak_processes,alive)
        writer.writerow([int(time.time()),alive,rss,round(cpu,1)])
        fh.flush()
        n+=1
        time.sleep(2)
tmp=receipt.with_suffix('.partial')
tmp.write_text(json.dumps({'schema':'atlas340_stage1_resource_v1','pid':pid,
    'samples':n,'peak_aggregate_rss_bytes':peak_rss,
    'peak_aggregate_rss_gib':round(peak_rss/2**30,3),
    'peak_processes':peak_processes,
    'monitor_wall_s':round(time.time()-start,1),
    'cpu_limit_logical':16,'gpu_used':False,
    'monitor_scope':'ingestion driver and descendant processes; sampled every 2s',
    'samples_path':str(samples)},indent=2)+'\n')
os.replace(tmp,receipt)
