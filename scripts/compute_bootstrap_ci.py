#!/usr/bin/env python3
"""Paired image-bootstrap CI for Top-1/Q4 difference between two archived runs.

The two comparison records must be from the same protocol and seed. This
supplementary exploratory analysis is not an archived manuscript CI.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def load(protocol,config,seed):
    file=ROOT/'predictions_full_precision'/f'{protocol}_{config}_seed{seed}_predictions.npz'
    ledger=json.loads((ROOT/'results/v3_run_ledger_and_replay_audit.json').read_text())
    record=next((r for r in ledger['per_run_records'] if r['protocol']==protocol
                 and r['config_name']==config and r['seed']==seed),None)
    if record is None:
        raise ValueError(f'Unknown frozen run: {protocol}/{config}/{seed}')
    digest=hashlib.sha256(file.read_bytes()).hexdigest()
    if digest!=record['prediction_sha256']:
        raise ValueError(f'Prediction SHA-256 mismatch: {file}')
    # Original object-dtype image IDs are unpickled only after SHA verification.
    with np.load(file,allow_pickle=True) as x:
        return (x['test_image_ids'].astype(str),x['test_y_true'],
            x['test_probs'].argmax(axis=1),x['test_clutter_scores'])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',choices=['P1','P2','FP'],required=True)
    p.add_argument('--baseline',required=True)
    p.add_argument('--method',default='cisd_net_single')
    p.add_argument('--seed',type=int,choices=[42,108,2026],default=42)
    p.add_argument('--replicates',type=int,default=10000)
    p.add_argument('--bootstrap-seed',type=int,default=20261115)
    a=p.parse_args()
    ids0,y0,p0,c0=load(a.protocol,a.baseline,a.seed)
    ids1,y1,p1,c1=load(a.protocol,a.method,a.seed)
    if not (np.array_equal(ids0,ids1) and np.array_equal(y0,y1)
            and np.array_equal(c0,c1)):
        p.error('Comparisons require identical ordered test cases and clutter scores')
    q=np.quantile(c0,[0,0.25,0.5,0.75,1.0])
    mask=(c0>=q[3])&(c0<=q[4])
    rng=np.random.default_rng(a.bootstrap_seed)
    for label,eligible in [('Overall',np.ones(len(y0),dtype=bool)),('Q4',mask)]:
        diff=(p1[eligible]==y0[eligible]).astype(int)-(p0[eligible]==y0[eligible]).astype(int)
        boot=np.empty(a.replicates,dtype=float)
        for b in range(a.replicates):
            boot[b]=diff[rng.integers(0,len(diff),size=len(diff))].mean()
        l,u=np.percentile(boot,[2.5,97.5])
        print(f'{label}: {100*diff.mean():+.2f} pp (95% percentile CI [{100*l:+.2f}, {100*u:+.2f}] pp; n={len(diff)})')

if __name__=='__main__': main()
