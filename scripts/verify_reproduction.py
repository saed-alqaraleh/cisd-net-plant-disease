#!/usr/bin/env python3
"""Offline integrity and numeric replay of all frozen CISD-Net study runs.

Only numpy and repository-local recorded evidence are required. The input
archives are first authenticated by SHA-256, before numeric arrays are read.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
NONTRACKED={'__pycache__','.git','.pytest_cache','.mypy_cache','.ruff_cache','data','outputs','.venv'}
METRICS=('top1_acc','macro_f1','q1_acc','q2_acc','q3_acc','q4_acc',
         'lac_coverage','lac_set_size','mod_coverage','mod_set_size','aurcc')

def sha256(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def all_tracked_files():
    return {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*')
            if p.is_file() and not (set(p.relative_to(ROOT).parts)&NONTRACKED)
            and not p.name.endswith(('.pyc','.pyo'))}

def verify_hashes():
    entries={}
    for line in (ROOT/'SHA256SUMS.txt').read_text().splitlines():
        if not line: continue
        if '  ' not in line: raise AssertionError(f'Malformed checksum line: {line}')
        digest,path=line.split('  ',1)
        if path in entries or not re.fullmatch(r'[a-f0-9]{64}',digest):
            raise AssertionError(f'Invalid checksum entry: {path}')
        entries[path]=digest
    assert 'SHA256SUMS.txt' not in entries, 'Self-referential hash entry'
    actual=all_tracked_files()-{'SHA256SUMS.txt'}
    if actual != set(entries):
        raise AssertionError(f'Unlisted files: {sorted(actual-set(entries))}; missing: {sorted(set(entries)-actual)}')
    for rel,expected in entries.items():
        got=sha256(ROOT/rel)
        if got!=expected: raise AssertionError(f'SHA-256 mismatch: {rel}')
    print(f'SHA256SUMS: OK ({len(entries)} tracked files)')

def macro_f1(y,p,num_classes):
    f=[]
    for c in range(num_classes):
        tp=np.sum((y==c)&(p==c));fp=np.sum((y!=c)&(p==c));fn=np.sum((y==c)&(p!=c))
        if np.sum(y==c)>0:
            f.append(float(2*tp/(2*tp+fp+fn)) if (2*tp+fp+fn)>0 else 0.0)
    return float(np.mean(f)) if f else 0.0

def aurcc(probs,y):
    conf=np.max(probs,axis=1)
    pred=np.argmax(probs,axis=1)
    err=(pred!=y).astype(np.float64)
    order=np.argsort(-conf,kind='mergesort')
    return float(np.mean(np.cumsum(err[order])/np.arange(1,len(err)+1,dtype=np.float64)))

def conformal(a,alpha=0.05):
    yv=a['val_y_true'];yc=a['cal_y_true'];yt=a['test_y_true']
    pv=a['val_probs'].astype(np.float64);pc=a['cal_probs'].astype(np.float64);pt=a['test_probs'].astype(np.float64)
    cv=a['val_clutter_scores'].astype(np.float64);cc=a['cal_clutter_scores'].astype(np.float64);ct=a['test_clutter_scores'].astype(np.float64)
    cmin=float(np.min(cv));denom=max(float(np.max(cv))-cmin,1e-8)
    nv=np.clip((cv-cmin)/denom,0.0,1.0)
    nc=np.clip((cc-cmin)/denom,0.0,1.0)
    nt=np.clip((ct-cmin)/denom,0.0,1.0)
    k=min(max(math.ceil((len(yc)+1)*(1.0-alpha))-1,0),len(yc)-1)
    vl=np.arange(len(yv));cal=np.arange(len(yc))
    q_l=float(np.sort(1.0-pc[cal,yc],kind='mergesort')[k])
    lac=(1.0-pt)<=q_l
    empty=~lac.any(axis=1)
    if empty.any(): lac[empty,np.argmax(pt[empty],axis=1)]=True
    best_gamma=0.0;best_size=1e9
    kv=min(max(math.ceil((len(yv)+1)*(1.0-alpha))-1,0),len(yv)-1)
    for gamma in (0.0,0.05,0.10,0.20):
        s=(1.0-pv[vl,yv])/(1.0+gamma*nv)
        q=float(np.sort(s,kind='mergesort')[kv])
        sets=((1.0-pv)/(1.0+gamma*nv[:,None]))<=q
        sz=float(np.mean(np.sum(sets,axis=1)))
        if sz<best_size: best_size=sz;best_gamma=gamma
    qm=float(np.sort((1.0-pc[cal,yc])/(1.0+best_gamma*nc),kind='mergesort')[k])
    mod=((1.0-pt)/(1.0+best_gamma*nt[:,None]))<=qm
    empty=~mod.any(axis=1)
    if empty.any(): mod[empty,np.argmax(pt[empty],axis=1)]=True
    return dict(sets_lac=lac,sets_mod=mod,lac_qhat=q_l,mod_qhat=qm,mod_gamma_val=best_gamma,
        lac_coverage=float(np.mean(lac[np.arange(len(yt)),yt])),
        mod_coverage=float(np.mean(mod[np.arange(len(yt)),yt])),
        lac_set_size=float(np.mean(np.sum(lac,axis=1))),
        mod_set_size=float(np.mean(np.sum(mod,axis=1))))

def numeric_run(a):
    y=a['test_y_true'];p=a['test_probs'];clutter=a['test_clutter_scores'];pred=np.argmax(p,axis=1)
    q=np.quantile(clutter,[0.0,0.25,0.50,0.75,1.0])
    metrics={"top1_acc":float(np.mean(pred==y)),"macro_f1":macro_f1(y,pred,p.shape[1]),
             "aurcc":aurcc(p.astype(np.float64),y)}
    for k in range(4):
        mask=((clutter>=q[k])&(clutter<(q[k+1]))) if k<3 else ((clutter>=q[k])&(clutter<=q[k+1]))
        metrics[f'q{k+1}_acc']=float(np.mean(pred[mask]==y[mask])) if mask.any() else metrics['top1_acc']
    return metrics

def assert_close(got,expected,label,tol=1e-12):
    if not np.isclose(float(got),float(expected),rtol=0,atol=tol):
        raise AssertionError(f'{label}: {got} differs from frozen {expected}')

def replay():
    ledger=json.loads((ROOT/'results/v3_run_ledger_and_replay_audit.json').read_text())
    assert ledger['total_runs_executed']==99
    assert ledger['conformal_sets_reproducible_runs_count']=='99/99'
    records=ledger['per_run_records']; assert len(records)==99
    files=sorted((ROOT/'predictions_full_precision').glob('*.npz'))
    assert len(files)==99, len(files)
    expected_names={r['prediction_file'].split('/')[-1] for r in records}
    assert expected_names=={p.name for p in files}
    extracted={}
    for r in records:
        tag=f"{r['protocol']}/{r['config_name']}/seed{r['seed']}"
        fn=ROOT/r['prediction_file']
        assert sha256(fn)==r['prediction_sha256'],f'Ledger file hash: {tag}'
        # Only use numerical arrays; do not unpickle object dtype image-ID arrays.
        with np.load(fn,allow_pickle=False) as a:
            numeric=numeric_run(a)
            conf=conformal(a)
            for k,v in numeric.items(): assert_close(v,r[k],tag+'/'+k)
            for k in ('lac_coverage','lac_set_size','mod_coverage','mod_set_size','mod_gamma_val'):
                assert_close(conf[k],r[k],tag+'/'+k)
            for k in ('lac','mod'):
                assert np.array_equal(conf['sets_'+k],a['test_sets_'+k]),f'Conformal sets: {tag}/{k}'
                assert_close(conf[k+'_qhat'],float(a[k+'_qhat'][0]),tag+'/'+k+'_qhat')
            assert_close(conf['mod_gamma_val'],float(a['mod_gamma'][0]),tag+'/stored_gamma')
        extracted[(r['protocol'],r['config_name'],r['seed'])]={**numeric,**conf}
    assert len(extracted)==99
    key_map={
        'top1_acc':['top1_mean','top1_std'],
        'macro_f1':['macro_f1_mean','macro_f1_std'],
        'q1_acc':['q1_acc_mean'], 'q4_acc':['q4_acc_mean'],
        'lac_coverage':['lac_coverage_mean'], 'lac_set_size':['lac_set_size_mean'],
        'mod_coverage':['mod_coverage_mean'], 'mod_set_size':['mod_set_size_mean'],
        'aurcc':['aurcc_mean']}
    for proto,configs in ledger['aggregated_3seed_results'].items():
        assert len(configs)==11
        for config,agg in configs.items():
            rs=[extracted[(proto,config,seed)] for seed in (42,108,2026)]
            for metric, fields in key_map.items():
                x=np.array([r[metric] for r in rs])
                assert_close(x.mean(),agg[fields[0]],f'{proto}/{config}/{fields[0]}')
                if len(fields)==2: assert_close(x.std(),agg[fields[1]],f'{proto}/{config}/{fields[1]}')
    # Manifest leakage checks: split membership and group-disjoint FP validation.
    import csv
    with (ROOT/'manifests/fieldplant_FP_grouped_manifest.csv').open(newline='') as f: fp=list(csv.DictReader(f))
    assert len(fp)==1273
    groups={}
    for row in fp:
        key=row['stem_group']; split=row['split_FP_grouped']
        if key in groups: assert groups[key]==split,f'FP stem_group leakage: {key}'
        groups[key]=split
    with (ROOT/'manifests/plantdoc_P1_P2_manifest.csv').open(newline='') as f: pd=list(csv.DictReader(f))
    assert len(pd)==2185
    assert len({r['image_id'] for r in pd})==2185
    print('Evidence: 99/99 complete; 99/99 exact LAC + modulated conformal set replay')
    print('Ledgers: numeric Top-1/Macro-F1/Q1-Q4/AURCC, quantiles, coverage, sizes, means/SDs match')
    print('Manifests: 2,185 PlantDoc images; 1,273 FieldPlant images; zero FP stem_group overlap')
    print('Note: HSIC, orthogonality and checkpoint bytes require external trained-state evidence;')
    print('      their archived scalar values are retained in the authenticated ledger, not recalculated here.')
    print()
    print('Protocol | Configuration              | Top-1%  | Macro-F1% | Q4%   | LAC size (coverage)')
    for proto in ('P1','P2','FP'):
        for cfg,aggregate in ledger['aggregated_3seed_results'][proto].items():
            print(f"{proto:8s} | {cfg:26s} | {100*aggregate['top1_mean']:6.2f} | "
                  f"{100*aggregate['macro_f1_mean']:9.2f} | {100*aggregate['q4_acc_mean']:5.2f} | "
                  f"{aggregate['lac_set_size_mean']:5.2f} ({100*aggregate['lac_coverage_mean']:4.1f}%)")
    print('VERIFICATION PASSED')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--skip-hash',action='store_true',help='Only for developer smoke testing; not for release verification')
    a=p.parse_args()
    if not a.skip_hash: verify_hashes()
    replay()

if __name__=='__main__':
    try: main()
    except (AssertionError,ValueError,KeyError) as e:
        print(f'VERIFICATION FAILED: {e}',file=sys.stderr)
        sys.exit(1)
