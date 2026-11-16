#!/usr/bin/env python3
"""Compare original frozen numbers_v3.json against a fresh no-pHash recomputation.

All shared numerical fields must agree; original FieldPlant near-duplicate
sensitivity results require the optional earlier pHash manifest.
"""
from pathlib import Path
import json
import math

ROOT = Path(__file__).resolve().parents[1]
source = json.loads((ROOT / 'analysis/numbers_v3.json').read_text())
regenerated = json.loads((ROOT / 'outputs/analysis/numbers_v3.json').read_text())
errors = []
count = 0

def compare(a, b, path):
    global count
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) & set(b)):
            compare(a[key], b[key], f'{path}/{key}')
    elif isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            errors.append(f'{path}: list length {len(a)} != {len(b)}')
        else:
            for idx, (x, y) in enumerate(zip(a, b)):
                compare(x, y, f'{path}[{idx}]')
    elif a is None or b is None:
        if a != b: errors.append(f'{path}: reference {a!r} != generated {b!r}')
    elif isinstance(a, (float, int)) and isinstance(b, (float, int)):
        count += 1
        if not math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=1e-12):
            errors.append(f'{path}: reference {a} != generated {b}')
    elif a != b:
        errors.append(f'{path}: reference {a!r} != generated {b!r}')

for key in ('R', 'PT', 'DATA', 'UNC', 'replay_lac', 'bootstrap'):
    compare(source[key], regenerated[key], key)
    if key in ('R','DATA','UNC') and set(source[key]) != set(regenerated[key]):
        errors.append(f'{key}: mismatched key sets')
missing = sorted(set(source['PT']) - set(regenerated['PT']))
unexpected = sorted(set(regenerated['PT']) - set(source['PT']))
if unexpected: errors.append(f'Unexpected paired contrasts: {unexpected}')
if missing and any(not k.startswith('FPclean|') for k in missing):
    errors.append(f'Unexpected missing contrasts: {missing}')
for key in ('FPLEAK', 'FPCLEAN'):
    if regenerated[key] is not None:
        compare(source[key], regenerated[key], key)
print(f'Matched shared numerical entries: {count}')
print(f'Paired contrasts: reference {len(source["PT"])}; generated {len(regenerated["PT"])}')
if missing:
    print(f'NOT VERIFIED without optional prior pHash manifest: {len(missing)} FPclean contrasts, FPLEAK and FPCLEAN')
if errors:
    for msg in errors[:30]: print('FAIL:', msg)
    raise SystemExit(1)
print('PASS: every comparable numerical entry matches the uploaded reference.')
