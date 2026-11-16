#!/usr/bin/env python3
"""Regenerates reported numeric summaries, six table bodies and two data figures from the
v3 evidence archive (99 prediction archives, run ledger, split manifests). No model is run.
Prediction archives are read member by member without unpickling; test images are aligned to the
manifests by order, label and clutter score (verified below).
Usage: python analysis/recompute_v3.py [<prior FP manifest with phash>]"""
import sys, os, json, math, zipfile
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from pathlib import Path
R = str(Path(__file__).resolve().parents[1]); MD = os.path.join(R, 'outputs', 'manuscript')
PH = sys.argv[1] if len(sys.argv) > 1 else None
TB, FG = os.path.join(MD, 'tables'), os.path.join(MD, 'figures')
os.makedirs(TB, exist_ok=True); os.makedirs(FG, exist_ok=True)
os.makedirs(os.path.join(R, 'outputs', 'analysis'), exist_ok=True)
PROT = ['P1', 'P2', 'FP']; SEEDS = [42, 108, 2026]
CFG = ['erm_cls', 'gem_linear_baseline', 'cosine_proto_baseline', 'ablation_unconstrained', 'ablation_qr_decoupled_v1', 'ablation_soft_ortho',
       'ablation_stiefel_no_cpi', 'ablation_stiefel_no_adv', 'cisd_net_single', 'dualstream_concat_ctrl', 'cisd_net_dualstream']
NAME = {'erm_cls': 'Linear probe, full image', 'gem_linear_baseline': 'GeM pooling, 4 views', 'cosine_proto_baseline': 'Cosine prototype, 4 views',
        'dualstream_concat_ctrl': 'Concatenation, 4 views', 'ablation_unconstrained': 'Subspace, unconstrained', 'ablation_qr_decoupled_v1': 'Subspace, decoupled QR',
        'ablation_soft_ortho': 'Subspace, soft penalty', 'ablation_stiefel_no_cpi': 'Cayley, no swap', 'ablation_stiefel_no_adv': 'Cayley, no probe',
        'cisd_net_single': r'\method{}-S', 'cisd_net_dualstream': r'\method{}-D'}
man = {'P1': (pd.read_csv(f'{R}/manifests/plantdoc_P1_P2_manifest.csv'), 'split_P1_official', 'label_19'),
       'P2': (pd.read_csv(f'{R}/manifests/plantdoc_P1_P2_manifest.csv'), 'split_P2_stratified', 'label_19'),
       'FP': (pd.read_csv(f'{R}/manifests/fieldplant_FP_grouped_manifest.csv'), 'split_FP_grouped', 'label_6')}
led = json.load(open(f'{R}/results/v3_run_ledger_and_replay_audit.json'))
rec = {(r['protocol'], r['config_name'], r['seed']): r for r in led['per_run_records']}
def load(p, c, s):
    out = {}
    with zipfile.ZipFile(f'{R}/predictions_full_precision/{p}_{c}_seed{s}_predictions.npz') as zf:
        for n in zf.namelist():
            if 'image_ids' in n: continue
            with zf.open(n) as fh: out[n[:-4]] = np.lib.format.read_array(fh, allow_pickle=False)
    return out
def macro_f1(y, p, C):
    f = []
    for c in range(C):
        if (y == c).sum() == 0: continue
        tp = np.sum((y == c) & (p == c)); fp = np.sum((y != c) & (p == c)); fn = np.sum((y == c) & (p != c)); f.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return float(np.mean(f))
def lac_sets(pc, yc, pt, alpha=0.05):  # procedure of the training script, re-implemented
    n = len(yc); k = min(max(int(math.ceil((n + 1) * (1 - alpha))) - 1, 0), n - 1)
    q = np.sort(1 - pc[np.arange(n), yc], kind='mergesort')[k]; S = (1 - pt) <= q; e = S.sum(1) == 0
    S[e, pt[e].argmax(1)] = True; return S, int(e.sum())
QK = ['Q1', 'Q2', 'Q3', 'Q4']
R_ = {}; COR = {}; QT = {}; BOX = {}; TESTMASK = {}
replay_ok = 0
for p in PROT:
    M, sc, lab = man[p]; te = M[M[sc] == 'test'].reset_index(drop=True)
    for c in CFG:
        rows = []
        for s in SEEDS:
            z = load(p, c, s); y = z['test_y_true']; pr = z['test_probs'].astype(np.float64); pred = pr.argmax(1); cor = pred == y; C = pr.shape[1]
            assert np.array_equal(y, te[lab].values)
            cl = z['test_clutter_scores']; e = np.quantile(cl, [0, .25, .5, .75, 1.0])
            q = np.where(cl < e[1], 'Q1', np.where(cl < e[2], 'Q2', np.where(cl < e[3], 'Q3', 'Q4'))); QT[p] = q; BOX[p] = te.box_frac.values
            d = {'top1': cor.mean(), 'f1': macro_f1(y, pred, C)}
            for k in QK: d[k] = cor[q == k].mean()
            S, ne = lac_sets(z['cal_probs'].astype(np.float64), z['cal_y_true'], pr); replay_ok += int(np.array_equal(S, z['test_sets_lac']))
            d['sz'] = S.sum(1).mean(); d['cov'] = S[np.arange(len(y)), y].mean(); d['filled'] = ne / len(y)
            Sm = z['test_sets_mod']; d['szmod'] = Sm.sum(1).mean(); d['covmod'] = Sm[np.arange(len(y)), y].mean(); d['gamma'] = float(z['mod_gamma'][0])
            o = np.argsort(-pr.max(1), kind='mergesort'); d['aurcc'] = np.mean(np.cumsum((~cor)[o]) / np.arange(1, len(y) + 1))
            r = rec[(p, c, s)]; assert abs(r['top1_acc'] - d['top1']) < 1e-12 and abs(r['macro_f1'] - d['f1']) < 1e-9 and abs(r['q4_acc'] - d['Q4']) < 1e-12 and abs(r['aurcc'] - d['aurcc']) < 1e-12
            d['cross'] = r['ortho_cross_up_ub_fro']; d['hsic'] = r['hsic_raw_unscaled']; d['gate'] = r['learned_gate_probs']; d['temp'] = r['temperature_val']
            rows.append(d); COR[(p, c, s)] = cor
        a = {}
        for k in rows[0]:
            v = np.array([r[k] for r in rows], float)
            if v.ndim == 2: a[k] = v.mean(0).tolist(); continue
            a[k] = float(v.mean()); a[k + '_sd'] = float(v.std(ddof=1)); a[k + '_min'] = float(v.min()); a[k + '_max'] = float(v.max())
        R_[(p, c)] = a
assert replay_ok == 99, replay_ok
# FieldPlant near-duplicate audit (pHash Hamming <= 6, using hashes from the earlier group-audited manifest)
FPLEAK = None
if PH:
    fm = man['FP'][0].merge(pd.read_csv(PH)[['sha256', 'phash']], on='sha256', how='left'); h = np.array([int(x, 16) for x in fm.phash]); sp = fm.split_FP_grouped.values
    leak = np.zeros(len(fm), bool); pairs = cross = 0
    for i in range(len(h)):
        dd = np.array([int(v).bit_count() for v in (h[i] ^ h)]); nb = np.where((dd <= 6) & (np.arange(len(h)) != i))[0]
        pairs += int((nb > i).sum()); cross += int(((nb > i) & (sp[nb] != sp[i])).sum())
        if sp[i] == 'test' and np.any(sp[nb] != 'test'): leak[i] = True
    TESTMASK['FP'] = ~leak[sp == 'test']; FPLEAK = dict(pairs=pairs, cross=cross, test_affected=int(leak.sum()), n_test=int((sp == 'test').sum()))
rng = np.random.default_rng(20261115); B = 20000
def paired(p, a, b, stratum=None, mask=None):
    d = np.mean([COR[(p, a, s)] for s in SEEDS], 0) - np.mean([COR[(p, b, s)] for s in SEEDS], 0)
    sel = np.ones(len(d), bool)
    if stratum: sel &= QT[p] == stratum
    if mask is not None: sel &= mask
    d = d[sel]; n = len(d); bs = d[rng.integers(0, n, (B, n))].mean(1); lo, hi = np.quantile(bs, [.025, .975])
    pv = (np.sum(np.abs((rng.choice([-1, 1], (B, n)) * d).mean(1)) >= abs(d.mean()) - 1e-12) + 1) / (B + 1)
    return dict(d=100 * d.mean(), lo=100 * lo, hi=100 * hi, p=float(pv), n=int(n), gain=int((d > 0.5).sum()), loss=int((d < -0.5).sum()))
CON = [('cisd_net_single', 'erm_cls'), ('cisd_net_single', 'gem_linear_baseline'), ('cisd_net_single', 'cosine_proto_baseline'),
       ('cisd_net_single', 'ablation_unconstrained'), ('cisd_net_single', 'ablation_qr_decoupled_v1'), ('cisd_net_single', 'ablation_soft_ortho'),
       ('cisd_net_single', 'ablation_stiefel_no_cpi'), ('cisd_net_single', 'ablation_stiefel_no_adv'),
       ('dualstream_concat_ctrl', 'gem_linear_baseline'), ('cisd_net_dualstream', 'dualstream_concat_ctrl'), ('cisd_net_dualstream', 'cisd_net_single')]
PT = {}
for p in PROT:
    for a, b in CON:
        PT[(p, a, b, 'all')] = paired(p, a, b)
        for k in QK: PT[(p, a, b, k)] = paired(p, a, b, k)
if 'FP' in TESTMASK:
    for a, b in CON: PT[('FPclean', a, b, 'all')] = paired('FP', a, b, mask=TESTMASK['FP'])
    FPCLEAN = {c: float(np.mean([COR[('FP', c, s)][TESTMASK['FP']].mean() for s in SEEDS])) for c in CFG}
# ---------------- tables
def w(fn, L): open(os.path.join(TB, fn), 'w').write('\n'.join(L) + '\n')
def pm(m, s): return f'${100*m:.2f}\\pm{100*s:.2f}$'
L = []
for p, title in zip(PROT, ['P1: PlantDoc, official test set ($n=203$)', 'P2: PlantDoc, stratified partition ($n=219$)', 'FP: FieldPlant ($n=257$)']):
    L.append(r'\multicolumn{7}{l}{\textit{' + title + r'}}\\'); best = {k: max(R_[(p, c)][k] for c in CFG) for k in ['top1', 'f1'] + QK}
    for c in CFG:
        a = R_[(p, c)]; cells = []
        for k in ['top1', 'f1']:
            t = pm(a[k], a[k + '_sd']); cells.append(r'\boldmath' + t if abs(a[k] - best[k]) < 1e-12 else t)
        for k in QK:
            t = f'{100*a[k]:.2f}'; cells.append(r'\textbf{' + t + '}' if abs(a[k] - best[k]) < 1e-12 else t)
        if c in ('ablation_unconstrained', 'dualstream_concat_ctrl'): L.append(r'\addlinespace[2pt]')
        L.append(NAME[c] + ' & ' + ' & '.join(cells) + r'\\')
    if p != 'FP': L.append(r'\midrule')
w('main_results.tex', L)
CN = {('cisd_net_single', 'erm_cls'): r'\method{}-S $-$ linear probe', ('cisd_net_single', 'gem_linear_baseline'): r'\method{}-S $-$ GeM pooling',
      ('cisd_net_single', 'cosine_proto_baseline'): r'\method{}-S $-$ cosine prototype', ('cisd_net_single', 'ablation_unconstrained'): r'\method{}-S $-$ unconstrained',
      ('cisd_net_single', 'ablation_qr_decoupled_v1'): r'\method{}-S $-$ decoupled QR', ('cisd_net_single', 'ablation_soft_ortho'): r'\method{}-S $-$ soft penalty',
      ('cisd_net_single', 'ablation_stiefel_no_cpi'): r'\method{}-S $-$ Cayley, no swap', ('cisd_net_single', 'ablation_stiefel_no_adv'): r'\method{}-S $-$ Cayley, no probe',
      ('dualstream_concat_ctrl', 'gem_linear_baseline'): r'Concatenation $-$ GeM pooling', ('cisd_net_dualstream', 'dualstream_concat_ctrl'): r'\method{}-D $-$ concatenation',
      ('cisd_net_dualstream', 'cisd_net_single'): r'\method{}-D $-$ \method{}-S'}
def ci(r): return f'${r["d"]:+.2f}$ $[{r["lo"]:+.2f},\\,{r["hi"]:+.2f}]$'
L = []
for a, b in CON:
    if (a, b) in [('cisd_net_single', 'ablation_unconstrained'), ('dualstream_concat_ctrl', 'gem_linear_baseline')]: L.append(r'\addlinespace[2pt]')
    L.append(CN[(a, b)] + ' & ' + ' & '.join(ci(PT[(p, a, b, 'all')]) for p in PROT) + r'\\')
w('paired_overall.tex', L)
CQ = {('cisd_net_single', 'erm_cls'): r'\method{}-S $-$ linear', ('cisd_net_single', 'gem_linear_baseline'): r'\method{}-S $-$ GeM', ('cisd_net_single', 'ablation_unconstrained'): r'\method{}-S $-$ unconstr.'}
def ciq(r): return f'${r["d"]:+.2f}\\;[{r["lo"]:+.2f},{r["hi"]:+.2f}]$'
L = []
for (a, b) in CQ:
    for p in PROT: L.append((CQ[(a, b)] if p == 'P1' else '') + f' & {p} & ' + ' & '.join(ciq(PT[(p, a, b, k)]) for k in QK) + r'\\')
    L.append(r'\addlinespace[2pt]')
w('paired_quartile.tex', L[:-1])
def sci(x):
    if x < 1e-6: e = int(math.floor(math.log10(x))); return f'${x/10**e:.1f}\\times10^{{{e}}}$'
    return f'{x:.2f}'
L = []
for c in ['ablation_unconstrained', 'ablation_qr_decoupled_v1', 'ablation_soft_ortho', 'ablation_stiefel_no_cpi', 'ablation_stiefel_no_adv', 'cisd_net_single', 'cisd_net_dualstream']:
    L.append(NAME[c] + ' & ' + ' & '.join(sci(R_[(p, c)]['cross_max']) for p in PROT) + ' & ' + ' & '.join(f'{1e3*R_[(p, c)]["hsic"]:.1f}' for p in PROT) + r'\\')
w('geometry.tex', L)
L = []
for c in CFG:
    if c in ('ablation_unconstrained', 'dualstream_concat_ctrl'): L.append(r'\addlinespace[2pt]')
    L.append(NAME[c] + ' & ' + ' & '.join(f'{R_[(p, c)]["sz"]:.2f} & {100*R_[(p, c)]["cov"]:.1f} & {100*R_[(p, c)]["aurcc"]:.2f}' for p in PROT) + r'\\')
w('uncertainty.tex', L)
L = []
for c in ['ablation_unconstrained', 'ablation_qr_decoupled_v1', 'ablation_soft_ortho', 'ablation_stiefel_no_cpi', 'ablation_stiefel_no_adv', 'cisd_net_single', 'cisd_net_dualstream']:
    L.append(NAME[c] + ' & ' + ' & '.join('(' + ', '.join(f'{g:.2f}' for g in R_[(p, c)]['gate']) + ')' for p in PROT) + r'\\')
w('gates.tex', L)
# ---------------- figures
INK, MUTED, GRID, BLUE, ORANGE, AQUA = '#0b0b0b', '#52514e', '#e4e3df', '#2a78d6', '#eb6834', '#1baf7a'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.linewidth': 0.6, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'legend.frameon': False})
TIT = {'P1': 'P1  PlantDoc, official test (n = 203)', 'P2': 'P2  PlantDoc, stratified (n = 219)', 'FP': 'FP  FieldPlant (n = 257)'}
SER = [('erm_cls', 'Linear probe, full image', MUTED, '--', 'o', 'white'), ('gem_linear_baseline', 'GeM pooling, 4 views', ORANGE, '--', 's', 'white'),
       ('ablation_unconstrained', 'Subspace, unconstrained', AQUA, ':', '^', 'white'), ('cisd_net_single', 'CISD-Net-S (exact frame)', BLUE, '-', 'o', BLUE)]
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.8))
for ax, p in zip(axs, PROT):
    for c, lab, col, ls, mk, mfc in SER:
        ax.plot(range(4), [100 * R_[(p, c)][k] for k in QK], ls, color=col, lw=1.6, marker=mk, ms=5.5, mfc=mfc, mec=col, mew=1.3, label=lab, zorder=3)
    ax.set_xticks(range(4)); ax.set_xticklabels(['Q1\nleast', 'Q2', 'Q3', 'Q4\nmost']); ax.set_title(TIT[p], fontsize=8, loc='left')
    ax.grid(axis='y', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-0.3, 3.3); ax.set_xlabel('Clutter quartile of test images')
    v = [100 * R_[(p, c)][k] for c, *_ in SER for k in QK]; ax.set_ylim(math.floor((min(v) - 3) / 5) * 5, min(100, math.ceil((max(v) + 2) / 5) * 5))
axs[0].set_ylabel('Top-1 accuracy (%)'); h, l = axs[0].get_legend_handles_labels()
fig.legend(h, l, loc='lower center', ncol=4, fontsize=7.5, handlelength=3, columnspacing=1.4, bbox_to_anchor=(0.5, -0.005)); fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(f'{FG}/fig2_accuracy_by_clutter.pdf'); fig.savefig(f'{FG}/fig2_accuracy_by_clutter.png', dpi=200); plt.close(fig)
fig, axs = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
ROWS = [('all', 'All images')] + [(k, k + (' (least clutter)' if k == 'Q1' else ' (most clutter)' if k == 'Q4' else '')) for k in QK]
for ax, p in zip(axs, PROT):
    for (a, b), col, mk, off, lab in [(('cisd_net_single', 'ablation_unconstrained'), AQUA, '^', 0.2, 'vs unconstrained subspace'), (('cisd_net_single', 'erm_cls'), MUTED, 'o', 0.0, 'vs linear probe, full image'), (('cisd_net_single', 'gem_linear_baseline'), ORANGE, 's', -0.2, 'vs GeM pooling, 4 views')]:
        for i, (k, _) in enumerate(ROWS):
            r = PT[(p, a, b, k)]; yv = len(ROWS) - 1 - i + off
            ax.plot([r['lo'], r['hi']], [yv, yv], color=col, lw=1.5, solid_capstyle='round', zorder=2)
            ax.plot(r['d'], yv, mk, color=col, ms=5, mec='white', mew=0.7, zorder=3, label=lab if i == 0 else None)
    ax.axvline(0, color=MUTED, lw=0.8); ax.axhline(3.5, color=GRID, lw=0.8); ax.set_yticks(range(len(ROWS))); ax.set_yticklabels([t for _, t in ROWS][::-1])
    ax.set_title(TIT[p], fontsize=8, loc='left'); ax.grid(axis='x', color=GRID, lw=0.6); ax.set_axisbelow(True); ax.set_xlim(-25, 25); ax.set_xlabel('Difference (points)'); ax.tick_params(axis='y', length=0)
h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, loc='lower center', ncol=3, fontsize=7.5, columnspacing=2, bbox_to_anchor=(0.5, -0.005))
fig.tight_layout(rect=(0, 0.08, 1, 1)); fig.savefig(f'{FG}/fig3_paired_differences.pdf'); fig.savefig(f'{FG}/fig3_paired_differences.png', dpi=200); plt.close(fig)
DATA = {}
for p in PROT:
    M, sc, lab = man[p]; DATA[p] = dict(n={k: int((M[sc] == k).sum()) for k in ('train', 'val', 'cal', 'test')}, nq={k: int((QT[p] == k).sum()) for k in QK},
        box_q={k: float(np.median(BOX[p][QT[p] == k])) for k in QK}, corr=float(np.corrcoef(M.box_frac, M.clutter_score)[0, 1]), ncls=int(M[lab].nunique()))
out = dict(R={f'{k[0]}|{k[1]}': v for k, v in R_.items()}, PT={'|'.join(map(str, k)): v for k, v in PT.items()}, DATA=DATA, FPLEAK=FPLEAK,
           FPCLEAN=FPCLEAN if PH else None, replay_lac=replay_ok, bootstrap=dict(B=B, seed=20261115))
json.dump(out, open(os.path.join(R, 'outputs', 'analysis', 'numbers_v3.json'), 'w'), indent=1)
print('replay', replay_ok, 'FPLEAK', FPLEAK)
for p in PROT:
    print(f'\n## {p}', DATA[p])
    for c in CFG:
        a = R_[(p, c)]; print(f' {c:26s} top1 {100*a["top1"]:.2f}±{100*a["top1_sd"]:.2f} f1 {100*a["f1"]:.2f} Q {" ".join(f"{100*a[k]:.2f}" for k in QK)} | sz {a["sz"]:.2f} cov {100*a["cov"]:.1f} filled {100*a["filled"]:.1f}% szmod {a["szmod"]:.2f} covmod {100*a["covmod"]:.1f} g {a["gamma"]:.2f} | aurcc {100*a["aurcc"]:.2f} | cross {a["cross_min"]:.2e}-{a["cross_max"]:.2e} hsic {1e3*a["hsic"]:.2f} | gate {np.round(a["gate"],2).tolist()} T {a["temp_min"]}-{a["temp_max"]}')
print('\n## paired')
for k, v in PT.items():
    if k[3] in ('all', 'Q4', 'Q1'): print(k, f'{v["d"]:+.2f} [{v["lo"]:+.2f},{v["hi"]:+.2f}] p={v["p"]:.3f} n={v["n"]} +{v["gain"]}/-{v["loss"]}')
if PH: print('FP clean acc', {c: round(100 * v, 2) for c, v in FPCLEAN.items()})
# ---------------- paired bootstrap of AURCC and LAC set size within the subspace family
def aurcc_of(pr, y, idx):
    p = pr[idx]; c = (p.argmax(1) == y[idx]); o = np.argsort(-p.max(1), kind='mergesort'); return np.mean(np.cumsum(~c[o]) / np.arange(1, len(idx) + 1))
UNC = {}
rng2 = np.random.default_rng(7)
for p in PROT:
    Z = {c: [load(p, c, s) for s in SEEDS] for c in ['cisd_net_single', 'ablation_unconstrained', 'ablation_qr_decoupled_v1', 'ablation_soft_ortho', 'erm_cls', 'gem_linear_baseline']}
    y = Z['cisd_net_single'][0]['test_y_true']; n = len(y)
    def stat(c, idx): return np.mean([aurcc_of(z['test_probs'].astype(np.float64), y, idx) for z in Z[c]])
    def size(c, idx): return np.mean([z['test_sets_lac'][idx].sum(1).mean() for z in Z[c]])
    for b in ['ablation_unconstrained', 'ablation_qr_decoupled_v1', 'ablation_soft_ortho', 'erm_cls', 'gem_linear_baseline']:
        full = np.arange(n); da = 100 * (stat('cisd_net_single', full) - stat(b, full)); ds = size('cisd_net_single', full) - size(b, full)
        bs = []; bz = []
        for _ in range(2000):
            idx = rng2.integers(0, n, n); bs.append(100 * (stat('cisd_net_single', idx) - stat(b, idx))); bz.append(size('cisd_net_single', idx) - size(b, idx))
        UNC[f'{p}|{b}'] = dict(aurcc=(da, *np.quantile(bs, [.025, .975])), size=(ds, *np.quantile(bz, [.025, .975])))
        print(p, f'CISD-S - {b:26s} AURCC {da:+.2f} [{np.quantile(bs,.025):+.2f},{np.quantile(bs,.975):+.2f}]  LAC size {ds:+.2f} [{np.quantile(bz,.025):+.2f},{np.quantile(bz,.975):+.2f}]')
nj = json.load(open(os.path.join(R, 'outputs', 'analysis', 'numbers_v3.json'))); nj['UNC'] = UNC; json.dump(nj, open(os.path.join(R, 'outputs', 'analysis', 'numbers_v3.json'), 'w'), indent=1)
