"""Shortcut probe: accuracy of the one-feature rule 'predict a fix iff the target event is a Commit'
under the original (unmatched) sampling, compared with the trained models, and the models' ROC-AUC on commit
targets only. Extract results/original_sampling/runs/task1_b20_nometa.tar.gz first, then:
    python3 code/analysis/shortcut_probe.py path/to/task1_b20_nometa
"""
import glob, os, re, sys, collections
import numpy as np

def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float)
    o = np.argsort(s, kind='mergesort'); ss = s[o]; ranks = np.empty(len(s)); i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and ss[j + 1] == ss[i]: j += 1
        ranks[o[i:j + 1]] = (i + j) / 2 + 1; i = j + 1
    P = y.sum(); N = len(y) - P
    return (ranks[y == 1].sum() - P * (P + 1) / 2) / (P * N)

d = sys.argv[1] if len(sys.argv) > 1 else 'task1_b20_nometa'
rows = collections.defaultdict(list)
for f in sorted(glob.glob(os.path.join(d, '*.npz'))):
    m = re.match(r'(preds|ext_near|ext_unseen)_(\w+?)_(sym|causal)_b20_s(\d+)_nometa', os.path.basename(f))
    if not m: continue
    st, arch, mode, _ = m.groups(); z = np.load(f, allow_pickle=True)
    if st == 'preds': y, s, et, thr = z['test_y'], z['test_pred'], z['test_event_type'], float(z['tau_val'])
    else: y, s, et, thr = z['labels'], z['scores'], z['event_type'], float(z['threshold'])
    y = y.astype(int); et = et.astype(str); rule = (et == 'Commit').astype(int); c = et == 'Commit'
    rows[(st, f'{arch}_{mode}')].append(((rule == y).mean(), ((s >= thr).astype(int) == y).mean(),
                                         auc(y, s), auc(y[c], s[c]), (et != 'Commit').mean()))
print(f"{'set':11s}{'model':20s}{'rule acc':>9s}{'model acc':>10s}{'model AUC':>10s}{'AUC commits':>12s}{'non-commit':>11s}")
for k in sorted(rows):
    a = np.array(rows[k]).mean(0)
    print(f"{k[0]:11s}{k[1]:20s}{a[0]:9.3f}{a[1]:10.3f}{a[2]:10.3f}{a[3]:12.3f}{a[4]:11.3f}  (seeds={len(rows[k])})")
