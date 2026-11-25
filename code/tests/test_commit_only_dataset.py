"""rerun_v3 tests for the Commit-only benign option in dataset_utils.create_dataset (written before any run).
On a subset of K small training repos (symlinked raw data, CPU, cache written to a scratch folder):
  T1 default path unchanged: the edited dataset_utils with benign_commit_only=False gives byte-identical windows,
     details and columns to the original file (rerun_v3/01_code_changes/originals/dataset_utils.py)
  T2 Commit-only: same columns; vulnerable windows and details identical to T1; every benign target is a Commit
     event and not vulnerable; benign count = vulnerable count per repo (ratio 1)
  T3 reproducible: two Commit-only builds with the same seed are identical
  T4 guard: flag without 'commitonly' in the name, and the name without the flag, are refused
Usage: CUDA_VISIBLE_DEVICES= python3 rerun_v3/01_code_changes/tests/test_commit_only_dataset.py SCRATCH [K] [before|only_before]
"""
import importlib.util
import os
import pickle
import random
import shutil
import sys

import numpy as np
import pandas as pd

ROOT = os.getcwd()
sys.path.insert(0, ROOT)
SCR, K = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 6
MODE = sys.argv[3] if len(sys.argv) > 3 else "before"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


new = load("du_new", "dataset_utils.py")
old = load("du_old", "rerun_v3/01_code_changes/originals/dataset_utils.py")

# subset data folder: K smallest raw repos that have a vulnerable event
data = os.path.join(SCR, "data"); shutil.rmtree(data, ignore_errors=True)
os.makedirs(os.path.join(data, "gh_cve_proccessed"))
for sub in ("graphql", "repo_metadata.json"):
    os.symlink(os.path.join(ROOT, "data_collection", sub), os.path.join(data, sub))
src = os.path.join(ROOT, "data_collection", "gh_cve_proccessed")
# candidates: repos present in the Phase 3 cache of this mode (they produce windows), smallest raw files first
in_cache = {f[:-4] for f in os.listdir(os.path.join(ROOT, "ready_data",
            "Aggregate.before_cve_R1_B20" if MODE == "before" else "Aggregate.only_before_R1_B20")) if f.endswith(".pkl")}
picked = sorted((f[:-8] for f in os.listdir(src) if f.endswith(".parquet") and f[:-8] in in_cache),
                key=lambda p: os.path.getsize(os.path.join(src, p + ".parquet")))[:K]
for p in picked:
    for ext in (".parquet", ".csv"):
        if os.path.exists(os.path.join(src, p + ext)):
            os.symlink(os.path.join(src, p + ext), os.path.join(data, "gh_cve_proccessed", p + ext))
print("repos:", picked)

types_seen = {}


def build(mod, flag, tag):
    out = os.path.join(SCR, "cache_" + tag); shutil.rmtree(out, ignore_errors=True); os.makedirs(out)
    mod.DATASET_DIRNAME = out
    orig_ohe = mod.add_type_one_hot_encoding

    def capture(df):  # record each row's event type under its final (created_at, idx) index
        types_seen.setdefault(tag, {}).update(dict(zip(df.index, df["type"].astype(str))))
        return orig_ohe(df)
    mod.add_type_one_hot_encoding = capture
    random.seed(42); np.random.seed(42)
    kw = {"benign_commit_only": True} if flag else {}
    comment = "commitonlytest" if flag else "test"
    os.makedirs(os.path.join(out, mod.make_new_dir_name(mod.Aggregate(MODE), 20, 1, 10, 0, 24, False, comment)))
    repos, cols = mod.create_dataset(data, mod.Aggregate(MODE), 1, 0, 10, 24, 20, metadata=False, comment=comment, **kw)
    mod.add_type_one_hot_encoding = orig_ohe
    return {r.file: r for r in repos}, list(cols)


def same_list(a, b):
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def canon(w, t=40):
    """A window as training sees it: pad_and_fix left-pads every window with zero rows to the global length."""
    return np.pad(w, ((t - w.shape[0], 0), (0, 0)))


def same_padded(a, b):
    return len(a) == len(b) and all(np.array_equal(canon(x), canon(y)) for x, y in zip(a, b))


fails = []
def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); ok or fails.append(name)


o, ocol = build(old, False, "old")
n0, n0col = build(new, False, "new_default")
check("T1 default columns identical to original", ocol == n0col)
check("T1 default repos identical", sorted(o) == sorted(n0))
check("T1 default windows+details byte-identical", all(
    same_list(o[r].vuln_lst, n0[r].vuln_lst) and same_list(o[r].benign_lst, n0[r].benign_lst)
    and o[r].vuln_details == n0[r].vuln_details and o[r].benign_details == n0[r].benign_details for r in o))

c1, c1col = build(new, True, "commit")
check("T2 columns identical (marker not a feature)", c1col == ocol)
check("T2 same repos", sorted(c1) == sorted(o))
check("T2 vulnerable windows identical after global padding (what training sees), details identical", all(
    same_padded(o[r].vuln_lst, c1[r].vuln_lst) and o[r].vuln_details == c1[r].vuln_details for r in c1))
nz = sum(not np.any(w) for r in c1 for w in c1[r].vuln_lst + c1[r].benign_lst)
print(f"  (all-zero or empty windows in the Commit-only subset: {nz})")
ty = types_seen["commit"]
btypes = [ty[d[1]] for r in c1 for d in c1[r].benign_details]
vtypes = [ty[d[1]] for r in c1 for d in c1[r].vuln_details]
check(f"T2 all {len(btypes)} benign targets are Commit", set(btypes) == {"Commit"})
check(f"T2 all {len(vtypes)} vulnerable targets are Commit", set(vtypes) == {"Commit"})
otypes = [types_seen["old"][d[1]] for r in o for d in o[r].benign_details]
print(f"  (original sampling on this subset: {sum(t != 'Commit' for t in otypes)} of {len(otypes)} benign targets not Commit)")
vset = {(r, d[1]) for r in c1 for d in c1[r].vuln_details}
check("T2 no benign target is a vulnerable event", not any((r, d[1]) in vset for r in c1 for d in c1[r].benign_details))
check("T2 benign = vulnerable count per repo", all(len(c1[r].benign_lst) == len(c1[r].vuln_lst) for r in c1))
check("T2 feature width matches original", all(c1[r].benign_lst[0].shape[1] == o[r].benign_lst[0].shape[1] for r in c1))
check("T2 no window longer than B+B (40 rows)", all(w.shape[0] <= 40 for r in c1 for w in c1[r].vuln_lst + c1[r].benign_lst))

c2, _ = build(new, True, "commit2")
check("T3 Commit-only build reproducible", all(
    same_list(c1[r].benign_lst, c2[r].benign_lst) and c1[r].benign_details == c2[r].benign_details for r in c1))

for flag, comment in ((True, "s7"), (False, "commitonly_s7")):
    try:
        new.extract_dataset(aggr_options=new.Aggregate(MODE), backs=20, cache=True, comment=comment,
                            benign_commit_only=flag, data_location=data, cache_location=SCR)
        check(f"T4 guard refuses flag={flag} comment={comment}", False)
    except ValueError:
        check(f"T4 guard refuses flag={flag} comment={comment}", True)

print(f"{'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
