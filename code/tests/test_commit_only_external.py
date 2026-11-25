"""rerun_v3 tests for the Commit-only option in evaluate_cvefixes.process_repo / make_model_dir_name (CPU, before
any run). On the K smallest near-domain (enriched) repos with a fix:
  E1 default path unchanged: edited file with benign_commit_only=False gives identical samples (windows, labels,
     provenance) to the original file, balanced and realistic
  E2 balanced Commit-only: fixes identical to E1; every benign target has provenance type Commit and is verified;
     benign count = fix count per repo where enough Commit events exist; per-repo seeded draw reproducible
  E3 realistic Commit-only: targets = fixes + every non-fix Commit window; no other type
  E4 model folder names equal the names train.py gives (dataset_utils.make_new_dir_name with the seed comment)
Usage: CUDA_VISIBLE_DEVICES= python3 rerun_v3/01_code_changes/tests/test_commit_only_external.py [K]
"""
import importlib.util
import os
import sys

import numpy as np

sys.path.insert(0, os.getcwd())
K = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); sys.modules[name] = m; spec.loader.exec_module(m); return m


new = load("ev_new", "evaluate_cvefixes.py")
old = load("ev_old", "rerun_v3/01_code_changes/originals/evaluate_cvefixes.py")
import dataset_utils as du  # noqa: E402

D = os.path.join(new.CVEFIXES_DIR_BASE, "enriched")
for m in (new, old):
    m.CVEFIXES_DIR = m.CVEFIXES_DIR_BASE  # as main(): CSVs from enriched/, GraphQL and metadata from the base folder
import pandas as pd  # noqa: E402
files = sorted((os.path.join(D, f) for f in os.listdir(D) if f.endswith(".csv")), key=os.path.getsize)
files = [f for f in files if pd.read_csv(f, usecols=["Vuln"]).Vuln.eq(1).any()][:K]
print("repos:", [os.path.basename(f) for f in files])
fails = []


def check(name, ok):
    print(("PASS " if ok else "FAIL ") + name); ok or fails.append(name)


def same(a, b):
    return (a is None) == (b is None) and (a is None or (len(a) == len(b) and all(
        np.array_equal(x[0], y[0]) and x[1] == y[1] and x[2] == y[2] for x, y in zip(a, b))))


def run(mod, f, **kw):
    return mod.process_repo(f, du.Aggregate.before_cve, 20, {}, None, **kw)[0]


def L(x):
    return [] if x is None else x


e1 = e2 = e3 = True; nb = nv = 0; n_short = 0
for f in files:
    for realistic in (False, True):
        e1 &= same(run(old, f, realistic=realistic), run(new, f, realistic=realistic))
    s0 = L(run(new, f)); s1 = L(run(new, f, benign_commit_only=True)); s1b = L(run(new, f, benign_commit_only=True))
    v0 = [x for x in s0 if x[1] == 1]; v1 = [x for x in s1 if x[1] == 1]; b1 = [x for x in s1 if x[1] == 0]
    e2 &= same(v0, v1) and same(s1, s1b) and all(x[2][2] == "Commit" and x[2][5] for x in b1)
    raw = pd.read_csv(f, usecols=["type", "Vuln"])
    if len(b1) != len(v1):
        n_short += 1
        e2 &= len(b1) < len(v1)
    nb += len(b1); nv += len(v1)
    r1 = L(run(new, f, realistic=True, benign_commit_only=True)); r0 = L(run(new, f, realistic=True))
    e3 &= all(x[2][2] == "Commit" for x in r1) and same([x for x in r0 if x[1] == 1], [x for x in r1 if x[1] == 1]) \
        and same([x for x in r0 if x[1] == 0 and x[2][2] == "Commit"], [x for x in r1 if x[1] == 0])
check("E1 default path identical to original (balanced and realistic)", e1)
check(f"E2 balanced Commit-only: fixes unchanged, {nb} benign all Commit+verified, reproducible "
      f"({n_short} repos with fewer Commit benign windows than fixes; {nv} fixes)", e2)
check("E3 realistic Commit-only = fixes + all non-fix Commit windows", e3)

ok4 = True
for aggr in (du.Aggregate.before_cve, du.Aggregate.only_before):
    for meta in (False, True):
        for seed in (42, 123, 7):
            c = "commitonly" if seed == 42 else f"commitonly_s{seed}"
            ok4 &= new.make_model_dir_name(aggr, 20, meta, seed, "commitonly") == \
                du.make_new_dir_name(aggr, 20, 1, 10, 0, 24, meta, c)
            ok4 &= new.make_model_dir_name(aggr, 20, meta, seed) == old.make_model_dir_name(aggr, 20, meta, seed)
check("E4 model folder names match train.py naming; default names unchanged", ok4)
print("ALL PASS" if not fails else "FAILED: " + ", ".join(fails))
sys.exit(1 if fails else 0)
