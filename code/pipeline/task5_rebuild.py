"""rerun_v3 Task 5 ("rebuild one cache from scratch rather than loading it, and send the hash of the test-repository
list"). Rebuilds the Phase 3 cache Aggregate.before_cve_R1_B20 (sym, seed 42, no metadata, original sampling) from
the raw data under the name Aggregate.before_cve_R1_B20_rebuildcheck (same RNG: the comment only names the folder),
compares every repo's windows and target details with the cached copy, replays the split (as train.py) for both, and
prints the test-repo list sha256 for each, plus the Phase 3 npz's test repos. Writes rerun_v3/05_task5/rebuild_check.txt.
"""
import hashlib, os, sys
sys.path.insert(0, os.getcwd())
import numpy as np
from dataset_utils import Aggregate, extract_dataset, pad_and_fix, split_repos
from train import init

def build(comment):
    init(seed=42)
    repos, name, cols = extract_dataset(aggr_options=Aggregate.before_cve, resample=24, benign_vuln_ratio=1, hours=0,
                                        days=10, backs=20, cache=True, metadata=False, comment=comment,
                                        data_location="data_collection", cache_location="ready_data")
    return {r.file: r for r in repos}, name, list(cols)

def split(comment):
    init(seed=42)
    repos, _, _ = extract_dataset(aggr_options=Aggregate.before_cve, resample=24, benign_vuln_ratio=1, hours=0, days=10,
                                  backs=20, cache=True, metadata=False, comment=comment,
                                  data_location="data_collection", cache_location="ready_data")
    repos, nv = pad_and_fix(repos)
    return [r.file for r in split_repos(repos, int(0.8 * nv) + int(0.1 * nv))[1]]

new, nname, ncol = build("rebuildcheck")          # built from raw data now (folder did not exist)
old, oname, ocol = build("")                       # the Phase 3 cache, loaded
same = sorted(new) == sorted(old) and ncol == ocol and all(
    len(new[r].vuln_lst) == len(old[r].vuln_lst) and len(new[r].benign_lst) == len(old[r].benign_lst)
    and all(np.array_equal(a, b) for a, b in zip(list(new[r].vuln_lst) + list(new[r].benign_lst),
                                                  list(old[r].vuln_lst) + list(old[r].benign_lst)))
    and new[r].vuln_details == old[r].vuln_details and new[r].benign_details == old[r].benign_details for r in old)
h = lambda xs: hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()
tn, to = split("rebuildcheck"), split("")
p3 = sorted(set(np.load("runs_v2/06_phase3/task1_b20_nometa/preds_transformer_sym_b20_s42_nometa.npz")["test_repo"].astype(str)))
out = [f"rebuilt from raw: ready_data/{nname}; Phase 3 cache: ready_data/{oname}",
       f"repos {len(new)} / {len(old)}; every window, target and column identical: {same}",
       f"test repos after the split: rebuilt {len(tn)}, cached {len(to)}, same list in same order: {tn == to}",
       f"test-repo list sha256 (sorted names): rebuilt {h(tn)}  cached {h(to)}",
       f"Phase 3 npz test repos (transformer_sym_b20_s42_nometa): {len(p3)}, sha256 {h(p3)}, equal: {sorted(tn) == p3}",
       f"VERDICT: {'IDENTICAL' if same and tn == to and sorted(tn) == p3 else 'DIFFERENT'}"]
os.makedirs("rerun_v3/05_task5", exist_ok=True)
open("rerun_v3/05_task5/rebuild_check.txt", "w").write("\n".join(out) + "\n"); print("\n".join(out))
