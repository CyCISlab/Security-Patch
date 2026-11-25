"""Split gate for rerun_v3 (supervisor: "keep the current repository split"; Task 5: hash of the test-repo list).
Replays train.py steps 1-3 exactly (init(seed); extract_dataset(cache=True) loads the cache; pad_and_fix shuffles the
repos with the seeded RNG; split_repos cuts by cumulative fix count) for the new Commit-only cache and for the Phase 3
cache of the same (window, metadata, seed), and compares the outcome.
PASS only if: same repos in the same order in train+val and in test, same fixes per repo, and every benign window
count equals the fix count per repo (ratio 1). Also writes the all-zero window count (Task 5).
Usage: CUDA_VISIBLE_DEVICES= python3 rerun_v3/02_caches/split_check.py <before|only_before> <seed> <meta 0|1>
"""
import hashlib
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np  # noqa: E402

from dataset_utils import Aggregate, extract_dataset, pad_and_fix, split_repos  # noqa: E402
from train import init  # noqa: E402

aggr, seed, meta = Aggregate(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
OUT = os.path.join(os.path.dirname(__file__), "split_checks")


def split_of(comment, commit_only):
    init(seed=seed)
    repos, name, _ = extract_dataset(aggr_options=aggr, resample=24, benign_vuln_ratio=1, hours=0, days=10, backs=20,
                                     cache=True, metadata=meta, comment=comment, data_location="data_collection",
                                     cache_location="ready_data", benign_commit_only=commit_only)
    if not os.path.isfile(os.path.join("ready_data", name, "column_names.pkl")):
        sys.exit(f"cache ready_data/{name} incomplete")
    repos, nv = pad_and_fix(repos)
    trv, test, _ = split_repos(repos, int(0.8 * nv) + int(0.1 * nv))
    zero = sum(int(not np.any(w)) for r in repos for w in list(r.vuln_lst) + list(r.benign_lst))
    nwin = sum(len(r.vuln_lst) + len(r.benign_lst) for r in repos)
    ratio_ok = all(len(r.benign_lst) == len(r.vuln_lst) for r in repos)
    nben, nvul = sum(len(r.benign_lst) for r in repos), sum(len(r.vuln_lst) for r in repos)
    return name, [r.file for r in trv], [r.file for r in test], {r.file: len(r.vuln_lst) for r in repos}, \
        zero, nwin, ratio_ok, nben, nvul, sum(len(r.vuln_lst) for r in test)


sfx = "" if seed == 42 else f"s{seed}"
new = split_of("commitonly" + ("_" + sfx if sfx else ""), True)
old = split_of(sfx, False)
h = lambda xs: hashlib.sha256("\n".join(sorted(xs)).encode()).hexdigest()  # noqa: E731
lines = [f"new cache ready_data/{new[0]}  vs  Phase 3 cache ready_data/{old[0]}",
         f"repos: new {len(new[3])}, old {len(old[3])}; train+val {len(new[1])} / {len(old[1])}; test {len(new[2])} / {len(old[2])}",
         f"test-repo list sha256 (sorted names): new {h(new[2])}  old {h(old[2])}",
         f"train+val-repo list sha256: new {h(new[1])}  old {h(old[1])}",
         f"same order train+val: {new[1] == old[1]}; same order test: {new[2] == old[2]}; same fixes per repo: {new[3] == old[3]}",
         f"fix windows: new {new[8]} (test {new[9]}), old {old[8]} (test {old[9]}); benign windows: new {new[7]}, old {old[7]}",
         f"new: benign = fixes in every repo: {new[6]}",
         f"all-zero windows: new {new[4]} of {new[5]} ({100 * new[4] / new[5]:.2f}%), old {old[4]} of {old[5]} "
         f"({100 * old[4] / old[5]:.2f}%) (kept, not dropped: train.py has no filter)"]
ok = new[1] == old[1] and new[2] == old[2] and new[3] == old[3] and new[6]
lines.append(f"VERDICT: {'PASS' if ok else 'FAIL'}")
os.makedirs(OUT, exist_ok=True)
open(os.path.join(OUT, f"split_{new[0]}.txt"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if ok else 1)
