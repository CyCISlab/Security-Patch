"""Build one Commit-only cache on CPU, ahead of training (rerun_v3).

Copied from runs_v2/05_burst_rate_pin/prebuild_nometa_caches.py (Phase 3), which reproduces exactly what
`train.py --cache` builds (same init(seed) seeding, same extract_dataset() arguments, same seed-suffixed comment).
Changes: (1) metadata on/off is an argument (Phase 3 built metadata caches earlier, in train.py); (2) the cache
comment is "commitonly" / "commitonly_s<seed>" and benign_commit_only=True, exactly as train.py builds them with
`--comment commitonly --benign-commit-only`. Training later loads the cache, so the split RNG is untouched (NOTES s6).
Usage (project root): CUDA_VISIBLE_DEVICES= python3 rerun_v3/02_caches/prebuild_cache.py <before|only_before> <seed> <meta 0|1>
"""
import os
import sys
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # project root

from dataset_utils import Aggregate, extract_dataset  # noqa: E402

from train import init  # noqa: E402


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    aggr, seed, meta = Aggregate(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
    init(seed=seed)
    comment = "commitonly" if seed == 42 else f"commitonly_s{seed}"   # = train.py with --comment commitonly
    t0 = time.time()
    all_repos, exp_name, columns = extract_dataset(
        aggr_options=aggr, resample=24, benign_vuln_ratio=1, hours=0, days=10,
        backs=20, cache=True, metadata=meta, comment=comment,
        data_location="data_collection", cache_location="ready_data", benign_commit_only=True,
    )
    print(f"[PREBUILD] {exp_name}: {len(all_repos)} repos, {len(columns)} columns, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
