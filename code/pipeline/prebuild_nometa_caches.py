"""Build the no-metadata (behavioural-only) dataset caches on CPU, ahead of training.

Reproduces exactly what `train.py --cache` (without --metadata) would build for a
given window mode and seed: same init(seed) RNG seeding, same extract_dataset()
arguments, same seed-suffixed cache comment. Building on CPU lets the slow cache
step run in parallel with GPU work; train.py later just loads the cache.

Usage: CUDA_VISIBLE_DEVICES= python3 prebuild_nometa_caches.py <before|only_before> <seed> [backs]
"""
import os
import sys
import time

from dataset_utils import Aggregate, extract_dataset

from train import init


def main():
    if len(sys.argv) not in (3, 4):
        sys.exit(__doc__)
    aggr = Aggregate(sys.argv[1])
    seed = int(sys.argv[2])
    backs = int(sys.argv[3]) if len(sys.argv) == 4 else 20

    init(seed=seed)
    comment = "" if seed == 42 else f"s{seed}"
    t0 = time.time()
    all_repos, exp_name, columns = extract_dataset(
        aggr_options=aggr, resample=24, benign_vuln_ratio=1, hours=0, days=10,
        backs=backs, cache=True, metadata=False, comment=comment,
        data_location="data_collection", cache_location="ready_data",
    )
    print(f"[PREBUILD] {exp_name}: {len(all_repos)} repos, {len(columns)} columns, "
          f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
