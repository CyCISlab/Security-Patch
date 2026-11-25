"""Check that the split replay used by rerun_v3/02_caches/split_check.py reproduces the real Phase 3 split: for
existing Phase 3 caches, the replayed test repos must equal the repos in that run's preds npz (test_repo key).
Usage: CUDA_VISIBLE_DEVICES= python3 rerun_v3/01_code_changes/tests/test_split_replay.py"""
import os
import sys
sys.path.insert(0, os.getcwd())
import numpy as np  # noqa: E402

from dataset_utils import Aggregate, extract_dataset, pad_and_fix, split_repos  # noqa: E402
from train import init  # noqa: E402

fails = 0
for aggr, seed, meta, npz in ((Aggregate.before_cve, 42, False, "task1_b20_nometa/preds_transformer_sym_b20_s42_nometa.npz"),
                             (Aggregate.only_before, 123, False, "task1_b20_nometa/preds_conv1d_causal_b20_s123_nometa.npz"),
                             (Aggregate.before_cve, 7, True, "task1_b20_meta/preds_transformer_sym_b20_s7.npz")):
    init(seed=seed)
    repos, name, _ = extract_dataset(aggr_options=aggr, resample=24, benign_vuln_ratio=1, hours=0, days=10, backs=20,
                                     cache=True, metadata=meta, comment="" if seed == 42 else f"s{seed}",
                                     data_location="data_collection", cache_location="ready_data")
    repos, nv = pad_and_fix(repos)
    _, test, _ = split_repos(repos, int(0.8 * nv) + int(0.1 * nv))
    want = set(np.load("runs_v2/06_phase3/" + npz)["test_repo"].astype(str))
    got = {r.file for r in test}
    ok = got == want
    fails += not ok
    print(f"{'PASS' if ok else 'FAIL'} {name}: replayed test repos {len(got)}, Phase 3 npz test repos {len(want)}")
print("ALL PASS" if not fails else "FAILED")
sys.exit(1 if fails else 0)
