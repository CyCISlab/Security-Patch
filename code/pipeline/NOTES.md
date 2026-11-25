# rerun_v3 working notes

Same rules as runs_v2 (RUNBOOK_v2): nothing dropped, repeated or adjusted because of a number; no tuning on test or
external labels (tau from validation only); every run reported, weak seeds included; failed attempts kept beside the
rerun; tests written and passed before anything runs; every number cites the file it comes from; tables are made by
scripts. Seeds 42 123 7 256 999, B=20, the Phase 3 repository split, all Phase 3 npz keys.

## 1. The finding (supervisor, 2026-11-19) and its check (2026-11-20 20:10)
- Every vulnerable target is a Commit event, but benign targets were drawn from any non-vulnerable event.
- Code: `dataset_utils.py:926-930` (training) and `evaluate_cvefixes.py:294-301` (external) select
  `cur_repo["Vuln"] == 0` with no condition on the event type, although comments and names say "benign commits".
- Confirmed on our own Phase 3 files (seed 42, Transformer sym, no metadata): internal test 48.5% of benign targets
  not Commit, "Commit => vulnerable" rule accuracy 0.745; near-domain 71.9%, 0.860; unseen 72.4%, 0.862
  (`runs_v2/06_phase3/task1_b20_nometa/{preds,ext_near,ext_unseen}_transformer_sym_b20_s42_nometa.npz`, keys
  test_event_type / event_type). Full corpus: 546,193 of 2,094,039 rows are Commit events
  (`runs_v2/06_phase3/task2_fullcorpus/fullcorpus_preds_conv1d_sym_b20_s42_nometa.npz`).

## 2. Origin: inherited from the original code (Farhi et al. 2023)
- `docs/original_code/SecurityPatchDetection-main/dataset_utils.py:180-182` and `.../train.py:241-243` contain the
  same three lines (`benigns = cur_repo.index[cur_repo['Vuln'] == 0].tolist()`, shuffle, cut). That code accompanies
  Farhi, Koenigstein, Shavitt, "Detecting Security Patches via Behavioral Data in Code Repositories",
  arXiv:2302.02112 (2023) (`docs/original_code/.../readme.md`). It came into this repo in commit 748d149
  (2026-03-08, "some of the Nitzan Farhi's original code (all data manipulation code)").
- It was not fixed when the code was extended; the extension made the shortcut easier: commit 0ccbaa0 (2026-04-05)
  added `is_commit` (`_has_hash`, `dataset_utils.py:496-501`) and `commit_urgency`, which mark exactly the events
  that can be fixes. The original code has neither (0 matches for `_has_hash|is_commit`).
- Why our checks did not catch it: they tested consistency (determinism, threshold leakage, commit lists, logs vs
  results), not the sampling design. The `event_type` key added in Phase 3 is what made it visible.

## 3. Pre-flight scan of the raw training data (`00_preflight/raw_type_scan.txt`)
- 2,656 repos, 1,085 with a vulnerable event; all 2,886 vulnerable events are type Commit.
- No repo with fixes lacks benign Commit candidates, and none has fewer than its number of fixes, so ratio 1 is
  possible everywhere in training.
- Unexpected, reported and NOT changed ("change nothing else"): the raw type is "Commit", but the one-hot list
  `dataset_utils.event_types` has "commits" (never present in the raw data) and no "Commit". Commit events therefore
  get an all-zero type one-hot; their type is still visible through `is_commit` and the other columns.

## 4. Code change (diffs in `01_code_changes/*.diff`, originals and sha256 in `01_code_changes/originals/`)
- `dataset_utils.create_dataset(..., benign_commit_only=False)`: when True, a temporary row marker
  `_target_is_commit = (type == "Commit")` is set before `fix_repo_shape` (which drops `type`), benign candidates are
  `Vuln == 0 & marker == 1`, then the same `random.shuffle` and cut as before, and the marker is dropped together with
  `Vuln` before any window is cut. Default False = byte-identical behaviour (test T1).
- `extract_dataset` passes the flag and refuses a mismatch: the flag requires "commitonly" in the cache name and only
  the flag may use it, so a Commit-only cache can never be mistaken for a Phase 3 one or vice versa.
- `train.py --benign-commit-only` (passes the flag). Caches/models: `*_commitonly[_sSEED]` via `--comment commitonly`.
- `evaluate_cvefixes.py --benign-commit-only --model-comment commitonly`: same marker and filter in balanced and
  realistic mode; same per-repo seeded draw; model folder name gets the comment (same order as training); the two
  flags must agree or it exits.
- Nothing else changed: features, windows, padding, split, models, training, thresholds, metrics.

## 5. Tests (all run before any real run; outputs in `01_code_changes/tests/*.txt`)
- `test_commit_only_dataset` (8 smallest Phase 3 repos, sym and causal): T1 default path byte-identical to the
  original file; T2 Commit-only: same columns, same repos, fix windows identical after the global padding train.py
  applies, every benign target Commit and not a fix, benign = fixes per repo; T3 reproducible; T4 name guard. ALL PASS.
  Found while testing: a fix near the start of a repo gives an empty window; the original padding turns it into 40
  zero rows, with Commit-only benign targets it stays 0 rows until the global padding, after which it is identical.
- `test_commit_only_external` (6 smallest near-domain repos): default path identical (balanced and realistic);
  Commit-only fixes unchanged, benign targets all Commit and provenance-verified, reproducible; realistic = fixes +
  all non-fix Commit windows; model folder names match training. ALL PASS. Note: 3 of these 6 small external repos
  have fewer Commit benign windows than fixes, so external sets can be slightly below 1:1 (counted in the results).
- `test_split_replay`: the split replay used by the gate reproduces the real Phase 3 test repos for three caches
  (sym s42 no metadata, causal s123 no metadata, sym s7 metadata). ALL PASS.

## 6. Route chosen (most identical to Phase 3)
- Caches built ahead on CPU with the Phase 3 prebuild route (`02_caches/prebuild_cache.py`, copied from
  `runs_v2/05_burst_rate_pin/prebuild_nometa_caches.py`; init(seed) then extract_dataset) and loaded by train.py, as
  in Phase 3. train.py must log "Loading Dataset <cache>" or the result is not used (building a cache inside train.py
  changes the split RNG, runs_v2/NOTES.md s6).
- Split gate per cache (`02_caches/split_check.py`): replays train.py's seed, load, pad_and_fix shuffle and
  split_repos for the new and the Phase 3 cache; PASS only if train+val and test repos are the same, in the same
  order, with the same fixes per repo, and benign = fixes. The test-repo sha256 it writes answers Task 5.
- Training and external commands are the Phase 3 ones plus the flags above (`run_v3.sh`, run_config).
- Checkpointing: `PAUSE_AT` (2026-11-21 09:45); no job starts unless its estimated length (Phase 3 medians x 1.6)
  ends before then; resume with `bash rerun_v3/resume.sh`; finished runs are skipped; an interrupted run is redone
  and its log kept as `*.interrupted.log`.

## 7. Runner bug at launch (fix 1, 2026-11-20 22:02)
- At 21:59 the first two caches finished and both split gates wrote VERDICT: PASS
  (`02_caches/split_checks/split_Aggregate.{before_cve,only_before}_R1_B20_commitonly.txt`: same 902 train+val and
  128 test repos in the same order, same fixes, benign = fixes). The runner still logged "no verdict" and FAIL for
  the six seed-42 no-metadata configurations: in `split_ok` the verdict path was set in the same `local` statement
  as NAME, so it expanded with an empty NAME (`split_.txt`). Nothing was trained or scored.
- Fixed in `run_v3.sh` (two separate statements; old copy `run_v3.before_fix1.sh`), tested on the two finished caches
  (rc 0) and an unbuilt one (rc 2); all other `local` lines checked for the same pattern (none). The six FAILED
  markers were removed; the summary log keeps the wrong FAIL lines with the FIX 1 note after them.
- First split results (seed 42, no metadata, sym): test-repo sha256 c1c3b8f2...deda8b identical for Commit-only and
  Phase 3; 2,699 fixes and 2,699 benign windows. All-zero windows: 70 of 5,398 (1.30%) Commit-only vs 33 of 5,398
  (0.61%) in Phase 3; kept, not dropped (train.py has no filter). More all-zero windows now because benign Commit
  targets near the start of a repo also give empty windows (s5).

## 8. Task 3 limited to the seed-42 models (2026-11-22 ~12:10)
- The brief names no seed for Task 3 (it does for Task 4: seed 42). All 5 seeds' models would take about 75 h with the
  Task 1 scoring command; the seed-42 models about 15 h. Anonusr: seed 42 only. `task3_draws.sh` default
  T3_SEEDS="42" (old copy `task3_draws.before_seed42only.sh`); `task3_seed42_switch.sh` lets the runner pause at a
  job boundary once the 160 seed-42 draws exist and resumes it, so Task 5 runs next. No job was stopped.
