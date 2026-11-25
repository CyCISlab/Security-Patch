# Commit-Time Behavioural Security Patch Detection — artefact (part 2)

This folder complements the files already in this repository (raw GraphQL responses `graphql.zip.part_*`,
commit event records `json_commits.zip.part_*`, CVE-to-commit mappings `gh_cve_proccessed.zip.part_*`,
the external CVEfixes test set `cvefixes_test.zip.part_*`, `timezones.zip`, and the data-collection scripts
`create_dataset.py`, `graphql.py`, `utils.py`). It adds the final training/evaluation code, all run outputs
reported in the paper, and the commit-message baseline.

`Analyzing-Behavioural-Data Code.zip` and `Seed results.zip` in the repository root are the earlier release;
`code/model/` here supersedes the code in that zip.

## Layout

| Path | Contents |
|---|---|
| `code/model/` | Final training and evaluation code (`train.py`, `dataset_utils.py`, `evaluate_cvefixes.py`, `models.py`, `model_outputs.py`, `provenance.py`, `ablation.py`, `operational_metrics.py`, `leakage_free_eval.py`, statistics and plotting scripts, `requirements.txt`). `code/pipeline/GIT_STATE.txt` records the exact working tree and file hashes used for the final runs |
| `code/pipeline/` | Exact runner scripts and commands for the event-type-matched runs (`run_v3.sh`, `exact_commands.txt`, `prebuild_cache.py`, `split_check.py`, `task*_*.sh`), environment record, code hashes (`SHA256_after.txt`), full run log, and diffs against the previous release |
| `code/tests/` | Unit/replay tests run before training (commit-only sampling, split replay) and pre-flight data scans |
| `code/analysis/` | Commit-message baseline (`commit_message_baseline.py`) |
| `data/commit_messages/` | Commit lists (`urls_core.tsv`, `urls_extra.tsv`), retrieved and cleaned messages (`messages_clean.tsv`), unavailable commits (`failed.tsv`), and the retrieval script (`fetch_messages.sh`) |
| `data/repo_metadata.json` | Static repository metadata snapshot used for the metadata feature block |
| `results/matched_sampling/` | Event-type-matched runs (headline results). See its `README.md` |
| `results/original_sampling/` | Same configurations under the original all-event sampling, plus determinism, feature, ablation, OOM-fix and audit checks |
| `results/commit_message_baseline/` | Message-only / behaviour-only / combined results |
| `results/temporal/` | Full-corpus commits by period (`temporal_by_period.csv`) and the figure |

Note: `results/matched_sampling/README.md` refers to `config/` and `code/`; these are `code/pipeline/` and `code/tests/` here.

## Paper section → files

| Paper | Files |
|---|---|
| §5.8 Reproducibility (determinism, provenance, cache rebuild) | `results/original_sampling/01_determinism/`, `results/matched_sampling/task5/`, `code/tests/test_split_replay.py`, `code/pipeline/split_check.py` |
| §6.1 RQ1, Table 7 (four architectures, B = 20) | `results/matched_sampling/task1/`, `results/matched_sampling/summary/summary.csv` |
| §6.2 Paired t-tests and TOST | Per-seed values in `results/matched_sampling/summary/summary.csv` (tests are paired by seed over these rows) |
| §6.3 RQ2 causal vs symmetric windows | `results/matched_sampling/task1/` (`*_causal_*` vs `*_sym_*`) |
| §6.4 Inflation from unmatched sampling; window sweep, Figure 3 | `results/original_sampling/` (`runs/task1_*`, `runs/task3_window_*`, `runs/task3_recurrent_*`) |
| §6.5 External validation, benign redraws, realistic prevalence | `results/matched_sampling/task1/ext_*`, `results/matched_sampling/task3/`, `results/matched_sampling/task4/` |
| §6.6 Commit-message comparison | `code/analysis/commit_message_baseline.py`, `data/commit_messages/`, `results/commit_message_baseline/` |
| §6.7 RQ3 feature-group ablation | `results/matched_sampling/task2/`, `results/original_sampling/runs/task3_ablation_*` |
| Figure 4 (full corpus by year) | `results/temporal/` |

## Prediction files

Each internal `preds_*.npz` holds the ensemble and per-fold test scores (`test_pred`, `test_logit`, `test_pred_folds`),
labels (`test_y`), the validation-chosen threshold (`tau_val`), validation outputs (`val_pred`, `val_y`, `val_fold`), and
the provenance record of every window (`*_repo`, `*_hash`, `*_event_type`, `*_created_at`, `*_event_idx`,
`*_prov_verified`). Each external `ext_near_*` / `ext_unseen_*.npz` holds `scores`, `logits`, `pred_folds`, `labels`,
`threshold` and the same provenance fields. File names encode architecture, window mode (`sym`/`causal`), window size
(`b20`), seed (`s42`, ...) and whether the metadata block is excluded (`nometa`).

## Reassembling archives

The `results/` folder is distributed as `results.tar.gz.part_00` ... `part_11` (GitHub upload limit). Rebuild it with:

    sha256sum -c results_parts.sha256
    cat results.tar.gz.part_* > results.tar.gz && tar -xzf results.tar.gz


Repository-root data archives are split for hosting limits:

    cat graphql.zip.part_* > graphql.zip   # likewise json_commits, gh_cve_proccessed, cvefixes_test

Here, the full-corpus prediction files are split the same way:

    cat fullcorpus_preds_transformer_sym_b20_s42_nometa.npz.part_* > fullcorpus_preds_transformer_sym_b20_s42_nometa.npz

The `results/original_sampling/runs/*.tar.gz` and `results/matched_sampling/task5/cache_build_logs.tar.gz` files are ordinary
archives (`tar -xzf`).

## Running

    pip install -r code/model/requirements.txt
    # exact training commands for every reported configuration:
    cat code/pipeline/exact_commands.txt
    # commit-message baseline (from this folder's root):
    python code/analysis/commit_message_baseline.py transformer_causal

`MANIFEST.sha256` lists a SHA-256 for every file in this folder (`sha256sum -c MANIFEST.sha256`).
