# Original PIT data recovery report — 1 November 2026

All ten datasets in the unchanged authoritative [data inventory](data_inventory.csv) were recovered exactly. Every SHA-256 and byte size matches, all recorded structural checks pass, and both requested original verification drivers executed successfully. No dataset-specific public redistribution authorization was authenticated. All ten Parquet files remain local. This update contains recovery documentation, provenance metadata and aggregate validation receipts only.

## Authority and storage

Study: **Do Current-Constituent Universes Distort Equity Forecasting Evidence? A Point-in-Time Benchmark Across Five ASEAN Markets**. Lineage: `paper_outputs/five_market_pit_universe_benchmark_v8_1_1/`; analytical tag `pit-five-market-v8.1.1-final`; commit `8777babb77b0666604923d35d3f490462a040fd6`.

`ORIGINAL_RESEARCH_ROOT` is the original local VSEF research tree. `ARCHIVED_PIT_RELEASE_ROOT` is `PIT_BENCHMARK_PUBLICATION_20261026/private/replay`. `RECOVERED_EVIDENCE_ROOT` is the isolated `PIT_DATA_RECOVERY_20261101/evidence` directory beside that original research tree. The [public recovery manifest](data_recovery_manifest.csv) records portable source/destination paths. The local `private/data_recovery_manifest.csv` retains exact resolved absolute locations; workstation paths and credentials are not published.

The original tree contained all ten exact Parquet files. The archived release supplied frozen analytical controls and original code. Available PIT raw prices, source caches, membership/event logs, manifests, QA, schemas, snapshots, earlier PIT dependencies and imported original helper modules were copied additively. Shared price/Philippine archive inputs were selected only by references and expected checksums in the original OHLC dataset. No other study's observations or outputs were substituted.

The recovered root contains **1,586 files / 431,215,982 bytes**. The ten requested datasets total **93,044,183 bytes (88.73 MiB)**. All **195** V8.1.1 analytical manifest records match. There are zero primary-dataset checksum/size mismatches and zero conflicting source copies. All recovered and copied source files remain byte-identical after execution. The complete source/destination supporting inventory and post-execution preservation hashes are retained locally. [Public supporting summary](recovered_supporting_evidence_summary.csv).

The search inventoried 79,161 available workspace paths, inspected five relevant PIT ZIP archives without modifying them, and hash-tested 208 candidate cache copies. No matching pre-cleanup backup directory was present at the inspected D: root. Available Git history contained neither the missing cache directories nor the two absent historical V6 test paths. No placeholder, reserialization, estimated OHLC, synthetic substitute or invented membership event was created.

## Dataset outcomes

| Dataset | Original SHA match | Recovery | Reproduction check | Published |
|---|---|---|---|---|
| Historical OHLC | Yes | EXACT_RECOVERY | Structural PASS | No: rights unestablished |
| Membership daily | Yes | EXACT_RECOVERY | Metrics PASS | No: rights unestablished |
| Membership intervals | Yes | EXACT_RECOVERY | Structural PASS | No: rights unestablished |
| Feature panel | Yes | EXACT_RECOVERY | Bounded refit PASS | No: rights unestablished |
| Fold calendar | Yes | EXACT_RECOVERY | Bounded refit PASS | No: rights unestablished |
| Observation panel | Yes | EXACT_RECOVERY | Structural PASS | No: rights unestablished |
| Seed-collapsed predictions | Yes | EXACT_RECOVERY | Metrics/bootstrap PASS | No: rights unestablished |
| Seed-level predictions | Yes | EXACT_RECOVERY | Bounded refit PASS | No: rights unestablished |
| KLCI daily membership | Yes | EXACT_RECOVERY | Structural PASS | No: rights unestablished |
| KLCI intervals | Yes | EXACT_RECOVERY | Structural PASS | No: rights unestablished |

Structural PASS covers byte size; recorded rows/columns, ordered names, Arrow dtypes and null counts; unique/null-free primary keys; markets and relevant dates; OHLC validity; full batch decoding; Parquet footer/magic and metadata. The fold calendar has no single date range in the original inventory; named training/calibration/test dates and first-valid-fold selection are checked by the refit driver. Structural checks do not claim fresh acquisition or reconstruction.

## Original verification executed

The unchanged drivers ran successfully against the recovered root:

```sh
python scripts/verify_private_predictions.py --evidence-root RECOVERED_EVIDENCE_ROOT --output reproduced/private_audit
python scripts/retrain_sample.py --evidence-root RECOVERED_EVIDENCE_ROOT --output reproduced/private_audit
```

Both exit codes were **0**; additional completion gates inspected every output. The isolated virtual environment reused installed compatible packages and the existing PyArrow target. Runtime: Python 3.13.5, pandas 3.0.1, NumPy 2.2.6, scikit-learn 1.8.0, SciPy 1.16.1, PyArrow 23.0.1. Pandas and SciPy differ from the historical lock; this is a recorded compatible runtime, not lockfile identity. No global packages or original source files were changed.

All 12 date/interval/market/macro Brier/log-loss/coverage comparisons pass at `1e-12`; maximum error **5.5511151231257837e-16**. The full-support 10,000-replication bootstrap matches all 76 rows at `1e-12`; maximum checked error **1.1102230246251563e-16**. Temporal/probability violations are zero, collapsed seed counts are five, and exact common-support pairing is 343,792.

The bounded refit covers the first valid fold of IDX30, SET50, VN30, PSEI and FBM_KLCI, all three original arms, four original models and five original seeds: **300 executions and 190,800 probabilities**. Prediction counts match and maximum absolute prediction error is **3.8230529852967276e-13**, within the original `1e-10` threshold. The nonzero discrepancy occurs in logistic regression; it is not reported as exact prediction equality. Fold/model/seed/arm identities, training counts and prediction counts match the existing release audit; errors satisfy its original tolerance. No every-fold retraining or model redesign was performed.

Receipts: [summary](DATA_RECOVERY_VALIDATION.json), [structure](data_recovery_structure.json), [metrics](recovered_evidence_metric_verification.csv), [bootstrap](recovered_evidence_bootstrap_verification.csv), [bounded refits](recovered_evidence_sample_model_retraining.csv). Original command/stdout/stderr receipts and absolute paths remain local under `PIT_DATA_RECOVERY_20261101/private/`.

## Supporting evidence still missing

Of **447** distinct stored provider/path/checksum references, **396** have exact source files in the recovered layout. **51** upstream daily caches remain `NOT_FOUND`: 35 Yahoo and 16 VNstock files. Their [exact paths and expected hashes](unrecovered_raw_source_dependencies.csv) are retained. Differing candidates were not substituted. These absent sources do not prevent the executed checks on the exact analytical panels, but prevent a claim of complete historical raw-acquisition reconstruction.

`tests/point_in_time/test_pit_multimarket_forecasting_v6.py` and `tests/test_manifest_portability_v6.py` remain unavailable. The broad historical suite was not rerun here and is not labeled PASS. The original December 2017 Malaysian anchor PDF remains absent; its stored factual extraction is preserved. The Tier 3 2018 event remains attributed evidence. Original 5014 missing prices, block-40 support and other scientific limits are unchanged.

**PASS**: exact ten-dataset recovery, structural checks, requested metric/bootstrap replay and bounded refits. **PARTIAL**: full upstream/end-to-end reconstruction. **BLOCKED**: public dataset redistribution pending suitable authorization evidence.

## Redistribution gate

The [per-file rights register](data_recovery_rights.csv) distinguishes vendor OHLC, constituent histories, researcher observations/features, predictions, reconstructed KLCI records and experiment metadata. All ten retain `WITHHELD_RIGHTS_UNESTABLISHED`. This records missing evidence of an appropriate public grant; it is not a finding that redistribution is legally prohibited.

The fold calendar is researcher-generated experiment metadata containing dates/counts/flags/reasons, without OHLC columns or a security-level roster. The original routine derives it from source membership dates and interval eligibility. The available evidence does not authenticate the applicable source-term clearance for publishing this exact derived file. Prediction files likewise include security/date identifiers and actual source-derived class labels. Research authorship does not establish permission for included provider-derived records.

All 29 Malaysian registry entries say `REPOSITORY_LOCAL_RESEARCH_CACHE` and supply no authenticated public grant. The Philippine archive README attributes the source to Investagrams/PSE; its software Unlicense does not authenticate rights in those observations. Recorded VNstock retrieval fields do not establish upstream data permission.

Primary-source context checked on 1 November 2026: [VNstock's official README](https://github.com/thinh-vu/vnstock/blob/main/README.md) distinguishes software access from source-data licensing. [LSEG redistribution information](https://www.lseg.com/en/data-analytics/market-data/data-redistribution) and [FTSE subscriptions](https://www.lseg.com/en/ftse-russell/index-resources/indices-subscribe-data) do not supply this project's public grant. Current pages do not establish the contractual terms of historical acquisitions. Yahoo terms retrieval returned HTTP 999/429; no grant or prohibition is inferred from that failed retrieval. No current policy is applied retrospectively.

Each withheld file needs an authenticated applicable license, permission or equivalent clearance covering its exact records, source/acquisition terms, derived-data use and public GitHub distribution. No provider was contacted, and no author-controlled grant was manufactured. No recovered dataset, raw/source cache, LFS object, archive, encrypted copy or release attachment is uploaded.

## Public update and paths

`main` was fetched and inspected at `51560591afaef8238051fdc03e7a4474e8dd570f` before edits. The requested commit message is `Recover and publish verified PIT benchmark research datasets`; despite that requested wording, the dataset publication count is **0**. Final commit and remote-tree/hash checks are bound by the external post-push task receipt and completion response, avoiding a circular committed self-hash.

Published recovery-artifact paths:

- `data/README.md`
- `data/DATA_PROVENANCE.md`
- `docs/REPRODUCIBILITY.md`
- `audit/DATA_RECOVERY_REPORT.md`
- `audit/data_recovery_manifest.csv`
- `audit/PUBLICATION_AUDIT_REPORT.md`
- `audit/DATA_RECOVERY_VALIDATION.json`
- `audit/data_recovery_rights.csv`
- `audit/data_recovery_structure.json`
- `audit/recovered_evidence_metric_verification.csv`
- `audit/recovered_evidence_bootstrap_verification.csv`
- `audit/recovered_evidence_sample_model_retraining.csv`
- `audit/recovered_supporting_evidence_summary.csv`
- `audit/unrecovered_raw_source_dependencies.csv`
- `audit/PUBLIC_SHA256_MANIFEST.csv`

The delivery manifest is refreshed for the current public files; its own hash and the retained historical publication receipt remain outside that manifest, following the existing convention. The authoritative original inventory/schema, scientific outputs, previous scientific audit CSVs and .gitignore exclusions are preserved. Public scope remains an aggregate/code/audit subset (`PARTIAL_MATCH`). Local recovery does not make private Parquet files downloadable or assign a public reuse license.
