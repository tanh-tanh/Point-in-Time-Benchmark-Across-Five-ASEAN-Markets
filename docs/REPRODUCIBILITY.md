# Reproduction guide
## Public aggregate reproduction
Install `requirements.txt`, then run:
```sh
python scripts/reproduce_public.py
python -m pytest tests -q
```
These commands need no vendor data, network access, accounts or credentials. Expected outputs: eight matching table CSVs, six matching figure-source CSVs, six regenerated PNG/PDF figures, and `reproduced/verification.json`. PNG/PDF bytes may differ with rendering libraries and metadata; numerical equality governs the scientific comparison.

## Private original-output validation
With lawful access to the exact complete historical evidence in its original directory layout:
```sh
python scripts/point_in_time/run_pit_v8_1_1_release.py --validate-only
python -m pytest archive_tests/point_in_time -q
```
The first command reads the full private release and is blocked in an ordinary public clone. The second intentionally retains historical dependency failures: two V6 test files remain unavailable. Do not count missing inputs as passing tests. The historical validation checks its own original code and control files, so moving the new public wrapper into the old release manifest is inappropriate.

## Original pipeline and restricted reconstruction
V8 data construction, features, folds and model fitting are implemented in `scripts/point_in_time/pit_multimarket_forecasting_v8.py`, reusing V7/V6 routines and the explicitly imported helper modules. `run_pit_v8_1_release.py` supplies original `--validate-only`, `--rebuild-inference`, and `--full-rerun` modes; the last builds isolated prediction staging. These modes require all earlier manifests and lawful inputs named by the original scripts. No claim is made that the public subset supplies them. Do not run an acquisition/build mode against frozen evidence.

The new audit independently recalculated all collapsed-prediction Brier/log-loss/coverage summaries, replayed full-support bootstrap inference and regenerated publication assets locally. See the audit report for bounded model-refitting evidence, the initial failures, and the remaining historical gate. No full fresh data acquisition and every-fold retraining is claimed.

## Independent audit drivers
With lawfully held complete private evidence, the new audit can be rerun without changing it:
```sh
python scripts/verify_private_predictions.py --evidence-root PRIVATE_EVIDENCE_ROOT --output reproduced/private_audit
python scripts/retrain_sample.py --evidence-root PRIVATE_EVIDENCE_ROOT --output reproduced/private_audit
```
Replace PRIVATE_EVIDENCE_ROOT with the genuine evidence directory containing research_data and the original script layout. These commands require restricted inputs and are not public-clone tests.

## Recovered original evidence — 1 November 2026

The local RECOVERED_EVIDENCE_ROOT (`PIT_DATA_RECOVERY_20261101/evidence`) now contains all ten exact datasets, original scripts and available dependencies in the required layout. Pass it directly to the audit drivers above. It is not included in a public clone.

Both original commands executed successfully. [The new receipt](../audit/DATA_RECOVERY_VALIDATION.json) records metric/bootstrap PASS and 300 first-valid-fold refits across five markets, three arms, four models and five seeds: 190,800 probabilities, maximum absolute error 3.8230529852967276e-13, original tolerance 1e-10. All ten structural profiles match the original inventory/schema. This is bounded reproduction, not every-fold retraining or complete raw-acquisition reconstruction.

There are 396 exact raw-source references recovered and [51 missing daily caches](../audit/unrecovered_raw_source_dependencies.csv). Two historical V6 tests remain absent; the broad suite was not rerun here. A separate virtual environment reused compatible installed packages and the existing PyArrow target. Recorded pandas/SciPy versions differ from the historical lock; the numerical checks passed in that recorded runtime.

Public observation-level reproduction remains limited by rights and absent upstream dependencies. The [recovery report](../audit/DATA_RECOVERY_REPORT.md) distinguishes local evidence, executed PASS checks, incomplete upstream reconstruction and public withholding.
