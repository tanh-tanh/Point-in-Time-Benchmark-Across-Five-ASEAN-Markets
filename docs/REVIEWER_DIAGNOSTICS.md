# Reviewer diagnostics: coverage, block-length sensitivity and common support

This note answers three review requests. Tables are generated, not hand-entered:

```sh
python scripts/reviewer_diagnostics.py          # public aggregates -> tables/reviewer_diagnostics/
python -m pytest tests -q
```

`tables/reviewer_diagnostics/SUMMARY.md` renders the tables; the CSVs hold full precision; `verification.json` records 14 checks against the frozen release (all PASS). No model is refit and no observation-level file is read. The parts that need restricted evidence come from `scripts/private_coverage_audit.py` (see the last section).

All counts are **per model**. The four models forecast the same security-dates in every arm, which the script checks. Table 1's "collapsed predictions" equal the sum over three arms × four models. Seed-level predictions are five times that.

## 1. Coverage by market and arm (R1)

### Definitions
- **Forecast opportunity**: a constituent security-date of an arm on an OOS forecast date. PIT membership is constant within an interval, so every arm's daily constituent count is too. PIT has 30 per date (SET50: 50). The current arm applies the 2026 roster to every PIT date, so it has a constant 30 (SET50: 50). The intersection arm has 20–30 depending on the interval. Opportunities equal OOS dates × constituents, summed over intervals. The script accepts each denominator only because it is an integer and reproduces every reported interval coverage exactly (all 28 intervals × 3 arms).
- **Scheduled versus forecast dates**: the four valid IDX30 folds schedule 80 test dates, but 79 carry forecasts. Published coverage, and the counts here, use the 79 forecast dates. The private audit reports both bases (`scheduled_opportunities` gives 2,400 for IDX30 PIT and current).
- **Valid forecast**: a seed-collapsed prediction scored in the benchmark.
- **Not forecast**: opportunities minus valid forecasts.
- **Reported coverage** (Table 7) is equal-weighted over dates, intervals and markets. **Pooled coverage** is the simple ratio. The two differ only through weighting.

### Results (`R1_coverage_by_market_arm.csv`)

| Index | PIT not forecast / opportunities | Current not forecast / opportunities |
|---|---|---|
| IDX30 | 40 / 2,370 | 0 / 2,370 |
| SET50 | 240 / 12,000 | 0 / 12,000 |
| VN30 | 120 / 5,400 | 0 / 5,400 |
| PSEI | 125 / 27,000 | **5,589 / 27,000** |
| FBM KLCI | 482 / 57,000 | **2,093 / 57,000** |

The intersection arm misses only 1 of 19,440 (PSEI) and 1 of 47,380 (FBM KLCI), and nothing elsewhere.

The −3.45 percentage-point macro coverage gap (current minus PIT) comes from PSEI (−19.4 pp) and FBM KLCI (−3.1 pp). In the other three markets the current arm covers 1.7–2.0 pp more.

### Exclusions by reason
`R1_full_panel_rejections_by_reason.csv` reorganises the frozen rejection audit (S05) by market, arm and reason. Its scope is **every member-day in every interval**. That includes training dates and intervals that failed the data gate, so these counts are not OOS counts. `MISSING_PREDICTION_RAW_OHLC` dominates in every market and arm: 75,291 of 77,714 rejected member-days.

The OOS-restricted split by reason (opportunities = valid forecasts + exclusions per reason, for all three arms) needs the observation-level panel. It is produced by `private_coverage_audit.py` as `R1_private_oos_coverage_by_reason.csv`.

### PSEI: data shortfall at interval level, roster selection inside OOS
`R1_interval_detail_PSEI_FBM_KLCI.csv` tracks each interval.

- **Why only 9 of 22 PSEI intervals are data-ready.** Twelve intervals from February 2021 to February 2026 fail the 95% data gate. Their PIT price coverage is 70.0–93.3%, meaning 2 to 9 *true historical members* per date have no raw OHLC. One 12-day interval (September 2017) fails only on target-pair coverage (88.6%). The gate is a protocol threshold, but missing price data for genuine PIT members trips it. The cause is the **data**, not stock selection. PSEI evaluation therefore covers 2017–2021 plus 2026.
- **The price sources explain the 2021 break** (`audit/raw_provenance_summary.csv`). Philippine prices come from an archive of 39 securities that ends on 2021-03-12, and from an open-source EOD set of 30 securities that starts on 2021-01-04. In the 2026 interval the current arm misses nothing, so all 30 roster names have 2026 prices. Only the EOD set covers 2026, and it has exactly 30 securities, so its 30 are the 2026 roster. After March 2021, PIT members outside the 2026 roster have no price source. The PSEI exclusion of 2021–2026 is therefore survivorship in the *price acquisition*, not a selection made by the benchmark.
- **PIT arm inside OOS (125 not forecast).** One member per date in the March 2017 and February 2026 intervals (120 member-days), plus 5 scattered member-days. These are data gaps for true members.
- **Current arm inside OOS (5,589 not forecast): not yet attributable from public data.** About 7 of the 30 constituents of the 2026 roster are not forecastable on each date from 2017 to August 2020. The number falls to 5.4 per date in the last 2017–2021 interval, whose test window ends on 2021-01-18, and to 0 in 2026. Two explanations fit:
  - *Selection*: the 2026 roster is applied to dates before these firms listed or traded under their identifiers.
  - *Acquisition scope*: before 2021 the only Philippine price source is the 39-security archive. If it holds only historical index members, current-only names have no pre-2021 prices whether or not they were trading.

  The private audit separates the two. It records each missing security's first observed price date and the provider that supplied it. A first price on the EOD set's first date (2021-01-04) is labelled `DATA_SCOPE_FIRST_PRICE_AT_PROVIDER_START`. A later first price inside a provider's coverage is labelled selection, pending a listing-date check. In the full panel, 12 PSEI securities have missing prices in the current arm.
- In common support, PSEI keeps 72.3% of PIT forecasts: 7,436 PIT forecasts belong to members later removed. It keeps 90.8% of current forecasts.

### FBM KLCI: one data gap in PIT, roster selection in current
- All 13 intervals are data-ready.
- **PIT (482 not forecast).** One member per date in the June 2018 and December 2018 intervals (121 + 360), plus 1. This matches the documented Malaysia Airports (5014) gap: 493 active member-days with no joined price and no imputation (see LIMITATIONS item 5 and S15). It is a **data** gap.
- **Current (2,093 not forecast).** 2 per date in 2018–mid 2020, 1 per date to mid 2024, 0.5 per date in the June 2024 interval, 0 from December 2024. The steps fall inside intervals (June–December 2020, June–December 2024). They do not fall at the Malaysian provider boundaries (2016-01-04, 2017-11-01, 2022-01-03). That fits **backward roster selection**, with names entering the price record when they list, pending the private audit and a listing-date check. In the full panel, 19 FBM securities have missing prices in the current arm.

### Attribution rules used by the private audit
| Status of an OOS opportunity | Attribution |
|---|---|
| Missing price; current arm; security not a PIT member that date; date before its first observed price; first price within 7 days of its provider's first date | `DATA_SCOPE_FIRST_PRICE_AT_PROVIDER_START` |
| Same, but first price later inside the provider's coverage | `SELECTION_BACKWARD_ROSTER_BEFORE_FIRST_PRICE` (confirm listing date) |
| Missing price, any other case (PIT member, internal gap, no series, after last price) | `DATA_PRICE_UNAVAILABLE` |
| No next same-security observation, or gap > 14 days | `DATA_TARGET_UNAVAILABLE` |
| Missing lagged features | `DATA_FEATURE_HISTORY` |
| Target date outside active membership; ticker-validity boundary | `PROTOCOL_*` |

## 2. Block-length sensitivity (R2)

`R2_bootstrap_block_sensitivity.csv` lists the frozen 10,000-replicate results (seed 271828, paired, interval-confined circular blocks, equal-market macro). For each of the four models it gives the current-minus-PIT Brier difference, the 95% interval, the raw p-value and the BH-adjusted p-value with its family. Full support is primary; common support is included for completeness.

| Model | Block 10 (5 markets) | Block 20 (5 markets, confirmatory) | Block 40 (**4 markets**, exploratory) |
|---|---|---|---|
| Historical probability | +0.000189 [−0.000039, +0.000415], BH p 0.361 | +0.000189 [−0.000018, +0.000388], BH p 0.159 | +0.000276 [+0.000087, +0.000463], BH p 0.015 |
| Logistic regression | −0.000101 [−0.000366, +0.000161], BH p 0.606 | −0.000101 [−0.000325, +0.000121], BH p 0.502 | +0.000019 [−0.000220, +0.000266], BH p 0.881 |
| Random forest | −0.000531 [−0.001326, +0.000210], BH p 0.361 | −0.000531 [−0.001125, +0.000052], BH p 0.159 | −0.000664 [−0.001313, −0.000076], BH p 0.069 |
| LightGBM | −0.000019 [−0.000835, +0.000778], BH p 0.967 | −0.000019 [−0.000710, +0.000682], BH p 0.961 | −0.000067 [−0.000685, +0.000513], BH p 0.881 |

- **Block 40 uses only four markets.** The five-market block-40 estimate is `INSUFFICIENT_INFORMATION`. The original engine admits an interval at block 40 only with at least 80 OOS dates (two independent blocks). IDX30's two intervals have 39 and 40 dates. The available-market macro therefore averages SET50, VN30, PSEI and FBM KLCI. Its four model comparisons are BH-adjusted as a separate family.
- **Block 40 also changes the intervals.** It drops the 60-date VN30 interval, the 40-date PSEI March 2017 interval and the 20-date FBM January 2018 interval. That leaves 23 of 28 intervals and 3,100 of 3,299 OOS dates (`R2_interval_scope_by_block.csv`). Blocks 10 and 20 use the same scope, so their point estimates are identical and only the intervals and p-values move.
- **Why the block-40 point estimates move** (`R2_block40_decomposition.csv`, reproduced exactly from public interval Brier scores). For historical probability: +0.000189 with five markets → +0.000188 with IDX30 removed → +0.000276 with the three short intervals removed. The shift comes from dropping short intervals, not from IDX30 and not from block length. The exploratory `SUPPORTED_CURRENT_PIT_DIFFERENCE` for historical probability at block 40 is therefore a different-scope result. It does not replace the five-market confirmatory conclusion. For random forest the block-40 interval excludes zero, but BH p = 0.069, so the joint gate is not passed.
- `private_coverage_audit.py` reruns blocks 10, 20 and 40 on the identical block-40 scope with the original engine (`R2_private_same_scope_bootstrap.csv`). That isolates the effect of block length on the intervals. It first replays the published confirmatory block-20 result as a check.

## 3. Data and common support (R3)

`R3_oos_forecasts_common_support.csv` (per model):

| Index | OOS dates | Valid folds | PIT forecasts | Current forecasts | Intersection-arm forecasts | Common-support pairs |
|---|---|---|---|---|---|---|
| IDX30 | 79 | 4 | 2,330 | 2,370 | 2,330 | 2,330 |
| SET50 | 240 | 12 | 11,760 | 12,000 | 11,520 | 11,520 |
| VN30 | 180 | 9 | 5,280 | 5,400 | 5,280 | 5,280 |
| PSEI | 900 | 45 | 26,875 | 21,411 | 19,439 | 19,439 |
| FBM KLCI | 1,900 | 95 | 56,518 | 54,907 | 47,379 | 47,379 |
| Total | 3,299 | 165 | 102,763 | 96,088 | 85,948 | 85,948 |

Common support matches PIT and current predictions on market, interval, fold, model, security, prediction date, target date and realised class. The counts equal the separately trained intersection arm's counts market by market, but the two are different objects: matched predictions from two fitted arms versus a third fitted arm.

## Restricted-evidence audit

With lawful access to the recovered evidence root:

```sh
python scripts/private_coverage_audit.py --evidence-root PRIVATE_EVIDENCE_ROOT --output reproduced/private_coverage_audit
```

It reads membership, the fold calendar, collapsed predictions, raw OHLC dates and `qa/pit_multimarket_observation_rejections_v8.parquet`. It writes:

- `R1_private_oos_coverage_by_reason.csv`: OOS opportunities, valid forecasts and exclusions per reason and per attribution, for all three arms, on forecast dates and on scheduled test dates.
- `R1_private_oos_exclusions_attributed.csv`: the same exclusions by membership role, reason, position against the price record, and attribution.
- `R1_private_security_detail_PSEI_FBM_KLCI.csv`: every excluded security, with first and last observed price dates and the provider of the first price, for the listing-date check.
- `R1_private_pit_price_gaps_*_PSEI_FBM_KLCI.csv`: unpriced PIT member-days in every interval, including those that failed the gate.
- `R3_private_common_support_by_model.csv` and the R2 same-scope bootstrap.

Its checks require that counts match the public tables and that no exclusion is left unexplained. Outputs contain security identifiers and stay out of the public repository (`reproduced/` is git-ignored). The software logic is tested on synthetic fixtures only (`tests/test_reviewer_diagnostics.py`). The script has not been run on the restricted evidence in this repository.
