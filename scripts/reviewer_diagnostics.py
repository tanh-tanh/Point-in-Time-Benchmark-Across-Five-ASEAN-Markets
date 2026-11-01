"""Reviewer diagnostics from public aggregate artifacts only.

Builds three groups of tables requested in review:

R1  Coverage by market and universe arm: forecast opportunities in the
    out-of-sample (OOS) test windows, valid forecasts, forecasts not made, and
    the full-panel rejection audit by reason, plus an interval-level drill-down
    for PSEI and FBM KLCI.
R2  Block-length sensitivity (10, 20, 40) of the current-minus-PIT Brier
    difference for the four models, with 95% intervals and BH-adjusted p-values.
    Block 40 has no five-market estimate; its available-market macro uses four
    markets (IDX30 excluded) and only intervals with at least 80 OOS dates.
R3  OOS dates, forecasts per arm and exact common-support pairs by market.

No model is refit and no observation-level input is read. Every derived count
is checked against the frozen published values; the script exits nonzero if
any check fails. The OOS-restricted split of exclusions by reason needs the
restricted observation panel and is produced by
``scripts/private_coverage_audit.py``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PIT = "POINT_IN_TIME"
CURRENT = "CURRENT_2026_APPLIED_BACKWARD"
INTERSECTION = "PIT_CURRENT_INTERSECTION_CONTROL"
ARM_LABEL = {
    PIT: "Point in time",
    CURRENT: "Current 2026 backward",
    INTERSECTION: "PIT-current intersection control",
}
ARM_ORDER = [PIT, CURRENT, INTERSECTION]
MARKET_ORDER = ["IDX30", "SET50", "VN30", "PSEI", "FBM_KLCI"]
MARKET_NAME = {
    "IDX30": "Indonesia",
    "SET50": "Thailand",
    "VN30": "Vietnam",
    "PSEI": "Philippines",
    "FBM_KLCI": "Malaysia",
}
MODEL_ORDER = [
    "historical_class_probability",
    "logistic_regression",
    "random_forest",
    "lightgbm",
]
MODEL_LABEL = {
    "historical_class_probability": "Historical probability",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}
REASON_ORDER = [
    "MISSING_PREDICTION_RAW_OHLC",
    "NO_NEXT_VALID_SAME_SECURITY_OBSERVATION",
    "LONG_TARGET_GAP",
    "TARGET_OUTSIDE_ACTIVE_MEMBERSHIP",
    "MISSING_FEATURES",
]
DEEP_DIVE = ["PSEI", "FBM_KLCI"]
# Original engine (run_pit_v8_1_release._hierarchical_bootstrap_replicates):
# an interval enters a block-L replicate only with >= L paired OOS dates, and
# block 40 additionally requires >= 80 dates (two independent blocks).
MIN_INTERVAL_DATES = {10: 10, 20: 20, 40: 80}
TOLERANCE = 1e-12


def _read(relative: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / relative, encoding="utf-8-sig")


def load_inputs() -> dict[str, pd.DataFrame]:
    return {
        "interval_perf": _read("tables/supplementary/S07_absolute_performance.csv"),
        "interval_inventory": _read("tables/supplementary/S04_interval_inventory.csv"),
        "rejections": _read("tables/supplementary/S05_rejection_audit.csv"),
        "common_support": _read("tables/supplementary/S09_common_support.csv"),
        "market_perf": _read("results/pit_multimarket_performance_by_market_v8.csv"),
        "market_inventory": _read("results/forecasting_ready_market_inventory_v8.csv"),
        "bootstrap": _read("results/pit_multimarket_primary_bootstrap_10000_v8_1_1.csv"),
        "table_04": _read("tables/table_04_absolute_brier.csv"),
        "table_06": _read("tables/table_06_common_support.csv"),
        "folds": _read("tables/supplementary/S06_fold_calendar.csv"),
        "provenance": _read("audit/raw_provenance_summary.csv"),
        "data_inventory": _read("audit/data_inventory.csv"),
        "data_schema": _read("audit/data_schema.csv"),
        "bootstrap_verification": _read("audit/bootstrap_verification.csv"),
    }


def _read_json(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _check(checks: list[dict], name: str, ok: bool, detail) -> None:
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


# ---------------------------------------------------------------- intervals
def interval_arm_counts(interval_perf: pd.DataFrame, checks: list[dict]) -> pd.DataFrame:
    """One row per interval x arm with OOS dates, forecasts and coverage.

    Forecast sets are identical across the four models in the frozen release;
    this is verified rather than assumed.
    """
    keys = ["market", "index_id", "interval_id", "universe_semantics"]
    spread = interval_perf.groupby(keys).agg(
        dates_min=("oos_trading_dates", "min"),
        dates_max=("oos_trading_dates", "max"),
        count_min=("prediction_count", "min"),
        count_max=("prediction_count", "max"),
        coverage_min=("prediction_coverage", "min"),
        coverage_max=("prediction_coverage", "max"),
        models=("model_id", "nunique"),
    )
    identical = bool(
        spread.dates_min.eq(spread.dates_max).all()
        and spread.count_min.eq(spread.count_max).all()
        and (spread.coverage_max - spread.coverage_min).abs().max() <= TOLERANCE
        and spread.models.eq(4).all()
    )
    _check(checks, "forecast_counts_identical_across_four_models", identical, int(len(spread)))
    out = spread.reset_index().rename(
        columns={
            "dates_min": "oos_dates",
            "count_min": "forecasts",
            "coverage_min": "reported_coverage",
        }
    )[keys + ["oos_dates", "forecasts", "reported_coverage"]]
    dates = out.pivot_table(index="interval_id", columns="universe_semantics", values="oos_dates")
    _check(
        checks,
        "oos_dates_identical_across_arms_within_interval",
        bool(dates.nunique(axis=1).eq(1).all()),
        int(len(dates)),
    )
    return out


def constituents_per_date(
    interval_arms: pd.DataFrame,
    interval_inventory: pd.DataFrame,
    checks: list[dict],
) -> pd.DataFrame:
    """Daily denominators of prediction coverage for all three arms.

    PIT membership is constant within an interval, so every arm's daily
    constituent count is constant within an interval:
    PIT: active member-days / trading dates of each ready interval (S04).
    Current: the 2026 roster is applied to every PIT date, so the denominator
    is the roster size, recovered as the largest forecasts-per-date ratio.
    Intersection: forecasts / reported coverage / OOS dates.
    Each is accepted only if it is an integer and reproduces every reported
    interval coverage.
    """
    inventory = interval_inventory.copy()
    inventory["pit_members_per_date"] = (
        inventory.required_active_member_days / inventory.trading_dates
    )
    pit = interval_arms[interval_arms.universe_semantics.eq(PIT)].merge(
        inventory[["interval_id", "pit_members_per_date"]], on="interval_id", validate="one_to_one"
    )
    pit["constituents_per_date"] = pit.pit_members_per_date
    current = interval_arms[interval_arms.universe_semantics.eq(CURRENT)].copy()
    roster = (
        (current.forecasts / current.oos_dates).groupby(current.index_id).max().round().astype(int)
    )
    current["constituents_per_date"] = current.index_id.map(roster).astype(float)
    intersection = interval_arms[interval_arms.universe_semantics.eq(INTERSECTION)].copy()
    intersection["constituents_per_date"] = (
        intersection.forecasts / intersection.reported_coverage / intersection.oos_dates
    )
    out = pd.concat([pit.drop(columns="pit_members_per_date"), current, intersection], ignore_index=True)
    integer = bool(
        np.allclose(out.constituents_per_date, out.constituents_per_date.round(), rtol=0, atol=1e-9)
    )
    _check(checks, "constant_integer_daily_constituent_count", integer, roster.to_dict())
    out["constituents_per_date"] = out.constituents_per_date.round()
    out["opportunities"] = out.oos_dates * out.constituents_per_date
    out["recomputed_coverage"] = out.forecasts / out.opportunities
    error = float((out.recomputed_coverage - out.reported_coverage).abs().max())
    _check(checks, "interval_coverage_reproduced_all_arms", error <= TOLERANCE, error)
    out["not_forecast"] = out.opportunities - out.forecasts
    return out


# ----------------------------------------------------------------------- R1
def build_r1_coverage(
    interval_rows: pd.DataFrame,
    market_perf: pd.DataFrame,
    checks: list[dict],
) -> pd.DataFrame:
    rows = []
    reported = (
        market_perf.groupby(["index_id", "universe_semantics"]).prediction_coverage.agg(["min", "max"])
    )
    _check(
        checks,
        "market_coverage_identical_across_models",
        bool(((reported["max"] - reported["min"]).abs() <= TOLERANCE).all()),
        int(len(reported)),
    )
    for (index_id, arm), group in interval_rows.groupby(["index_id", "universe_semantics"]):
        equal_weight = float(group.recomputed_coverage.mean())
        reported_cov = float(reported.loc[(index_id, arm), "min"])
        opportunities = float(group.opportunities.sum())
        forecasts = int(group.forecasts.sum())
        per_date = group.constituents_per_date.unique()
        rows.append(
            {
                "index_id": index_id,
                "market": MARKET_NAME[index_id],
                "universe_semantics": arm,
                "arm": ARM_LABEL[arm],
                "ready_intervals": int(group.interval_id.nunique()),
                "oos_dates": int(group.oos_dates.sum()),
                "constituents_per_oos_date": int(per_date[0]) if len(per_date) == 1 else np.nan,
                "constituents_per_oos_date_range": (
                    f"{int(per_date.min())}-{int(per_date.max())}" if len(per_date) > 1 else str(int(per_date[0]))
                ),
                "forecast_opportunities": opportunities,
                "valid_forecasts_per_model": forecasts,
                "not_forecast": opportunities - forecasts,
                "pooled_coverage": forecasts / opportunities,
                "not_forecast_share": 1 - forecasts / opportunities,
                "equal_weight_coverage_recomputed": equal_weight,
                "equal_weight_coverage_reported": reported_cov,
                "opportunity_basis": (
                    "Sum over intervals of OOS forecast dates x constant daily constituent count; "
                    "reproduces reported coverage. Excludes scheduled test dates with no forecast "
                    "(one IDX30 date)."
                ),
            }
        )
    out = pd.DataFrame(rows)
    for column in ["constituents_per_oos_date", "forecast_opportunities", "not_forecast"]:
        out[column] = out[column].round().astype("Int64")
    error = float(
        (out.equal_weight_coverage_recomputed - out.equal_weight_coverage_reported).abs().max()
    )
    _check(checks, "market_equal_weight_coverage_reproduced", error <= TOLERANCE, error)
    return _order(out, arm=True)


def build_r1_rejections(rejections: pd.DataFrame) -> pd.DataFrame:
    """S05 reorganised by reason. Scope is the full membership panel."""
    out = rejections[~rejections.valid_for_model.astype(bool)].copy()
    out["market"] = out.index_id.map(MARKET_NAME)
    out["arm"] = out.universe_semantics.map(ARM_LABEL)
    out["scope"] = "ALL_MEMBER_DAYS_ALL_INTERVALS_NOT_RESTRICTED_TO_OOS"
    out["reason_order"] = out.rejection_reason.map({r: i for i, r in enumerate(REASON_ORDER)})
    out = out.rename(columns={"rows": "rejected_member_days", "securities": "securities_affected"})
    out = _order(out, arm=True, extra=["reason_order"])
    return out[
        [
            "index_id",
            "market",
            "universe_semantics",
            "arm",
            "rejection_reason",
            "rejected_member_days",
            "securities_affected",
            "scope",
        ]
    ]


def build_r1_interval_detail(
    interval_rows: pd.DataFrame, interval_inventory: pd.DataFrame
) -> pd.DataFrame:
    inventory = interval_inventory[interval_inventory.index_id.isin(DEEP_DIVE)].copy()
    inventory["pit_member_days_without_price"] = (
        inventory.required_active_member_days - inventory.covered_active_member_days
    )
    inventory["pit_members_without_price_per_date"] = (
        inventory.pit_member_days_without_price / inventory.trading_dates
    )
    base = inventory[
        [
            "index_id",
            "interval_id",
            "interval_start",
            "interval_end_exclusive",
            "trading_dates",
            "pit_security_count",
            "eligible_for_forecasting",
            "price_coverage",
            "target_pair_coverage",
            "pit_member_days_without_price",
            "pit_members_without_price_per_date",
            "blocking_reasons",
        ]
    ].rename(
        columns={
            "eligible_for_forecasting": "data_ready",
            "price_coverage": "pit_price_coverage_full_interval",
            "target_pair_coverage": "pit_target_pair_coverage_full_interval",
        }
    )
    oos = interval_rows[interval_rows.index_id.isin(DEEP_DIVE)]
    wide = None
    for arm, prefix in [(PIT, "pit"), (CURRENT, "current"), (INTERSECTION, "intersection")]:
        part = oos[oos.universe_semantics.eq(arm)].set_index("interval_id")
        part = part.assign(not_forecast_per_date=part.not_forecast / part.oos_dates)
        columns = {
            "constituents_per_date": f"{prefix}_constituents_per_date",
            "opportunities": f"{prefix}_oos_opportunities",
            "forecasts": f"{prefix}_oos_forecasts",
            "not_forecast": f"{prefix}_oos_not_forecast",
            "not_forecast_per_date": f"{prefix}_oos_not_forecast_per_date",
            "reported_coverage": f"{prefix}_oos_coverage",
        }
        keep = (["oos_dates"] if arm == PIT else []) + list(columns)
        part = part[keep].rename(columns=columns)
        wide = part if wide is None else wide.join(part)
    out = base.merge(wide.reset_index(), on="interval_id", how="left", validate="one_to_one")
    out.insert(1, "market", out.index_id.map(MARKET_NAME))
    return out.sort_values(["index_id", "interval_start"], key=_market_key).reset_index(drop=True)


# ----------------------------------------------------------------------- R2
def interval_deltas(interval_rows: pd.DataFrame, interval_perf: pd.DataFrame) -> pd.DataFrame:
    pivot = interval_perf.pivot_table(
        index=["index_id", "interval_id", "model_id"], columns="universe_semantics", values="brier_score"
    )
    delta = (pivot[CURRENT] - pivot[PIT]).rename("delta_brier").reset_index()
    dates = interval_rows[interval_rows.universe_semantics.eq(PIT)][["interval_id", "oos_dates"]]
    return delta.merge(dates, on="interval_id", validate="many_to_one")


def build_r2_scope(interval_rows: pd.DataFrame) -> pd.DataFrame:
    out = interval_rows[interval_rows.universe_semantics.eq(PIT)][
        ["index_id", "interval_id", "oos_dates"]
    ].copy()
    out.insert(1, "market", out.index_id.map(MARKET_NAME))
    for block, minimum in MIN_INTERVAL_DATES.items():
        out[f"in_block_{block}"] = out.oos_dates >= minimum
    return out.sort_values(["index_id", "interval_id"], key=_market_key).reset_index(drop=True)


def build_r2_bootstrap(
    bootstrap: pd.DataFrame,
    scope: pd.DataFrame,
    deltas: pd.DataFrame,
    checks: list[dict],
) -> pd.DataFrame:
    boot = bootstrap[bootstrap.metric.eq("brier_score")].copy()
    family_size = boot[boot.valid.astype(bool)].groupby("family_id").size()
    macro = boot[boot.inference_level.ne("market")].copy()

    # Verify full-support point estimates from public interval Brier scores and
    # the interval-eligibility rule; this pins down which intervals each
    # block length used.
    max_error = 0.0
    for block, minimum in MIN_INTERVAL_DATES.items():
        eligible = deltas[deltas.oos_dates >= minimum]
        market_points = eligible.groupby(["model_id", "index_id"]).delta_brier.mean()
        macro_points = market_points.groupby("model_id").mean()
        saved = macro[
            macro.support_specification.eq("FULL_SUPPORT")
            & macro.block_length.eq(block)
            & macro.valid.astype(bool)
        ].set_index("model_id").point_estimate
        max_error = max(max_error, float((macro_points - saved).abs().max()))
        saved_market = boot[
            boot.support_specification.eq("FULL_SUPPORT")
            & boot.inference_level.eq("market")
            & boot.block_length.eq(block)
            & boot.valid.astype(bool)
        ].set_index(["model_id", "index_id"]).point_estimate
        max_error = max(max_error, float((market_points - saved_market).abs().max()))
        if set(market_points.index) != set(saved_market.index):
            max_error = float("inf")
    _check(checks, "bootstrap_point_estimates_reproduced_from_interval_brier", max_error <= TOLERANCE, max_error)

    rows = []
    for record in macro.to_dict("records"):
        block = int(record["block_length"])
        valid = bool(record["valid"])
        excluded = record.get("excluded_markets")
        excluded = [] if pd.isna(excluded) or not valid else str(excluded).split(";")
        markets = [m for m in MARKET_ORDER if m not in excluded] if valid else []
        used = scope[scope.index_id.isin(markets) & scope[f"in_block_{block}"]]
        dropped = scope[scope.index_id.isin(markets) & ~scope[f"in_block_{block}"]]
        if record["family_id"] == "CONFIRMATORY_MACRO_BRIER_BLOCK20":
            note = "Primary confirmatory test."
        elif not valid:
            note = "Not estimable: IDX30 has no interval with >= 80 OOS dates (39 and 40 dates)."
        elif block == 40:
            note = (
                "Four markets only (IDX30 excluded); also drops intervals with < 80 OOS dates. "
                "Exploratory; not a five-market result."
            )
        else:
            note = "Exploratory block-length sensitivity; same five-market scope as block 20."
        if (
            valid
            and record["result_classification"] == "NO_SUPPORTED_DIFFERENCE"
            and (record["ci_lower"] > 0 or record["ci_upper"] < 0)
        ):
            note += " CI excludes zero but BH-adjusted p > 0.05, so the joint gate is not passed."
        rows.append(
            {
                "support_specification": record["support_specification"],
                "model_id": record["model_id"],
                "model": MODEL_LABEL[record["model_id"]],
                "block_length": block,
                "macro_scope": record["macro_scope"],
                "role": record["confirmatory_or_exploratory"],
                "markets_required": 5 if record["macro_scope"] == "FIVE_MARKET_MACRO" else np.nan,
                "markets_used": len(markets) if valid else np.nan,
                "markets": ";".join(markets),
                "excluded_markets": "IDX30" if block == 40 else "",
                "intervals_used": len(used) if valid else np.nan,
                "intervals_dropped_within_used_markets": ";".join(dropped.interval_id),
                "oos_dates_used": int(used.oos_dates.sum()) if valid else np.nan,
                "current_minus_pit_brier": record["point_estimate"],
                "ci95_lower": record["ci_lower"],
                "ci95_upper": record["ci_upper"],
                "raw_p_value": record["raw_p_value"],
                "bh_adjusted_p_value": record["adjusted_p_value"],
                "bh_family_id": record["family_id"],
                "bh_family_size": int(family_size.get(record["family_id"], 0)),
                "bootstrap_replicates": int(record["bootstrap_samples"]),
                "bootstrap_seed": int(record["bootstrap_seed"]),
                "classification": record["result_classification"],
                "note": note,
            }
        )
    out = pd.DataFrame(rows)
    for column in ["markets_required", "markets_used", "intervals_used", "oos_dates_used"]:
        out[column] = out[column].astype("Int64")
    support_rank = {"FULL_SUPPORT": 0, "COMMON_SUPPORT": 1}
    scope_rank = {"FIVE_MARKET_MACRO": 0, "AVAILABLE_MARKET_MACRO": 1}
    out = out.sort_values(
        ["support_specification", "model_id", "block_length", "macro_scope"],
        key=lambda s: (
            s.map(support_rank)
            if s.name == "support_specification"
            else s.map({m: i for i, m in enumerate(MODEL_ORDER)})
            if s.name == "model_id"
            else s.map(scope_rank)
            if s.name == "macro_scope"
            else s
        ),
    ).reset_index(drop=True)
    confirm = out[out.bh_family_id.eq("CONFIRMATORY_MACRO_BRIER_BLOCK20")]
    _check(
        checks,
        "confirmatory_family_is_four_five_market_block20",
        bool(len(confirm) == 4 and confirm.markets_used.eq(5).all() and confirm.bh_family_size.eq(4).all()),
        int(len(confirm)),
    )
    four = out[out.macro_scope.eq("AVAILABLE_MARKET_MACRO")]
    _check(
        checks,
        "block40_available_macro_has_four_markets",
        bool(len(four) == 8 and four.markets_used.eq(4).all() and four.block_length.eq(40).all()),
        int(len(four)),
    )
    for family, group in out[out.raw_p_value.notna()].groupby("bh_family_id"):
        p = group.raw_p_value.to_numpy()
        order = np.argsort(p)
        n = len(p)
        adjusted = np.minimum(1, np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1])
        error = float(np.abs(adjusted - group.bh_adjusted_p_value.to_numpy()[order]).max())
        if n != group.bh_family_size.iloc[0] or error > 1e-14:
            _check(checks, f"bh_reproduced_{family}", False, error)
            break
    else:
        _check(checks, "bh_adjusted_p_reproduced_within_each_macro_family", True, None)
    return out


def build_r2_decomposition(bootstrap: pd.DataFrame) -> pd.DataFrame:
    """Separate the block-40 change into market exclusion and interval exclusion.

    Point estimates only. A same-scope bootstrap (blocks 10/20/40 on the
    block-40 scope) needs date-level differences: see private_coverage_audit.py.
    """
    boot = bootstrap[bootstrap.metric.eq("brier_score") & bootstrap.valid.astype(bool)]
    rows = []
    for (support, model), group in boot.groupby(["support_specification", "model_id"]):
        market20 = group[group.inference_level.eq("market") & group.block_length.eq(20)].set_index("index_id")
        five20 = group[group.macro_scope.eq("FIVE_MARKET_MACRO") & group.block_length.eq(20)].point_estimate
        four40 = group[group.macro_scope.eq("AVAILABLE_MARKET_MACRO") & group.block_length.eq(40)].point_estimate
        four_all = float(market20.drop(index="IDX30").point_estimate.mean())
        rows.append(
            {
                "support_specification": support,
                "model_id": model,
                "model": MODEL_LABEL[model],
                "five_market_all_intervals": float(five20.iloc[0]),
                "four_market_all_intervals": four_all,
                "four_market_block40_intervals": float(four40.iloc[0]),
                "change_from_dropping_idx30": four_all - float(five20.iloc[0]),
                "change_from_dropping_short_intervals": float(four40.iloc[0]) - four_all,
            }
        )
    out = pd.DataFrame(rows)
    out["support_rank"] = out.support_specification.map({"FULL_SUPPORT": 0, "COMMON_SUPPORT": 1})
    out["model_rank"] = out.model_id.map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["support_rank", "model_rank"]).drop(columns=["support_rank", "model_rank"]).reset_index(drop=True)


# ----------------------------------------------------------------------- R3
def build_r3(
    r1: pd.DataFrame,
    common_support: pd.DataFrame,
    market_inventory: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    checks: list[dict],
) -> pd.DataFrame:
    cs = common_support[common_support.aggregation_level.eq("market")]
    spread = cs.groupby("index_id").common_support_count.agg(["min", "max"])
    _check(
        checks,
        "common_support_identical_across_models",
        bool(spread["min"].eq(spread["max"]).all() and cs.groupby("index_id").model_id.nunique().eq(4).all()),
        spread["min"].to_dict(),
    )
    forecasts = r1.pivot_table(index="index_id", columns="universe_semantics", values="valid_forecasts_per_model")
    oos = r1[r1.universe_semantics.eq(PIT)].set_index("index_id")
    inventory = market_inventory.set_index("index_id")
    rows = []
    for index_id in MARKET_ORDER:
        common = int(spread.loc[index_id, "min"])
        pit = int(forecasts.loc[index_id, PIT])
        current = int(forecasts.loc[index_id, CURRENT])
        rows.append(
            {
                "index_id": index_id,
                "market": MARKET_NAME[index_id],
                "ready_intervals": int(inventory.loc[index_id, "data_ready_intervals"]),
                "valid_folds": int(inventory.loc[index_id, "valid_continuous_folds"]),
                "oos_dates": int(oos.loc[index_id, "oos_dates"]),
                "pit_forecasts": pit,
                "current_forecasts": current,
                "intersection_arm_forecasts": int(forecasts.loc[index_id, INTERSECTION]),
                "common_support_pairs": common,
                "pit_only_forecasts": pit - common,
                "current_only_forecasts": current - common,
                "common_share_of_pit": common / pit,
                "common_share_of_current": common / current,
            }
        )
    out = pd.DataFrame(rows)
    _check(
        checks,
        "oos_dates_match_market_inventory",
        bool(out.set_index("index_id").oos_dates.eq(inventory.loc[MARKET_ORDER, "oos_trading_dates"]).all()),
        out.oos_dates.tolist(),
    )
    total = {
        "index_id": "ALL",
        "market": "Five markets (sum)",
        **{c: int(out[c].sum()) for c in out.columns if c not in {"index_id", "market"} and not c.startswith("common_share")},
    }
    total["common_share_of_pit"] = total["common_support_pairs"] / total["pit_forecasts"]
    total["common_share_of_current"] = total["common_support_pairs"] / total["current_forecasts"]
    out = pd.concat([out, pd.DataFrame([total])], ignore_index=True)
    t4 = tables["table_04"].groupby("Universe arm")["Collapsed predictions"].agg(["min", "max"])
    expected = {
        PIT: t4.loc["Point in time"],
        CURRENT: t4.loc["Current 2026 backward"],
        INTERSECTION: t4.loc["PIT-current intersection"],
    }
    observed = {
        PIT: total["pit_forecasts"],
        CURRENT: total["current_forecasts"],
        INTERSECTION: total["intersection_arm_forecasts"],
    }
    _check(
        checks,
        "arm_forecast_totals_match_table_04",
        all(expected[a]["min"] == expected[a]["max"] == observed[a] for a in ARM_ORDER),
        {a: observed[a] for a in ARM_ORDER},
    )
    pairs = tables["table_06"]["Matched pairs per model"]
    _check(
        checks,
        "common_support_total_matches_table_06",
        bool(pairs.eq(total["common_support_pairs"]).all()),
        total["common_support_pairs"],
    )
    return out


# ------------------------------------------------------- R1 interval gate
def build_r1_interval_gate(
    interval_inventory: pd.DataFrame, rejections: pd.DataFrame, checks: list[dict]
) -> pd.DataFrame:
    """First cut on coverage: the interval data gate, computed on the PIT roster.

    An interval is forecast in every arm only if PIT price and target-pair
    coverage are both >= 0.95 (S04). Every PIT missing-price rejection in S05
    must equal an unpriced PIT member-day in S04, ready or not.
    """
    inv = interval_inventory.copy()
    inv["unpriced"] = inv.required_active_member_days - inv.covered_active_member_days
    ready = inv.eligible_for_forecasting.astype(bool)
    rows = []
    for index_id in MARKET_ORDER:
        m = inv[inv.index_id.eq(index_id)]
        r = ready[m.index]
        failed = m[~r]
        rows.append(
            {
                "index_id": index_id,
                "market": MARKET_NAME[index_id],
                "pit_intervals": int(len(m)),
                "ready_intervals": int(r.sum()),
                "not_ready_intervals": int((~r).sum()),
                "required_pit_member_days": int(m.required_active_member_days.sum()),
                "unpriced_pit_member_days_ready": int(m[r].unpriced.sum()),
                "unpriced_pit_member_days_not_ready": int(failed.unpriced.sum()),
                "not_ready_price_coverage_min": float(failed.price_coverage.min()) if len(failed) else np.nan,
                "not_ready_price_coverage_max": float(failed.price_coverage.max()) if len(failed) else np.nan,
                "not_ready_failing_price_gate": int((failed.price_coverage < 0.95).sum()),
                "not_ready_failing_target_gate_only": int(
                    ((failed.price_coverage >= 0.95) & (failed.target_pair_coverage < 0.95)).sum()
                ),
            }
        )
    out = pd.DataFrame(rows)
    s05 = rejections[
        rejections.universe_semantics.eq(PIT) & rejections.rejection_reason.eq("MISSING_PREDICTION_RAW_OHLC")
    ].set_index("index_id").rows
    unpriced = out.set_index("index_id")
    total = unpriced.unpriced_pit_member_days_ready + unpriced.unpriced_pit_member_days_not_ready
    _check(
        checks,
        "s05_pit_missing_price_equals_s04_unpriced_member_days",
        bool(total.eq(s05.reindex(total.index)).all()),
        {k: int(v) for k, v in total.items()},
    )
    return out


def build_r1_price_sources(provenance: pd.DataFrame) -> pd.DataFrame:
    """Raw OHLC sources by market (audit/raw_provenance_summary.csv)."""
    out = provenance[
        ["market", "provider", "price_semantics", "rows", "securities", "date_min", "date_max"]
    ].copy()
    order = {MARKET_NAME[m]: i for i, m in enumerate(MARKET_ORDER)}
    out["_m"] = out.market.map(order)
    return out.sort_values(["_m", "date_min"]).drop(columns="_m").reset_index(drop=True)


# ------------------------------------------------------ R2 market level
def build_r2_market_level(bootstrap: pd.DataFrame) -> pd.DataFrame:
    boot = bootstrap[bootstrap.metric.eq("brier_score") & bootstrap.inference_level.eq("market")].copy()
    out = boot[
        ["support_specification", "index_id", "model_id", "block_length", "valid", "point_estimate",
         "ci_lower", "ci_upper", "raw_p_value", "adjusted_p_value", "family_id", "result_classification"]
    ].rename(columns={"point_estimate": "current_minus_pit_brier", "ci_lower": "ci95_lower",
                      "ci_upper": "ci95_upper", "adjusted_p_value": "bh_adjusted_p_value",
                      "family_id": "bh_family_id", "result_classification": "classification"})
    out.insert(2, "market", out.index_id.map(MARKET_NAME))
    out["_s"] = out.support_specification.map({"FULL_SUPPORT": 0, "COMMON_SUPPORT": 1})
    out["_m"] = out.index_id.map({m: i for i, m in enumerate(MARKET_ORDER)})
    out["_o"] = out.model_id.map({m: i for i, m in enumerate(MODEL_ORDER)})
    return out.sort_values(["_s", "_m", "_o", "block_length"]).drop(columns=["_s", "_m", "_o"]).reset_index(drop=True)


# ------------------------------------------------------- R4 horizon 5
H5 = 5


def _sessions_after_last_test(row: pd.Series) -> tuple[float, str]:
    """Trading sessions between an interval's last test date and its end."""
    start = pd.Timestamp(row.interval_start)
    if abs((pd.Timestamp(row.training_start) - start).days) <= 7:
        # Market history begins with this interval: fold positions are counted from it.
        return float(row.trading_dates - row.last_fold_position), "fold_position_from_history_start"
    if abs((pd.Timestamp(row.first_test) - start).days) <= 7:
        # Test windows start at the interval's first trading date and tile it.
        return float(row.trading_dates - row.scheduled_test_dates), "trading_dates_minus_test_dates"
    return np.nan, "undetermined"


def build_r4_h5_scope(
    interval_rows: pd.DataFrame, interval_inventory: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expected effect of a 5-session label on the OOS calendar (estimate only).

    Holding folds, rosters and the interval gate fixed, a longer label removes
    test dates only where fewer than five sessions follow the last test date:
    at the end of a market's history every arm loses those dates; at an
    internal rebalance only securities leaving the PIT roster lose them (PIT
    and intersection arms). Exact counts come from the h = 5 run itself.
    """
    valid = folds[folds.valid_fold.astype(bool)].copy()
    valid["position"] = valid.training_dates + valid.calibration_dates + valid.test_dates
    by_interval = valid.groupby("interval_id").agg(
        folds=("fold_id", "size"),
        scheduled_test_dates=("test_dates", "sum"),
        first_test=("test_start", "min"),
        last_test=("test_end", "max"),
        training_start=("training_start", "min"),
        last_fold_position=("position", "max"),
    )
    inv = interval_inventory.set_index("interval_id")
    last_interval = inv.groupby("index_id").interval_end_exclusive.transform("max").eq(inv.interval_end_exclusive)
    h1 = interval_rows[interval_rows.universe_semantics.eq(PIT)].set_index("interval_id")
    rows = []
    for interval_id, f in by_interval.iterrows():
        row = pd.concat([f, inv.loc[interval_id, ["index_id", "interval_start", "interval_end_exclusive", "trading_dates"]]])
        after, method = _sessions_after_last_test(row)
        end_of_history = bool(last_interval.loc[interval_id])
        h1_dates = int(h1.loc[interval_id, "oos_dates"])
        if end_of_history and np.isfinite(after):
            h5_dates = int(row.scheduled_test_dates - max(0, H5 - after))
            h1_check = int(row.scheduled_test_dates - max(0, 1 - after))
        else:
            h5_dates, h1_check = h1_dates, h1_dates
        rows.append(
            {
                "index_id": row.index_id,
                "market": MARKET_NAME[row.index_id],
                "interval_id": interval_id,
                "interval_start": row.interval_start,
                "interval_end_exclusive": row.interval_end_exclusive,
                "valid_folds": int(row.folds),
                "scheduled_test_dates": int(row.scheduled_test_dates),
                "last_test_date": row.last_test,
                "sessions_after_last_test": after,
                "sessions_method": method,
                "end_of_market_history": end_of_history,
                "h1_oos_forecast_dates": h1_dates,
                "h1_rule_reproduces_oos_dates": h1_check == h1_dates,
                "h5_expected_oos_dates": h5_dates,
                "h5_rebalance_exposed": bool(not end_of_history and (not np.isfinite(after) or after < H5)),
                "h5_block40_eligible": h5_dates >= MIN_INTERVAL_DATES[40],
            }
        )
    detail = pd.DataFrame(rows).sort_values(["index_id", "interval_start"], key=_market_key).reset_index(drop=True)
    pit_forecasts = h1.groupby("index_id").forecasts.sum()
    summary = []
    for index_id in MARKET_ORDER:
        m = detail[detail.index_id.eq(index_id)]
        h1_dates, h5_dates = int(m.h1_oos_forecast_dates.sum()), int(m.h5_expected_oos_dates.sum())
        approx_forecasts = float(pit_forecasts[index_id]) * h5_dates / h1_dates
        summary.append(
            {
                "index_id": index_id,
                "market": MARKET_NAME[index_id],
                "valid_folds": int(m.valid_folds.sum()),
                "h1_oos_dates": h1_dates,
                "h5_expected_oos_dates": h5_dates,
                "sessions_after_final_test": float(m[m.end_of_market_history].sessions_after_last_test.iloc[0]),
                "rebalance_exposed_intervals": int(m.h5_rebalance_exposed.sum()),
                "h5_pit_forecasts_approx": round(approx_forecasts),
                "passes_validity_gate_approx": bool(
                    m.valid_folds.sum() >= 2 and h5_dates >= 40 and approx_forecasts >= 800
                ),
                "h5_block40_intervals": int(m.h5_block40_eligible.sum()),
            }
        )
    return pd.DataFrame(summary), detail


# --------------------------------------------------------- audit appendix
def build_audit_cross_checks(
    data: dict[str, pd.DataFrame], r3: pd.DataFrame, checks: list[dict]
) -> pd.DataFrame:
    """Public aggregates against counts the audit took on restricted files."""
    inventory = data["data_inventory"].assign(name=lambda d: d.path.str.split("/").str[-1]).set_index("name").rows
    total = r3.set_index("index_id").loc["ALL"]
    arm_sum = int(total.pit_forecasts + total.current_forecasts + total.intersection_arm_forecasts)
    rejections = data["rejections"]
    feature_rejections = int(rejections[rejections.rejection_reason.eq("MISSING_FEATURES")].rows.sum())
    target_rejections = int(rejections.rows.sum()) - feature_rejections
    observations = int(inventory["pit_multimarket_observation_panel_v8.parquet"])
    support = _read_json("audit/temporal_and_support_checks.json")
    schema = data["data_schema"]
    null_reasons = int(
        schema[schema.path.str.endswith("fold_calendar_v8.parquet") & schema.column.eq("invalid_reason")].null_count.iloc[0]
    )
    s06_valid = int(data["folds"].valid_fold.astype(bool).sum())
    boot = data["bootstrap_verification"]
    violations = sum(int(v) for k, v in support.items() if k != "exact_common_support_pairs")
    rows = [
        ("seed_collapsed_predictions", "4 models x (PIT + current + intersection)", 4 * arm_sum,
         int(inventory["pit_multimarket_predictions_seed_collapsed_v8.parquet"]), "audit/data_inventory.csv"),
        ("seed_level_predictions", "5 seeds x seed-collapsed", 20 * arm_sum,
         int(inventory["pit_multimarket_predictions_seed_level_v8.parquet"]), "audit/data_inventory.csv"),
        ("matched_current_pit_pairs", "4 models x common-support pairs", 4 * int(total.common_support_pairs),
         int(support["exact_common_support_pairs"]), "audit/temporal_and_support_checks.json"),
        ("membership_rows", "observation rows + target-stage rejections (S05)", observations + target_rejections,
         int(inventory["pit_membership_daily_all_arms_v8.parquet"]), "audit/data_inventory.csv"),
        ("feature_rows", "observation rows - MISSING_FEATURES (S05)", observations - feature_rejections,
         int(inventory["pit_multimarket_feature_panel_v8.parquet"]), "audit/data_inventory.csv"),
        ("valid_folds", "valid folds in S06", s06_valid, null_reasons,
         "audit/data_schema.csv (fold calendar invalid_reason nulls)"),
        ("raw_price_rows", "sum of rows over price sources", int(data["provenance"].rows.sum()),
         int(inventory["pit_historical_raw_ohlc_recovered_v8.parquet"]), "audit/data_inventory.csv"),
        ("fbm_klci_pit_member_days", "FBM KLCI required PIT member-days (S04)",
         int(data["interval_inventory"].query("index_id == 'FBM_KLCI'").required_active_member_days.sum()),
         int(inventory["fbm_klci_membership_daily_v8.parquet"]), "audit/data_inventory.csv"),
        ("full_support_bootstrap_rows_verified", "FULL_SUPPORT rows in S08",
         int(data["bootstrap"].support_specification.eq("FULL_SUPPORT").sum()), int(boot.cells.min()),
         "audit/bootstrap_verification.csv"),
        ("temporal_or_probability_violations", "expected zero", 0, violations,
         "audit/temporal_and_support_checks.json"),
    ]
    out = pd.DataFrame(rows, columns=["check", "public_derivation", "from_public_tables", "from_audit", "audit_source"])
    out["status"] = np.where(out.from_public_tables.eq(out.from_audit), "MATCH", "MISMATCH")
    _check(checks, "audit_cross_checks_match", bool(out.status.eq("MATCH").all()),
           out.set_index("check").status.to_dict())
    _check(checks, "audit_bootstrap_verification_within_tolerance",
           bool(boot.status.eq("TOLERANCE_MATCH").all()), float(boot.max_abs_error.max()))
    return out


# -------------------------------------------------------------------- utils
def _market_key(series: pd.Series) -> pd.Series:
    if series.name == "index_id":
        return series.map({m: i for i, m in enumerate(MARKET_ORDER)})
    return series


def _order(frame: pd.DataFrame, *, arm: bool = False, extra: list[str] | None = None) -> pd.DataFrame:
    out = frame.copy()
    out["_m"] = out.index_id.map({m: i for i, m in enumerate(MARKET_ORDER)})
    keys = ["_m"]
    if arm:
        out["_a"] = out.universe_semantics.map({a: i for i, a in enumerate(ARM_ORDER)})
        keys.append("_a")
    keys += extra or []
    out = out.sort_values(keys).drop(columns=[k for k in ["_m", "_a"] if k in out] + (extra or []))
    return out.reset_index(drop=True)


def _fmt(value, spec: str) -> str:
    if value is None or value is pd.NA or (isinstance(value, (float, np.floating)) and not np.isfinite(value)):
        return "n/a"
    if spec == "int":
        return f"{int(round(value)):,}"
    if spec == "pct":
        return f"{100 * value:.2f}%"
    if spec == "brier":
        return f"{value:+.6f}"
    if spec == "p":
        return f"{value:.4f}"
    if spec == "f2":
        return f"{value:.2f}"
    if spec == "yesno":
        return "Yes" if bool(value) else "No"
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def markdown_table(frame: pd.DataFrame, columns: dict[str, tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for label, _ in columns.values()) + " |"
    rule = "|" + "|".join("---" for _ in columns) + "|"
    lines = [header, rule]
    for record in frame.to_dict("records"):
        lines.append("| " + " | ".join(_fmt(record[c], spec) for c, (_, spec) in columns.items()) + " |")
    return "\n".join(lines)


def write_summary(path: Path, outputs: dict[str, pd.DataFrame]) -> None:
    r1 = outputs["R1_coverage_by_market_arm"]
    r1_show = r1
    rej = outputs["R1_full_panel_rejections_by_reason"]
    rej_wide = rej.pivot_table(
        index=["index_id", "universe_semantics"], columns="rejection_reason",
        values="rejected_member_days", fill_value=0,
    ).reindex(columns=REASON_ORDER, fill_value=0).reset_index()
    rej_wide["arm"] = rej_wide.universe_semantics.map(ARM_LABEL)
    rej_wide = _order(rej_wide, arm=True)
    detail = outputs["R1_interval_detail_PSEI_FBM_KLCI"]
    r2 = outputs["R2_bootstrap_block_sensitivity"]
    r2_full = r2[r2.support_specification.eq("FULL_SUPPORT")]
    dec = outputs["R2_block40_decomposition"]
    r3 = outputs["R3_oos_forecasts_common_support"]
    gate = outputs["R1_interval_gate_by_market"]
    sources = outputs["R1_price_sources"]
    sources = sources[sources.market.isin(["Philippines", "Malaysia"])]
    market_boot = outputs["R2_bootstrap_market_level"]
    h5 = outputs["R4_h5_expected_scope_by_market"]
    audit = outputs["A_audit_cross_checks"]
    supported = market_boot[market_boot.classification.eq("SUPPORTED_CURRENT_PIT_DIFFERENCE")]
    parts = [
        "# Reviewer diagnostics (generated)",
        "",
        "Generated by `scripts/reviewer_diagnostics.py` from public aggregate artifacts. "
        "Do not edit by hand. Interpretation: `docs/REVIEWER_DIAGNOSTICS.md`.",
        "",
        "## R1a. Interval data gate (PIT roster; decides the intervals of every arm)",
        "",
        markdown_table(
            gate,
            {
                "index_id": ("Index", "str"),
                "pit_intervals": ("PIT intervals", "int"),
                "ready_intervals": ("Ready", "int"),
                "not_ready_intervals": ("Not ready", "int"),
                "required_pit_member_days": ("Required member-days", "int"),
                "unpriced_pit_member_days_ready": ("Unpriced (ready)", "int"),
                "unpriced_pit_member_days_not_ready": ("Unpriced (not ready)", "int"),
                "not_ready_price_coverage_min": ("Not-ready price cov. min", "pct"),
                "not_ready_price_coverage_max": ("Not-ready price cov. max", "pct"),
            },
        ),
        "",
        "Unpriced PIT member-days (ready + not ready) equal the PIT `MISSING_PREDICTION_RAW_OHLC` "
        "rejections in S05 for every market.",
        "",
        "### Price sources, Philippines and Malaysia (`audit/raw_provenance_summary.csv`)",
        "",
        markdown_table(
            sources,
            {
                "market": ("Market", "str"),
                "provider": ("Provider", "str"),
                "rows": ("Rows", "int"),
                "securities": ("Securities", "int"),
                "date_min": ("From", "str"),
                "date_max": ("To", "str"),
            },
        ),
        "",
        "## R1. OOS forecast coverage by market and arm",
        "",
        "Forecast opportunities = constituent security-dates on OOS forecast dates "
        "(daily constituent counts are constant within each interval). "
        "Counts are per model (identical for all four models).",
        "",
        markdown_table(
            r1_show,
            {
                "index_id": ("Index", "str"),
                "arm": ("Arm", "str"),
                "oos_dates": ("OOS dates", "int"),
                "constituents_per_oos_date_range": ("Constituents/date", "str"),
                "forecast_opportunities": ("Opportunities", "int"),
                "valid_forecasts_per_model": ("Valid forecasts", "int"),
                "not_forecast": ("Not forecast", "int"),
                "pooled_coverage": ("Pooled coverage", "pct"),
                "equal_weight_coverage_reported": ("Reported coverage (eq. weight)", "pct"),
            },
        ),
        "",
        "### R1b. Rejected member-days by reason (full membership panel, all intervals; not OOS-restricted)",
        "",
        markdown_table(
            rej_wide,
            {"index_id": ("Index", "str"), "arm": ("Arm", "str"),
             **{r: (r, "int") for r in REASON_ORDER}},
        ),
        "",
        "### R1c. PSEI and FBM KLCI by interval",
        "",
        markdown_table(
            detail,
            {
                "index_id": ("Index", "str"),
                "interval_start": ("Start", "str"),
                "data_ready": ("Data-ready", "yesno"),
                "pit_price_coverage_full_interval": ("PIT price coverage", "pct"),
                "pit_members_without_price_per_date": ("PIT members w/o price per date", "f2"),
                "oos_dates": ("OOS dates", "int"),
                "pit_oos_not_forecast": ("PIT not forecast", "int"),
                "pit_oos_not_forecast_per_date": ("PIT not forecast/date", "f2"),
                "current_oos_not_forecast": ("Current not forecast", "int"),
                "current_oos_not_forecast_per_date": ("Current not forecast/date", "f2"),
                "intersection_oos_not_forecast": ("Intersection not forecast", "int"),
            },
        ),
        "",
        "## R2. Block-length sensitivity, full support, current minus PIT Brier",
        "",
        "Block 40 has no five-market estimate (IDX30 has 39 and 40 OOS dates per interval, "
        "below the 80-date minimum). The block-40 row uses **four markets only** and also "
        "drops shorter intervals in the remaining markets.",
        "",
        markdown_table(
            r2_full,
            {
                "model": ("Model", "str"),
                "block_length": ("Block", "int"),
                "macro_scope": ("Scope", "str"),
                "markets_used": ("Markets", "int"),
                "intervals_used": ("Intervals", "int"),
                "current_minus_pit_brier": ("Brier diff", "brier"),
                "ci95_lower": ("95% CI low", "brier"),
                "ci95_upper": ("95% CI high", "brier"),
                "raw_p_value": ("Raw p", "p"),
                "bh_adjusted_p_value": ("BH p", "p"),
                "role": ("Role", "str"),
                "classification": ("Classification", "str"),
            },
        ),
        "",
        "### R2b. Why block-40 estimates differ (point estimates)",
        "",
        markdown_table(
            dec,
            {
                "support_specification": ("Support", "str"),
                "model": ("Model", "str"),
                "five_market_all_intervals": ("5 markets, all intervals", "brier"),
                "four_market_all_intervals": ("4 markets, all intervals", "brier"),
                "four_market_block40_intervals": ("4 markets, block-40 intervals", "brier"),
            },
        ),
        "",
        f"Market-level cells (`R2_bootstrap_market_level.csv`): {len(supported)} of "
        f"{int(market_boot.valid.astype(bool).sum())} valid cells are SUPPORTED, all in common support "
        f"({', '.join(sorted(set(supported.support_specification)))}).",
        "",
        "## R3. OOS dates, forecasts and common support by market",
        "",
        markdown_table(
            r3,
            {
                "index_id": ("Index", "str"),
                "oos_dates": ("OOS dates", "int"),
                "valid_folds": ("Valid folds", "int"),
                "pit_forecasts": ("PIT forecasts", "int"),
                "current_forecasts": ("Current forecasts", "int"),
                "intersection_arm_forecasts": ("Intersection-arm forecasts", "int"),
                "common_support_pairs": ("Common-support pairs", "int"),
                "common_share_of_pit": ("Share of PIT", "pct"),
                "common_share_of_current": ("Share of current", "pct"),
            },
        ),
        "",
        "## R4. Horizon 5: expected OOS scope (estimate from the public fold calendar)",
        "",
        "Same folds, rosters and interval gate; only the label changes. End-of-history dates are lost "
        "in every arm; at internal rebalances only securities leaving the PIT roster lose dates. "
        "Exact counts come from the h = 5 run.",
        "",
        markdown_table(
            h5,
            {
                "index_id": ("Index", "str"),
                "valid_folds": ("Valid folds", "int"),
                "h1_oos_dates": ("OOS dates h=1", "int"),
                "h5_expected_oos_dates": ("Expected OOS dates h=5", "int"),
                "sessions_after_final_test": ("Sessions after final test", "int"),
                "rebalance_exposed_intervals": ("Rebalance-exposed intervals", "int"),
                "h5_pit_forecasts_approx": ("PIT forecasts h=5 (approx.)", "int"),
                "passes_validity_gate_approx": ("Validity gate", "yesno"),
                "h5_block40_intervals": ("Block-40 intervals", "int"),
            },
        ),
        "",
        "## Appendix A. Public tables against audit counts on restricted files",
        "",
        markdown_table(
            audit,
            {
                "check": ("Check", "str"),
                "public_derivation": ("Public derivation", "str"),
                "from_public_tables": ("Public", "int"),
                "from_audit": ("Audit", "int"),
                "audit_source": ("Audit source", "str"),
                "status": ("Status", "str"),
            },
        ),
        "",
    ]
    path.write_text("\n".join(parts), encoding="utf-8")


def build(output: Path) -> dict:
    data = load_inputs()
    checks: list[dict] = []
    arms = interval_arm_counts(data["interval_perf"], checks)
    interval_rows = constituents_per_date(arms, data["interval_inventory"], checks)
    r1 = build_r1_coverage(interval_rows, data["market_perf"], checks)
    scope = build_r2_scope(interval_rows)
    deltas = interval_deltas(interval_rows, data["interval_perf"])
    outputs = {
        "R1_coverage_by_market_arm": r1,
        "R1_full_panel_rejections_by_reason": build_r1_rejections(data["rejections"]),
        "R1_interval_detail_PSEI_FBM_KLCI": build_r1_interval_detail(interval_rows, data["interval_inventory"]),
        "R2_bootstrap_block_sensitivity": build_r2_bootstrap(data["bootstrap"], scope, deltas, checks),
        "R2_block40_decomposition": build_r2_decomposition(data["bootstrap"]),
        "R2_interval_scope_by_block": scope,
        "R3_oos_forecasts_common_support": build_r3(
            r1, data["common_support"], data["market_inventory"], data, checks
        ),
    }
    outputs["R1_interval_gate_by_market"] = build_r1_interval_gate(
        data["interval_inventory"], data["rejections"], checks
    )
    outputs["R1_price_sources"] = build_r1_price_sources(data["provenance"])
    outputs["R2_bootstrap_market_level"] = build_r2_market_level(data["bootstrap"])
    h5_summary, h5_detail = build_r4_h5_scope(interval_rows, data["interval_inventory"], data["folds"])
    _check(
        checks,
        "h5_scope_rule_reproduces_h1_oos_dates",
        bool(h5_detail.h1_rule_reproduces_oos_dates.all() and h5_detail.sessions_method.ne("undetermined").all()),
        int(h5_detail.h1_rule_reproduces_oos_dates.sum()),
    )
    outputs["R4_h5_expected_scope_by_market"] = h5_summary
    outputs["R4_h5_expected_scope_by_interval"] = h5_detail
    outputs["A_audit_cross_checks"] = build_audit_cross_checks(
        data, outputs["R3_oos_forecasts_common_support"], checks
    )
    output.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    write_summary(output / "SUMMARY.md", outputs)
    status = "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL"
    result = {
        "scope": "PUBLIC_AGGREGATE_REVIEWER_DIAGNOSTICS",
        "model_training": False,
        "observation_level_inputs": False,
        "outputs": sorted(f"{name}.csv" for name in outputs) + ["SUMMARY.md"],
        "checks": checks,
        "status": status,
    }
    (output / "verification.json").write_text(
        json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "tables" / "reviewer_diagnostics",
        help="output directory (default: tables/reviewer_diagnostics)",
    )
    args = parser.parse_args(argv)
    result = build(args.output.resolve())
    for check in result["checks"]:
        print(f"{check['status']}  {check['check']}")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
