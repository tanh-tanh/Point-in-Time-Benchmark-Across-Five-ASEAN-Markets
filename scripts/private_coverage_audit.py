"""Observation-level coverage attribution and same-scope bootstrap (restricted evidence).

Read-only companion to ``scripts/reviewer_diagnostics.py`` for the parts that
public aggregates cannot answer:

R1  OOS forecast opportunities, valid forecasts and exclusions by reason for
    every market and arm (including the intersection arm), with each excluded
    member-day attributed to DATA, SELECTION or PROTOCOL; security-level detail
    for PSEI and FBM KLCI; PIT price gaps across all PSEI/FBM KLCI intervals.
R2  Blocks 10, 20 and 40 re-run on one identical scope (the block-40 scope:
    four markets, intervals with >= 80 OOS dates) using the original engine,
    so the block-length effect is separated from the scope change.
R3  Exact common-support pairs, recomputed from collapsed predictions.

Requires lawfully held evidence in the original layout (``research_data/
point_in_time/processed``, ``qa`` and ``scripts/point_in_time``). Outputs
contain security identifiers: keep them out of the public repository
(``reproduced/`` and ``private/`` are git-ignored).

Usage:
    python scripts/private_coverage_audit.py --evidence-root PRIVATE_EVIDENCE_ROOT \
        --output reproduced/private_coverage_audit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
PIT = "POINT_IN_TIME"
CURRENT = "CURRENT_2026_APPLIED_BACKWARD"
INTERSECTION = "PIT_CURRENT_INTERSECTION_CONTROL"
ARMS = [PIT, CURRENT, INTERSECTION]
MARKETS = ["IDX30", "SET50", "VN30", "PSEI", "FBM_KLCI"]
DEEP_DIVE = ["PSEI", "FBM_KLCI"]
MODELS = ["historical_class_probability", "lightgbm", "logistic_regression", "random_forest"]
KEY = ["index_id", "universe_semantics", "security_id", "prediction_date"]
FORECAST = "VALID_FORECAST"
MISSING_PRICE = "MISSING_PREDICTION_RAW_OHLC"
UNEXPLAINED = "NOT_FORECAST_UNEXPLAINED"
PARTIAL = "PARTIAL_MODEL_FORECAST"
BLOCK40_MIN_DATES = 80
TOLERANCE = 1e-12
# A first observed price within this many calendar days of the start of the
# provider that supplied it is treated as the provider's coverage start, not
# as evidence about when the security began trading.
PROVIDER_START_TOLERANCE_DAYS = 7


def as_day(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series).dt.normalize()


# --------------------------------------------------------------- R1 logic
def oos_opportunities(members: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Member-days of every arm that fall inside a valid fold's test window."""
    m = members.rename(columns={"membership_date": "prediction_date"}).copy()
    m["prediction_date"] = as_day(m.prediction_date)
    f = folds[folds.valid_fold.astype(bool)][["interval_id", "fold_id", "test_start", "test_end"]].copy()
    f["test_start"] = as_day(f.test_start)
    f["test_end"] = as_day(f.test_end)
    x = m.merge(f, on="interval_id", how="inner")
    x = x[(x.prediction_date >= x.test_start) & (x.prediction_date <= x.test_end)]
    if x.duplicated(KEY).any():
        raise ValueError("a member-day falls in more than one valid OOS test window")
    return x.drop(columns=["test_start", "test_end"]).reset_index(drop=True)


def forecast_model_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    p = predictions[KEY + ["model_id"]].copy()
    p["prediction_date"] = as_day(p.prediction_date)
    return (
        p.drop_duplicates()
        .groupby(KEY)
        .model_id.nunique()
        .rename("models_forecasting")
        .reset_index()
    )


def membership_roles(members: pd.DataFrame) -> pd.DataFrame:
    """Whether each security-date is a PIT member, a current-2026 member, or both."""
    m = members.rename(columns={"membership_date": "prediction_date"})[
        ["index_id", "universe_semantics", "security_id", "prediction_date"]
    ].copy()
    m["prediction_date"] = as_day(m.prediction_date)
    keys = ["index_id", "security_id", "prediction_date"]
    pit = m[m.universe_semantics.eq(PIT)][keys].drop_duplicates().assign(in_pit=True)
    cur = m[m.universe_semantics.eq(CURRENT)][keys].drop_duplicates().assign(in_current=True)
    roles = pit.merge(cur, on=keys, how="outer")
    roles["in_pit"] = roles.in_pit.astype("boolean").fillna(False).astype(bool)
    roles["in_current"] = roles.in_current.astype("boolean").fillna(False).astype(bool)
    roles["membership_role"] = np.select(
        [roles.in_pit & roles.in_current, roles.in_pit],
        ["IN_BOTH_PIT_AND_CURRENT", "PIT_ONLY_LATER_REMOVED"],
        "CURRENT_ONLY_NOT_PIT_MEMBER_ON_DATE",
    )
    return roles[keys + ["membership_role"]]


def price_spans(raw: pd.DataFrame) -> pd.DataFrame:
    """First/last observed price per security and the provider of the first price."""
    columns = ["market", "security_id", "date"] + (["provider"] if "provider" in raw else [])
    r = raw[columns].copy()
    r["date"] = as_day(r.date)
    spans = (
        r.groupby(["market", "security_id"])
        .date.agg(first_price_date="min", last_price_date="max", price_days="count")
        .reset_index()
    )
    if "provider" not in r:
        spans["first_price_provider"] = np.nan
        spans["first_price_provider_start"] = pd.NaT
        return spans
    first = (
        r.sort_values(["market", "security_id", "date", "provider"], kind="stable")
        .drop_duplicates(["market", "security_id"])[["market", "security_id", "provider"]]
        .rename(columns={"provider": "first_price_provider"})
    )
    starts = (
        r.groupby(["market", "provider"]).date.min().rename("first_price_provider_start").reset_index()
        .rename(columns={"provider": "first_price_provider"})
    )
    return spans.merge(first, on=["market", "security_id"], validate="one_to_one").merge(
        starts, on=["market", "first_price_provider"], validate="many_to_one"
    )


def classify_price_position(frame: pd.DataFrame, spans: pd.DataFrame) -> pd.DataFrame:
    """Where an unpriced member-day sits relative to the security's price record."""
    x = frame.merge(spans, on=["market", "security_id"], how="left", validate="many_to_one")
    x["price_position"] = np.select(
        [
            x.first_price_date.isna(),
            x.prediction_date < x.first_price_date,
            x.prediction_date > x.last_price_date,
        ],
        ["NO_RAW_PRICE_SERIES", "BEFORE_FIRST_OBSERVED_PRICE", "AFTER_LAST_OBSERVED_PRICE"],
        "INTERNAL_PRICE_GAP",
    )
    lag = (x.first_price_date - x.first_price_provider_start).dt.days
    x["first_price_at_provider_start"] = lag.le(PROVIDER_START_TOLERANCE_DAYS).fillna(False).astype(bool)
    return x


def attribute(frame: pd.DataFrame) -> pd.Series:
    """DATA / SELECTION / PROTOCOL attribution of each opportunity's status.

    SELECTION is assigned only to current-arm member-days of securities that
    were not PIT members on that date and had not yet appeared in the price
    record, when the first price falls inside the supplying provider's
    coverage: the 2026 roster was applied to a date before the security traded
    under its identifier. If the first price coincides with the provider's
    coverage start, the gap reflects acquisition scope (DATA_SCOPE), e.g. an
    archive that holds only historical members. Confirm SELECTION against
    listing dates, because an unmapped ticker change looks the same.
    """
    status = frame.exclusion_reason
    missing_price = status.eq(MISSING_PRICE)
    position = frame.get("price_position", pd.Series("", index=frame.index)).fillna("")
    at_start = frame.get("first_price_at_provider_start", pd.Series(False, index=frame.index))
    at_start = at_start.astype("boolean").fillna(False).astype(bool)
    role = frame.membership_role.fillna("")
    backward_before_first = (
        missing_price
        & frame.universe_semantics.eq(CURRENT)
        & role.eq("CURRENT_ONLY_NOT_PIT_MEMBER_ON_DATE")
        & position.eq("BEFORE_FIRST_OBSERVED_PRICE")
    )
    return pd.Series(
        np.select(
            [
                status.eq(FORECAST),
                backward_before_first & at_start,
                backward_before_first,
                missing_price,
                status.isin(["NO_NEXT_VALID_SAME_SECURITY_OBSERVATION", "LONG_TARGET_GAP"]),
                status.eq("MISSING_FEATURES"),
                status.eq("TARGET_OUTSIDE_ACTIVE_MEMBERSHIP"),
                status.isin(["PREDICTION_TICKER_VALIDITY_BOUNDARY", "TARGET_TICKER_VALIDITY_BOUNDARY"]),
            ],
            [
                "FORECAST",
                "DATA_SCOPE_FIRST_PRICE_AT_PROVIDER_START",
                "SELECTION_BACKWARD_ROSTER_BEFORE_FIRST_PRICE",
                "DATA_PRICE_UNAVAILABLE",
                "DATA_TARGET_UNAVAILABLE",
                "DATA_FEATURE_HISTORY",
                "PROTOCOL_MEMBERSHIP_BOUNDARY",
                "PROTOCOL_TICKER_BOUNDARY",
            ],
            "UNEXPLAINED",
        ),
        index=frame.index,
    )


def build_status_panel(
    members: pd.DataFrame,
    folds: pd.DataFrame,
    predictions: pd.DataFrame,
    rejections: pd.DataFrame,
    raw: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    opps = oos_opportunities(members, folds)
    counts = forecast_model_counts(predictions)
    n_models = int(predictions.model_id.nunique())
    rej = rejections[KEY + ["rejection_reason"]].copy()
    rej["prediction_date"] = as_day(rej.prediction_date)
    conflicting = int(rej.groupby(KEY).rejection_reason.nunique().gt(1).sum())
    rej = rej.drop_duplicates(KEY)
    x = opps.merge(counts, on=KEY, how="left", validate="one_to_one")
    x = x.merge(rej, on=KEY, how="left", validate="one_to_one")
    x["models_forecasting"] = x.models_forecasting.fillna(0).astype(int)
    forecast_and_rejected = int((x.models_forecasting.gt(0) & x.rejection_reason.notna()).sum())
    x["exclusion_reason"] = np.where(
        x.models_forecasting.eq(n_models),
        FORECAST,
        np.where(x.models_forecasting.gt(0), PARTIAL, x.rejection_reason.fillna(UNEXPLAINED)),
    )
    x = x.merge(
        membership_roles(members), on=["index_id", "security_id", "prediction_date"],
        how="left", validate="many_to_one",
    )
    unpriced = x.exclusion_reason.eq(MISSING_PRICE)
    positioned = classify_price_position(x[unpriced], price_spans(raw))
    x["price_position"] = ""
    x["first_price_provider"] = ""
    x["first_price_at_provider_start"] = False
    x["first_price_date"] = pd.NaT
    x["last_price_date"] = pd.NaT
    x["first_price_provider_start"] = pd.NaT
    for column in ["price_position", "first_price_provider", "first_price_at_provider_start",
                   "first_price_date", "last_price_date", "first_price_provider_start"]:
        x.loc[unpriced, column] = positioned[column].to_numpy()
    x["first_price_at_provider_start"] = x.first_price_at_provider_start.astype(bool)
    x["attribution"] = attribute(x)

    p = predictions[KEY].drop_duplicates().copy()
    p["prediction_date"] = as_day(p.prediction_date)
    outside = len(p.merge(opps[KEY], on=KEY, how="left", indicator=True).query("_merge == 'left_only'"))
    # Published coverage counts only dates with at least one forecast in the
    # market; a scheduled test date with none (one IDX30 date) is kept but
    # flagged so both totals can be reported.
    forecast_dates = p[["index_id", "prediction_date"]].drop_duplicates().assign(forecast_date=True)
    x = x.merge(forecast_dates, on=["index_id", "prediction_date"], how="left", validate="many_to_one")
    x["forecast_date"] = x.forecast_date.astype("boolean").fillna(False).astype(bool)
    no_forecast_dates = x[~x.forecast_date].groupby("index_id").prediction_date.nunique()
    diagnostics = {
        "models": n_models,
        "rejection_keys_with_conflicting_reasons": conflicting,
        "forecast_and_rejected_member_days": forecast_and_rejected,
        "forecasts_outside_oos_opportunity_set": outside,
        "partial_model_forecasts": int(x.exclusion_reason.eq(PARTIAL).sum()),
        "unexplained_not_forecast": int(x.exclusion_reason.eq(UNEXPLAINED).sum()),
        "scheduled_test_dates_without_any_forecast": {k: int(v) for k, v in no_forecast_dates.items()},
    }
    return x, diagnostics


def summarise_by_reason(panel: pd.DataFrame) -> pd.DataFrame:
    """Coverage by reason on OOS forecast dates (the published coverage basis).

    ``scheduled_opportunities`` also counts test dates on which the market has
    no forecast at all; those member-days are excluded from the reason columns
    and reported in ``opportunities_on_dates_without_forecast``.
    """
    scheduled = panel.groupby(["index_id", "universe_semantics"]).agg(
        scheduled_oos_dates=("prediction_date", "nunique"),
        scheduled_opportunities=("security_id", "size"),
    )
    panel = panel[panel.forecast_date]
    wide = (
        panel.pivot_table(
            index=["index_id", "universe_semantics"], columns="exclusion_reason",
            values="security_id", aggfunc="size", fill_value=0,
        )
        .reset_index()
    )
    wide.columns.name = None
    reasons = [c for c in wide.columns if c not in {"index_id", "universe_semantics", FORECAST}]
    wide["forecast_opportunities"] = wide[[FORECAST, *reasons]].sum(axis=1)
    wide["valid_forecasts"] = wide[FORECAST]
    wide["excluded"] = wide[reasons].sum(axis=1)
    wide["coverage"] = wide.valid_forecasts / wide.forecast_opportunities
    wide["oos_dates"] = wide.apply(
        lambda r: panel[panel.index_id.eq(r.index_id)].prediction_date.nunique(), axis=1
    )
    for name, rule in [("DATA", "DATA_"), ("SELECTION", "SELECTION_"), ("PROTOCOL", "PROTOCOL_")]:
        part = panel[panel.attribution.str.startswith(rule)]
        sizes = part.groupby(["index_id", "universe_semantics"]).size()
        wide[f"excluded_{name.lower()}"] = [
            int(sizes.get((a, b), 0)) for a, b in zip(wide.index_id, wide.universe_semantics)
        ]
    wide = wide.merge(scheduled.reset_index(), on=["index_id", "universe_semantics"], validate="one_to_one")
    wide["opportunities_on_dates_without_forecast"] = wide.scheduled_opportunities - wide.forecast_opportunities
    first = [
        "index_id", "universe_semantics", "oos_dates", "forecast_opportunities", "valid_forecasts",
        "excluded", "coverage", "excluded_data", "excluded_selection", "excluded_protocol",
        "scheduled_oos_dates", "scheduled_opportunities", "opportunities_on_dates_without_forecast",
    ]
    return _order(wide[first + sorted(reasons)])


def summarise_long(panel: pd.DataFrame) -> pd.DataFrame:
    excluded = panel[panel.exclusion_reason.ne(FORECAST)]
    out = (
        excluded.groupby(
            ["index_id", "universe_semantics", "forecast_date", "membership_role", "exclusion_reason",
             "price_position", "first_price_at_provider_start", "attribution"],
            dropna=False,
        )
        .agg(member_days=("security_id", "size"), securities=("security_id", "nunique"))
        .reset_index()
    )
    return _order(out)


def security_detail(panel: pd.DataFrame, markets: list[str] = DEEP_DIVE) -> pd.DataFrame:
    excluded = panel[panel.exclusion_reason.ne(FORECAST) & panel.index_id.isin(markets)]
    out = (
        excluded.groupby(
            ["index_id", "universe_semantics", "security_id", "ticker_on_date", "forecast_date",
             "membership_role", "exclusion_reason", "price_position", "attribution"],
            dropna=False,
        )
        .agg(
            member_days=("prediction_date", "size"),
            first_oos_date=("prediction_date", "min"),
            last_oos_date=("prediction_date", "max"),
            first_price_date=("first_price_date", "min"),
            last_price_date=("last_price_date", "max"),
            first_price_provider=("first_price_provider", "first"),
            first_price_provider_start=("first_price_provider_start", "min"),
        )
        .reset_index()
    )
    return _order(out).sort_values(
        ["index_id", "universe_semantics", "member_days"], ascending=[True, True, False], kind="stable",
        key=lambda s: s.map({m: i for i, m in enumerate(MARKETS)}) if s.name == "index_id"
        else s.map({a: i for i, a in enumerate(ARMS)}) if s.name == "universe_semantics" else s,
    ).reset_index(drop=True)


def pit_price_gaps(
    members: pd.DataFrame, raw: pd.DataFrame, interval_inventory: pd.DataFrame,
    markets: list[str] = DEEP_DIVE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Unpriced PIT member-days in every interval (explains the interval data gate)."""
    m = members[members.universe_semantics.eq(PIT) & members.index_id.isin(markets)].rename(
        columns={"membership_date": "prediction_date"}
    ).copy()
    m["prediction_date"] = as_day(m.prediction_date)
    r = raw[["market", "security_id", "date"]].copy()
    r["prediction_date"] = as_day(r.date)
    m = m.merge(r[["market", "security_id", "prediction_date"]].assign(priced=True),
                on=["market", "security_id", "prediction_date"], how="left", validate="many_to_one")
    gaps = classify_price_position(m[m.priced.isna()].drop(columns="priced"), price_spans(raw))
    ready = interval_inventory.set_index("interval_id").eligible_for_forecasting.astype(bool)
    gaps["data_ready_interval"] = gaps.interval_id.map(ready)
    detail = (
        gaps.groupby(["index_id", "interval_id", "data_ready_interval", "security_id", "ticker_on_date",
                      "price_position"], dropna=False)
        .agg(unpriced_member_days=("prediction_date", "size"),
             first_unpriced_date=("prediction_date", "min"),
             last_unpriced_date=("prediction_date", "max"),
             first_price_date=("first_price_date", "min"),
             last_price_date=("last_price_date", "max"),
             first_price_provider=("first_price_provider", "first"))
        .reset_index()
    )
    summary = (
        gaps.groupby(["index_id", "interval_id", "data_ready_interval", "price_position"], dropna=False)
        .agg(unpriced_member_days=("prediction_date", "size"), securities=("security_id", "nunique"))
        .reset_index()
    )
    return _order(detail), _order(summary)


# --------------------------------------------------------------- R3 logic
def common_support(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["market", "index_id", "interval_id", "fold_id", "model_id", "security_id",
            "prediction_date", "target_date", "actual_class"]
    p = predictions[keys + ["universe_semantics"]].copy()
    pit = p[p.universe_semantics.eq(PIT)].drop(columns="universe_semantics")
    cur = p[p.universe_semantics.eq(CURRENT)].drop(columns="universe_semantics")
    pairs = pit.merge(cur, on=keys, validate="one_to_one")
    counts = p.groupby(["index_id", "universe_semantics", "model_id"]).size().unstack("universe_semantics")
    counts["common_support_pairs"] = pairs.groupby(["index_id", "model_id"]).size()
    counts = counts.fillna(0).astype(int).reset_index()
    counts.columns.name = None
    dates = predictions.groupby("index_id").prediction_date.nunique().rename("oos_dates")
    counts = counts.merge(dates.reset_index(), on="index_id")
    return _order(counts.rename(columns={PIT: "pit_forecasts", CURRENT: "current_forecasts",
                                         INTERSECTION: "intersection_arm_forecasts"}))


# --------------------------------------------------------------- R2 logic
def date_level_deltas(predictions: pd.DataFrame) -> pd.DataFrame:
    keys = ["interval_id", "market", "index_id", "universe_semantics", "model_id", "prediction_date"]
    p = predictions.copy()
    p["_brier"] = (p.prediction_probability - p.actual_class) ** 2
    date = p.groupby(keys, sort=True)._brier.mean().rename("brier_score").reset_index()
    on = ["interval_id", "market", "index_id", "model_id", "prediction_date"]
    dd = date[date.universe_semantics.eq(CURRENT)].merge(
        date[date.universe_semantics.eq(PIT)], on=on, suffixes=("_current", "_pit"), validate="one_to_one"
    )
    dd["delta_brier_score_current_minus_pit"] = dd.brier_score_current - dd.brier_score_pit
    return dd


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    n = len(p)
    adjusted = np.minimum(1, np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1])
    out = np.empty(n)
    out[order] = adjusted
    return out


def same_scope_bootstrap(dd: pd.DataFrame, engine, samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blocks 10/20/40 on the block-40 scope with the original bootstrap engine."""
    lengths = dd.groupby(["index_id", "interval_id", "model_id"]).size().groupby(["index_id", "interval_id"]).min()
    eligible = lengths[lengths >= BLOCK40_MIN_DATES].reset_index()[["index_id", "interval_id"]]
    scoped = dd.merge(eligible, on=["index_id", "interval_id"])
    result = engine._bootstrap_family(scoped, support="FULL_SUPPORT", metrics=["brier_score"], samples=samples)
    result = result[result.valid.astype(bool)].copy()
    # The engine names its macro "selected_five_market_macro"; here it averages
    # only the markets left in the block-40 scope.
    result["inference_level"] = result.inference_level.replace(
        {"selected_five_market_macro": "available_market_macro"}
    )
    markets = sorted(eligible.index_id.unique(), key=MARKETS.index)
    result["scope"] = np.where(
        result.inference_level.eq("market"), "MARKET_BLOCK40_ELIGIBLE_INTERVALS",
        f"{len(markets)}_MARKET_MACRO_BLOCK40_ELIGIBLE_INTERVALS",
    )
    result["markets_in_scope"] = ";".join(markets)
    result["intervals_in_scope"] = len(eligible)
    result["oos_dates_in_scope"] = int(lengths[lengths >= BLOCK40_MIN_DATES].sum())
    result["adjusted_p_value"] = np.nan
    result["bh_family"] = ""
    macro = result.inference_level.ne("market")
    for block, group in result[macro].groupby("block_length"):
        result.loc[group.index, "adjusted_p_value"] = benjamini_hochberg(group.raw_p_value.to_numpy())
        result.loc[group.index, "bh_family"] = f"SAME_SCOPE_MACRO_BLOCK{block}_FOUR_MODELS"
    result["classification"] = np.where(
        result.adjusted_p_value.notna(),
        np.where(
            ((result.ci_lower > 0) | (result.ci_upper < 0)) & (result.adjusted_p_value <= 0.05),
            "SUPPORTED_CURRENT_PIT_DIFFERENCE", "NO_SUPPORTED_DIFFERENCE",
        ),
        "MARKET_LEVEL_DESCRIPTIVE",
    )
    keep = ["scope", "inference_level", "index_id", "model_id", "block_length", "bootstrap_samples",
            "bootstrap_seed", "markets_in_scope", "intervals_in_scope", "oos_dates_in_scope",
            "point_estimate", "ci_lower", "ci_upper", "raw_p_value", "adjusted_p_value", "bh_family",
            "classification"]
    order = {"market": 1}
    result = result.sort_values(
        ["inference_level", "model_id", "block_length"],
        key=lambda s: s.map(lambda v: order.get(v, 0)) if s.name == "inference_level" else s,
    )
    return result[keep].reset_index(drop=True), eligible


# ----------------------------------------------------------------- utils
def _order(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    sort = []
    if "index_id" in out:
        out["_m"] = out.index_id.map({m: i for i, m in enumerate(MARKETS)})
        sort.append("_m")
    if "universe_semantics" in out:
        out["_a"] = out.universe_semantics.map({a: i for i, a in enumerate(ARMS)})
        sort.append("_a")
    if sort:
        out = out.sort_values(sort, kind="stable").drop(columns=sort)
    return out.reset_index(drop=True)


def _check(checks: list[dict], name: str, ok: bool, detail) -> None:
    checks.append({"check": name, "status": "PASS" if ok else "FAIL", "detail": detail})


def _clean(value):
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value) if np.isfinite(value) else None
    return value


def run(evidence_root: Path, output: Path, *, samples: int, bootstrap: bool) -> dict:
    sys.path.insert(0, str(HERE))
    import reviewer_diagnostics as public  # noqa: E402

    P = evidence_root / "research_data" / "point_in_time"
    rejection_path = P / "qa" / "pit_multimarket_observation_rejections_v8.parquet"
    if not rejection_path.is_file():
        raise FileNotFoundError(f"restricted rejection panel not found: {rejection_path}")
    output.mkdir(parents=True, exist_ok=True)
    checks: list[dict] = []

    members = pd.read_parquet(P / "processed" / "pit_membership_daily_all_arms_v8.parquet")
    folds = pd.read_parquet(P / "processed" / "pit_multimarket_fold_calendar_v8.parquet")
    predictions = pd.read_parquet(P / "processed" / "pit_multimarket_predictions_seed_collapsed_v8.parquet")
    predictions["prediction_date"] = as_day(predictions.prediction_date)
    rejections = pd.read_parquet(rejection_path, columns=KEY + ["rejection_reason"])
    raw = pd.read_parquet(
        P / "processed" / "pit_historical_raw_ohlc_recovered_v8.parquet",
        columns=["market", "security_id", "date", "provider"],
    )
    inventory = pd.read_csv(ROOT / "tables" / "supplementary" / "S04_interval_inventory.csv")

    public_result = public.build(output / "public_reference")
    _check(checks, "public_reference_diagnostics", public_result["status"] == "PASS", public_result["status"])
    public_r1 = pd.read_csv(output / "public_reference" / "R1_coverage_by_market_arm.csv")
    public_r3 = pd.read_csv(output / "public_reference" / "R3_oos_forecasts_common_support.csv")

    # R1 -----------------------------------------------------------------
    panel, diagnostics = build_status_panel(members, folds, predictions, rejections, raw)
    for name in ["rejection_keys_with_conflicting_reasons", "forecast_and_rejected_member_days",
                 "forecasts_outside_oos_opportunity_set", "partial_model_forecasts",
                 "unexplained_not_forecast"]:
        _check(checks, name, diagnostics[name] == 0, diagnostics[name])
    by_reason = summarise_by_reason(panel)
    merged = by_reason.merge(public_r1, on=["index_id", "universe_semantics"], validate="one_to_one")
    _check(checks, "valid_forecasts_match_public_r1",
           bool(merged.valid_forecasts.eq(merged.valid_forecasts_per_model).all()), int(len(merged)))
    derivable = merged.forecast_opportunities_y.notna()
    _check(checks, "opportunities_match_public_r1_all_arms",
           bool(merged.loc[derivable, "forecast_opportunities_x"].eq(merged.loc[derivable, "forecast_opportunities_y"]).all()),
           int(derivable.sum()))
    _check(checks, "opportunities_reconcile",
           bool(by_reason.forecast_opportunities.eq(by_reason.valid_forecasts + by_reason.excluded).all()), None)
    by_reason.to_csv(output / "R1_private_oos_coverage_by_reason.csv", index=False)
    summarise_long(panel).to_csv(output / "R1_private_oos_exclusions_attributed.csv", index=False)
    security_detail(panel).to_csv(output / "R1_private_security_detail_PSEI_FBM_KLCI.csv", index=False)
    gap_detail, gap_summary = pit_price_gaps(members, raw, inventory)
    gap_detail.to_csv(output / "R1_private_pit_price_gaps_security_PSEI_FBM_KLCI.csv", index=False)
    gap_summary.to_csv(output / "R1_private_pit_price_gaps_interval_PSEI_FBM_KLCI.csv", index=False)

    # R3 -----------------------------------------------------------------
    cs = common_support(predictions)
    per_market = cs.groupby("index_id").agg(
        pairs_min=("common_support_pairs", "min"), pairs_max=("common_support_pairs", "max"),
        oos_dates=("oos_dates", "first"),
    )
    reference = public_r3[public_r3.index_id.isin(MARKETS)].set_index("index_id")
    _check(checks, "common_support_identical_across_models",
           bool(per_market.pairs_min.eq(per_market.pairs_max).all()), None)
    _check(checks, "common_support_matches_public_r3",
           bool(per_market.pairs_min.eq(reference.loc[per_market.index, "common_support_pairs"]).all()),
           per_market.pairs_min.to_dict())
    _check(checks, "oos_dates_match_public_r3",
           bool(per_market.oos_dates.eq(reference.loc[per_market.index, "oos_dates"]).all()),
           per_market.oos_dates.to_dict())
    cs.to_csv(output / "R3_private_common_support_by_model.csv", index=False)

    # R2 -----------------------------------------------------------------
    if bootstrap:
        sys.path.insert(0, str(evidence_root))
        from scripts.point_in_time import run_pit_v8_1_release as engine  # noqa: E402

        dd = date_level_deltas(predictions)
        replay = engine._bootstrap_family(dd, support="FULL_SUPPORT", metrics=["brier_score"], samples=samples)
        published = pd.read_csv(ROOT / "results" / "pit_multimarket_primary_bootstrap_10000_v8_1_1.csv")
        published = published[published.support_specification.eq("FULL_SUPPORT")
                              & published.inference_level.eq("selected_five_market_macro")
                              & published.block_length.eq(20)]
        replay = replay[replay.inference_level.eq("selected_five_market_macro") & replay.block_length.eq(20)]
        both = replay.merge(published, on="model_id", suffixes=("_replay", "_published"))
        error = max(float((both[f"{c}_replay"] - both[f"{c}_published"]).abs().max())
                    for c in ["point_estimate", "ci_lower", "ci_upper", "raw_p_value"])
        if samples == 10000:
            _check(checks, "published_confirmatory_bootstrap_replayed", error <= TOLERANCE and len(both) == 4, error)
        same_scope, eligible = same_scope_bootstrap(dd, engine, samples)
        same_scope.to_csv(output / "R2_private_same_scope_bootstrap.csv", index=False)
        eligible.to_csv(output / "R2_private_same_scope_intervals.csv", index=False)

    result = {
        "scope": "PRIVATE_OBSERVATION_LEVEL_COVERAGE_ATTRIBUTION",
        "evidence_root": "PRIVATE_EVIDENCE_ROOT",
        "redistribution": "WITHHELD_RIGHTS_UNESTABLISHED",
        "bootstrap_samples": samples if bootstrap else 0,
        "diagnostics": diagnostics,
        "checks": checks,
        "status": "PASS" if all(c["status"] == "PASS" for c in checks) else "FAIL",
    }
    (output / "verification.json").write_text(json.dumps(_clean(result), indent=2, default=str) + "\n",
                                              encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--evidence-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=10000)
    parser.add_argument("--skip-bootstrap", action="store_true")
    args = parser.parse_args(argv)
    result = run(args.evidence_root.resolve(), args.output.resolve(),
                 samples=args.samples, bootstrap=not args.skip_bootstrap)
    for check in result["checks"]:
        print(f"{check['status']}  {check['check']}")
    print(result["status"])
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
