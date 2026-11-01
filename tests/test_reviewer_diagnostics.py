"""Reviewer diagnostics: public aggregate checks and software fixtures.

The fixture frames below are explicitly synthetic software fixtures used to
exercise classification logic. They are not market evidence.
"""
from pathlib import Path
import importlib.util
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_public_diagnostics_pass_and_match_committed(tmp_path):
    public = _load("reviewer_diagnostics")
    result = public.build(tmp_path)
    assert result["status"] == "PASS", [c for c in result["checks"] if c["status"] != "PASS"]
    committed = ROOT / "tables" / "reviewer_diagnostics"
    for name in result["outputs"]:
        if name.endswith(".csv"):
            pd.testing.assert_frame_equal(pd.read_csv(tmp_path / name), pd.read_csv(committed / name))
    assert (tmp_path / "SUMMARY.md").read_text(encoding="utf-8") == (committed / "SUMMARY.md").read_text(
        encoding="utf-8"
    )


def test_public_diagnostic_values(tmp_path):
    public = _load("reviewer_diagnostics")
    public.build(tmp_path)
    r1 = pd.read_csv(tmp_path / "R1_coverage_by_market_arm.csv").set_index(["index_id", "universe_semantics"])
    assert r1.loc[("PSEI", "CURRENT_2026_APPLIED_BACKWARD"), "not_forecast"] == 5589
    assert r1.loc[("FBM_KLCI", "POINT_IN_TIME"), "not_forecast"] == 482
    assert np.isnan(r1.loc[("PSEI", "PIT_CURRENT_INTERSECTION_CONTROL"), "forecast_opportunities"])
    r2 = pd.read_csv(tmp_path / "R2_bootstrap_block_sensitivity.csv")
    full = r2[r2.support_specification.eq("FULL_SUPPORT")]
    assert len(full) == 16 and set(full.block_length) == {10, 20, 40}
    five40 = full[full.block_length.eq(40) & full.macro_scope.eq("FIVE_MARKET_MACRO")]
    assert five40.classification.eq("INSUFFICIENT_INFORMATION").all()
    four40 = full[full.block_length.eq(40) & full.macro_scope.eq("AVAILABLE_MARKET_MACRO")]
    assert four40.markets_used.eq(4).all() and four40.excluded_markets.eq("IDX30").all()
    assert four40.intervals_used.eq(23).all()
    r3 = pd.read_csv(tmp_path / "R3_oos_forecasts_common_support.csv").set_index("index_id")
    assert r3.loc["ALL", "common_support_pairs"] == 85948
    assert r3.loc["ALL", "oos_dates"] == 3299


def _fixture():
    days = pd.bdate_range("2001-01-01", periods=4)
    d1, d2, d3, d4 = days
    base = dict(market="Philippines", index_id="PSEI", interval_id="I1")
    members = pd.DataFrame(
        [dict(base, universe_semantics="POINT_IN_TIME", security_id=s, ticker_on_date=s, membership_date=d)
         for s in ["A", "B"] for d in days]
        + [dict(base, universe_semantics="CURRENT_2026_APPLIED_BACKWARD", security_id=s, ticker_on_date=s,
                membership_date=d) for s in ["A", "C"] for d in days]
        + [dict(base, universe_semantics="PIT_CURRENT_INTERSECTION_CONTROL", security_id="A",
                ticker_on_date="A", membership_date=d) for d in days]
    )
    folds = pd.DataFrame([
        dict(interval_id="I1", fold_id="F1", test_start=d2, test_end=d4, valid_fold=True),
        dict(interval_id="I1", fold_id="F0", test_start=d1, test_end=d1, valid_fold=False),
    ])
    raw = pd.DataFrame(
        [dict(market="Philippines", security_id="A", date=d) for d in days]
        + [dict(market="Philippines", security_id="B", date=d) for d in [d1, d2, d4]]
        + [dict(market="Philippines", security_id="C", date=d4)]
    )
    forecast = (
        [("POINT_IN_TIME", "A", d) for d in [d2, d3, d4]]
        + [("POINT_IN_TIME", "B", d) for d in [d2, d4]]
        + [("CURRENT_2026_APPLIED_BACKWARD", "A", d) for d in [d2, d3, d4]]
        + [("PIT_CURRENT_INTERSECTION_CONTROL", "A", d) for d in [d2, d3, d4]]
    )
    models = ["historical_class_probability", "lightgbm", "logistic_regression", "random_forest"]
    predictions = pd.DataFrame(
        [dict(base, universe_semantics=a, security_id=s, prediction_date=d, model_id=m, fold_id="F1",
              target_date=d + pd.Timedelta(days=1), actual_class=1, prediction_probability=0.5)
         for a, s, d in forecast for m in models]
    )
    rejections = pd.DataFrame([
        dict(index_id="PSEI", universe_semantics="POINT_IN_TIME", security_id="B", prediction_date=d3,
             rejection_reason="MISSING_PREDICTION_RAW_OHLC"),
        dict(index_id="PSEI", universe_semantics="CURRENT_2026_APPLIED_BACKWARD", security_id="C",
             prediction_date=d2, rejection_reason="MISSING_PREDICTION_RAW_OHLC"),
        dict(index_id="PSEI", universe_semantics="CURRENT_2026_APPLIED_BACKWARD", security_id="C",
             prediction_date=d3, rejection_reason="MISSING_PREDICTION_RAW_OHLC"),
        dict(index_id="PSEI", universe_semantics="CURRENT_2026_APPLIED_BACKWARD", security_id="C",
             prediction_date=d4, rejection_reason="MISSING_FEATURES"),
        dict(index_id="PSEI", universe_semantics="POINT_IN_TIME", security_id="A", prediction_date=d1,
             rejection_reason="MISSING_FEATURES"),
    ])
    return members, folds, predictions, rejections, raw


def test_private_status_panel_attribution_fixture():
    private = _load("private_coverage_audit")
    panel, diagnostics = private.build_status_panel(*_fixture())
    assert all(v == 0 for k, v in diagnostics.items() if k != "models")
    summary = private.summarise_by_reason(panel).set_index("universe_semantics")
    assert summary.loc["POINT_IN_TIME", "forecast_opportunities"] == 6
    assert summary.loc["POINT_IN_TIME", "valid_forecasts"] == 5
    assert summary.loc["CURRENT_2026_APPLIED_BACKWARD", "valid_forecasts"] == 3
    assert summary.loc["CURRENT_2026_APPLIED_BACKWARD", "excluded_selection"] == 2
    assert summary.loc["CURRENT_2026_APPLIED_BACKWARD", "excluded_data"] == 1
    assert summary.loc["PIT_CURRENT_INTERSECTION_CONTROL", "forecast_opportunities"] == 3
    excluded = panel[panel.exclusion_reason.ne(private.FORECAST)]
    b = excluded[excluded.universe_semantics.eq("POINT_IN_TIME") & excluded.security_id.eq("B")]
    assert len(b) == 1
    b = b.iloc[0]
    assert b.price_position == "INTERNAL_PRICE_GAP" and b.attribution == "DATA_PRICE_UNAVAILABLE"
    assert b.membership_role == "PIT_ONLY_LATER_REMOVED"
    c = excluded[excluded.universe_semantics.eq("CURRENT_2026_APPLIED_BACKWARD") & excluded.security_id.eq("C")]
    assert len(c) == 3 and len(excluded) == 4
    assert set(c.attribution) == {"SELECTION_BACKWARD_ROSTER_BEFORE_FIRST_PRICE", "DATA_FEATURE_HISTORY"}
    assert set(c.membership_role) == {"CURRENT_ONLY_NOT_PIT_MEMBER_ON_DATE"}
    cs = private.common_support(_fixture()[2])
    assert cs.common_support_pairs.eq(3).all() and len(cs) == 4


def test_private_same_scope_bootstrap_fixture():
    private = _load("private_coverage_audit")
    sys.path.insert(0, str(ROOT))
    from scripts.point_in_time import run_pit_v8_1_release as engine

    rng = np.random.default_rng(0)
    rows = []
    for index_id, interval_id, n in [("PSEI", "P1", 100), ("PSEI", "P2", 50), ("SET50", "S1", 90)]:
        for model in ["logistic_regression", "random_forest"]:
            for d in pd.bdate_range("2001-01-01", periods=n):
                rows.append(dict(interval_id=interval_id, market="m", index_id=index_id, model_id=model,
                                 prediction_date=d, delta_brier_score_current_minus_pit=rng.normal(0, 0.01)))
    result, eligible = private.same_scope_bootstrap(pd.DataFrame(rows), engine, samples=200)
    assert set(eligible.interval_id) == {"P1", "S1"}
    macro = result[result.inference_level.ne("market")]
    assert len(macro) == 6 and set(macro.block_length) == {10, 20, 40}
    assert macro.scope.eq("2_MARKET_MACRO_BLOCK40_ELIGIBLE_INTERVALS").all()
    assert macro.groupby("block_length").bh_family.nunique().eq(1).all()
    points = macro.groupby("model_id").point_estimate.nunique()
    assert points.eq(1).all(), "same scope must give the same point estimate at every block length"
