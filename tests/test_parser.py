"""Pytest unit tests for query parser, structured filters, zone comparison, and scenario diff."""
import pandas as pd
import pytest

from analyze import (Context, answer, apply_structured_json_filters,
                    compare_scenarios, compare_zones, parse_filters, top_n)


@pytest.fixture(scope="module")
def ctx():
    return Context()


def test_top_n_extraction():
    assert top_n("top 15 units by risk") == 15
    assert top_n("show 5 riskiest areas") == 5
    assert top_n("which areas are prone to landslides", default=10) == 10


def test_parse_filters_lithology(ctx):
    mask, notes = parse_filters("clay and shale rock types", ctx.df)
    assert mask.any()
    assert (ctx.df.loc[mask, "lithology"] == "clay_shale").all()
    assert any("clay_shale" in n for n in notes)


def test_parse_filters_numeric_operators(ctx):
    mask, notes = parse_filters("slope > 30 and rainfall > 1100", ctx.df)
    assert mask.any()
    assert (ctx.df.loc[mask, "slope"] > 30).all()
    assert (ctx.df.loc[mask, "rainfall"] > 1100).all()


def test_parse_filters_proximity(ctx):
    mask, notes = parse_filters("near fault", ctx.df)
    assert mask.any()
    assert (ctx.df.loc[mask, "fault_dist"] <= 500).all()


def test_apply_structured_json_filters(ctx):
    filters = {
        "lithology": ["sandstone", "limestone"],
        "slope": [">=", 20.0],
        "fault_dist": ["<=", 1000.0],
    }
    mask, notes = apply_structured_json_filters(filters, ctx.df)
    assert mask.any()
    subset = ctx.df[mask]
    assert subset["lithology"].isin(["sandstone", "limestone"]).all()
    assert (subset["slope"] >= 20.0).all()
    assert (subset["fault_dist"] <= 1000.0).all()


def test_compare_zones(ctx):
    summary, comp_df = compare_zones(ctx, "A1", "C2")
    assert not comp_df.empty
    assert len(comp_df) == 2
    assert "A1" in comp_df["Zone"].values
    assert "C2" in comp_df["Zone"].values
    assert "Mean Risk" in comp_df.columns
    assert "ZONE COMPARISON: A1 vs C2" in summary


def test_compare_scenarios(ctx):
    summary, diff_df = compare_scenarios(ctx, 10.0, 30.0)
    assert not diff_df.empty
    assert "delta_prob" in diff_df.columns
    assert "SCENARIO COMPARISON: Scenario A (+10%) vs Scenario B (+30%)" in summary
    assert (diff_df["delta_prob"] >= 0).mean() > 0.8  # higher rain increases risk for vast majority


def test_answer_explain_unit(ctx):
    first_uid = ctx.df.iloc[0]["unit_id"]
    txt, tab, mask, title = answer(f"explain {first_uid}", ctx)
    assert tab is not None
    assert mask.sum() == 1
    assert "What drives this score" in txt
