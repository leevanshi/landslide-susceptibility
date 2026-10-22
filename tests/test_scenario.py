"""Pytest unit tests for rainfall scenario simulation and early warning shock."""
import pandas as pd
import pytest

from analyze import Context, early_warning_assessment, rainfall_scenario


@pytest.fixture(scope="module")
def ctx():
    return Context()


def test_rainfall_scenario_monotonicity(ctx):
    mask = pd.Series(True, index=ctx.df.index)
    res_plus30 = rainfall_scenario(ctx, mask, pct=30.0)
    res_minus30 = rainfall_scenario(ctx, mask, pct=-30.0)

    # Increased rainfall must on average increase susceptibility
    assert res_plus30["scenario_prob"].mean() > ctx.df["probability"].mean()
    # Decreased rainfall must on average decrease susceptibility
    assert res_minus30["scenario_prob"].mean() < ctx.df["probability"].mean()
    assert (res_plus30["change"] >= 0).mean() > 0.85


def test_early_warning_assessment(ctx):
    summary, alerts, counts = early_warning_assessment(ctx, rainfall_24h_mm=50.0, use_open_meteo=False)
    assert not alerts.empty
    assert "alert_level" in alerts.columns
    assert "live_prob" in alerts.columns
    assert set(counts.keys()) == {"Red (Warning)", "Orange (Watch)", "Yellow (Advisory)", "Green (Normal)"}
    assert sum(counts.values()) == len(ctx.df)
    assert "EARLY WARNING BULLETIN" in summary
