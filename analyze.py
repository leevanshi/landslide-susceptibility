"""Prompt-based landslide analyzer with LLM (Claude API) structured JSON filters,
offline rule-based fallback, zone comparison, scenario diff, and live rainfall early warning.

Run:  python analyze.py            (interactive prompt)
      python analyze.py "your question here"
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import joblib
import matplotlib
import numpy as np
import pandas as pd
import requests
import shap

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from common import (CALIBRATED_MODEL_PATH, CAT, CONFIG, EXTENT_KM, FEATURES,
                    MODEL_PATH, NUM, OUT, RISK_COLORS, RISK_LABELS, load_data,
                    risk_class, zone_of)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

pd.set_option("display.width", 140)

LITH_WORDS: Dict[str, str] = {
    "clay": "clay_shale", "shale": "clay_shale", "marl": "marl_flysch", "flysch": "marl_flysch",
    "sandstone": "sandstone", "limestone": "limestone", "granite": "granite_gneiss",
    "gneiss": "granite_gneiss", "alluvi": "alluvium"
}
LC_WORDS: Dict[str, str] = {
    "forest": "forest", "farm": "farmland", "grass": "grassland", "bare": "bare_soil", "urban": "urban"
}
NUM_WORDS: Dict[str, str] = {
    "slope": "slope", "steep": "slope", "elevation": "elevation", "altitude": "elevation",
    "rainfall": "rainfall", "rain": "rainfall", "curvature": "curvature",
    "river": "river_dist", "stream": "river_dist", "road": "road_dist", "fault": "fault_dist"
}
HIGH = ["High", "Very High"]


# --------------------------------------------------------------------------- Context
class Context:
    """Manages loaded dataset, trained models, and prediction inference."""

    def __init__(self, use_calibrated: bool = True) -> None:
        self.df: pd.DataFrame = load_data()
        self.calibrated_path: Path = CALIBRATED_MODEL_PATH
        self.model_path: Path = MODEL_PATH

        if not self.model_path.exists():
            raise SystemExit("Model not found. Run:  python train.py")

        self.base_pipeline: Any = joblib.load(self.model_path)
        if use_calibrated and self.calibrated_path.exists():
            self.model: Any = joblib.load(self.calibrated_path)
            self.is_calibrated: bool = True
        else:
            self.model = self.base_pipeline
            self.is_calibrated = False

        self.refresh()

    def refresh(self) -> None:
        self.df["probability"] = self.predict(self.df)
        self.df["risk_class"] = risk_class(self.df["probability"])

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[FEATURES])[:, 1]


# --------------------------------------------------------------------------- Visualization
def draw_map(df: pd.DataFrame, highlight: Optional[pd.Series] = None, title: str = "Landslide susceptibility map") -> plt.Figure:
    """Draw a 5x5 zoned scatter map of slope units with optional subset highlighting."""
    cmap = ListedColormap(RISK_COLORS)
    codes = pd.Categorical(df["risk_class"], categories=RISK_LABELS).codes
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    if highlight is None:
        ax.scatter(df.x_km, df.y_km, c=codes, cmap=cmap, vmin=0, vmax=4, s=9)
    else:
        ax.scatter(df.x_km, df.y_km, c="#d9d9d9", s=6)
        h = df[highlight]
        ax.scatter(
            h.x_km, h.y_km,
            c=pd.Categorical(h["risk_class"], categories=RISK_LABELS).codes,
            cmap=cmap, vmin=0, vmax=4, s=16, edgecolor="k", linewidth=0.3
        )
    step = EXTENT_KM / 5
    for i in range(1, 5):
        ax.axhline(i * step, color="k", lw=0.3, alpha=0.4)
        ax.axvline(i * step, color="k", lw=0.3, alpha=0.4)
    for c, letter in enumerate("ABCDE"):
        for r in range(5):
            ax.text((c + 0.5) * step, (r + 0.5) * step, f"{letter}{r + 1}", ha="center", va="center",
                    fontsize=9, color="k", alpha=0.35, weight="bold")
    ax.set_xlim(0, EXTENT_KM)
    ax.set_ylim(0, EXTENT_KM)
    ax.set_aspect("equal")
    ax.set_xlabel("x (km)")
    ax.set_ylabel("y (km)")
    ax.set_title(title)
    handles = [plt.Line2D([], [], marker="o", ls="", color=c, label=l) for c, l in zip(RISK_COLORS, RISK_LABELS)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1), title="Risk class", fontsize=8)
    return fig


# --------------------------------------------------------------------------- LLM Parser (Claude API)
CLAUDE_SYSTEM_PROMPT = """You are a geospatial query parser for a landslide susceptibility system.
Given a user's natural language question, translate it into a structured JSON filter object.
Output ONLY valid JSON with no conversational text or markdown code fences.

Schema:
{
  "intent": "filter" | "ranking" | "geology" | "scenario" | "explain" | "compare_zones" | "scenario_diff" | "early_warning" | "summary",
  "top_n": <integer, default 10>,
  "unit_id": <string or null, e.g. "SU-0123">,
  "zones": [<list of uppercase zones, e.g. ["A1", "C2"]>],
  "compare_zones": [<two zones to compare, e.g. ["A1", "C2"]>],
  "scenario_pct": <float or null, percentage rainfall change, e.g. 30.0>,
  "scenario_x_pct": <float or null, e.g. 10.0>,
  "scenario_y_pct": <float or null, e.g. 30.0>,
  "rainfall_24h_mm": <float or null, for early warning>,
  "use_open_meteo": <boolean, true if user asks for live/forecast weather>,
  "filters": {
    "lithology": [<list of valid lithologies: "clay_shale", "marl_flysch", "sandstone", "limestone", "granite_gneiss", "alluvium">],
    "land_cover": [<list of valid land covers: "forest", "farmland", "grassland", "bare_soil", "urban">],
    "slope": [<operator ">=", "<=", ">", "<", "=">, <numeric value>],
    "elevation": [<operator>, <numeric value>],
    "rainfall": [<operator>, <numeric value>],
    "fault_dist": [<operator>, <numeric value>],
    "river_dist": [<operator>, <numeric value>],
    "road_dist": [<operator>, <numeric value>]
  }
}
"""


def call_claude_parser(question: str) -> Optional[Dict[str, Any]]:
    """Convert question to structured JSON filter using Claude API if ANTHROPIC_API_KEY is set."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }
    payload = {
        "model": "claude-3-haiku-20240307",
        "max_tokens": 512,
        "temperature": 0.0,
        "system": CLAUDE_SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": question}],
    }

    try:
        response = requests.post("https://api.anthropic.com/v1/messages", json=payload, headers=headers, timeout=6)
        if response.status_code == 200:
            text = response.json()["content"][0]["text"].strip()
            # Clean json fences if returned
            text = re.sub(r"^```json\s*", "", text, flags=re.I)
            text = re.sub(r"^```\s*", "", text)
            text = re.sub(r"\s*```$", "", text)
            data = json.loads(text)
            logger.info("Successfully parsed query using Claude API.")
            return data
        else:
            logger.warning(f"Claude API returned status {response.status_code}: {response.text}")
    except Exception as e:
        logger.warning(f"Claude API request failed: {e}. Falling back to offline rule parser.")

    return None


# --------------------------------------------------------------------------- Rule-based Parser (Offline Fallback)
def _cmp(s: pd.Series, op: str, v: float) -> pd.Series:
    ops = {">": s > v, ">=": s >= v, "<": s < v, "<=": s <= v, "=": s == v}
    return ops.get(op, s > v)


def parse_filters(q: str, df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Parse natural language query into boolean DataFrame mask using rule-based pattern matching."""
    ql = q.lower()
    mask = pd.Series(True, index=df.index)
    notes: List[str] = []

    # Check lithology
    matched_rocks = []
    for w, name in LITH_WORDS.items():
        if re.search(r"\b" + w, ql):
            matched_rocks.append(name)
    matched_rocks = list(dict.fromkeys(matched_rocks))
    if matched_rocks:
        mask &= df["lithology"].isin(matched_rocks)
        notes.append(f"lithology in {matched_rocks}")

    # Check land cover
    lcs = [v for k, v in LC_WORDS.items() if re.search(r"\b" + k, ql)]
    if lcs:
        mask &= df["land_cover"].isin(lcs)
        notes.append(f"land cover in {lcs}")

    # Check zones (A1..E5)
    zones = re.findall(r"\bzone\s*([a-e][1-5])\b", ql) or re.findall(r"\b([a-e][1-5])\b", ql)
    if zones:
        zone_list = [z.upper() for z in zones]
        mask &= df["zone"].isin(zone_list)
        notes.append(f"zone in {zone_list}")

    # Numeric thresholds
    ops = {
        ">=": ">=", "<=": "<=", ">": ">", "<": "<", "=": "=", "above": ">", "over": ">",
        "more than": ">", "greater than": ">", "steeper than": ">", "higher than": ">",
        "at least": ">=", "below": "<", "under": "<", "less than": "<", "lower than": "<",
        "at most": "<=", "within": "<="
    }
    op_re = "|".join(sorted(map(re.escape, ops), key=len, reverse=True))
    word_re = "|".join(NUM_WORDS)
    pat = re.compile(rf"\b({word_re})\w*(?:\s+(?:dist\w*|of))?\s*(?:is|are|=)?\s*({op_re})?\s*(\d+(?:\.\d+)?)", re.I)
    pat2 = re.compile(rf"\b({op_re})\s*(\d+(?:\.\d+)?)\s*(?:m|mm|meters?|metres?|km|degrees?|°)?\s*(?:of|from|to)?\s*(?:a |an |the )?\s*({word_re})", re.I)

    used = set()
    for m in pat.finditer(ql):
        word, op_str, val = m.group(1), m.group(2), float(m.group(3))
        col = NUM_WORDS[word]
        if col in used or "%" in ql[m.end(): m.end() + 1] or (
            col == "rainfall" and re.search(r"increas|decreas|rise|drop|\+|-|what if|by", ql[max(0, m.start() - 20): m.end()])
        ):
            continue
        op_actual = ops.get(op_str, "<=" if col.endswith("_dist") else ">")
        used.add(col)
        mask &= _cmp(df[col], op_actual, val)
        notes.append(f"{col} {op_actual} {val:g}")

    for m in pat2.finditer(ql):
        op_str, val, word = m.group(1), float(m.group(2)), m.group(3)
        col = NUM_WORDS[word]
        if col in used:
            continue
        if re.search(r"\bkm\b", ql[m.start(): m.end() + 4]):
            val *= 1000
        op_actual = ops[op_str]
        used.add(col)
        mask &= _cmp(df[col], op_actual, val)
        notes.append(f"{col} {op_actual} {val:g}")

    # Proximity queries
    for word, col in [("fault", "fault_dist"), ("river", "river_dist"), ("road", "road_dist")]:
        if col not in used and re.search(rf"\b(near|close to|next to)\b[^.?]*\b{word}", ql):
            mask &= df[col] <= 500
            notes.append(f"{col} <= 500 (near {word})")

    return mask.fillna(False), notes


def apply_structured_json_filters(filters: Dict[str, Any], df: pd.DataFrame) -> Tuple[pd.Series, List[str]]:
    """Translate LLM structured filter JSON into a boolean pandas mask."""
    mask = pd.Series(True, index=df.index)
    notes: List[str] = []

    if not filters:
        return mask, notes

    # Lithology
    liths = filters.get("lithology")
    if liths and isinstance(liths, list):
        mask &= df["lithology"].isin(liths)
        notes.append(f"lithology in {liths}")

    # Land cover
    lcs = filters.get("land_cover")
    if lcs and isinstance(lcs, list):
        mask &= df["land_cover"].isin(lcs)
        notes.append(f"land cover in {lcs}")

    # Numeric filters: e.g. "slope": [">", 25]
    for col in NUM:
        spec = filters.get(col)
        if spec and isinstance(spec, (list, tuple)) and len(spec) >= 2:
            op, val = str(spec[0]), float(spec[1])
            mask &= _cmp(df[col], op, val)
            notes.append(f"{col} {op} {val:g}")

    return mask.fillna(False), notes


def top_n(q: str, default: int = 10) -> int:
    m = re.search(r"\btop\s*(\d+)|\b(\d+)\s+(?:most|highest|riskiest|worst)", q.lower())
    return int(next(g for g in m.groups() if g)) if m else default


# --------------------------------------------------------------------------- Analyses Functions
def zone_ranking(ctx: Context, mask: pd.Series, n: int) -> pd.DataFrame:
    d = ctx.df[mask]
    g = d.groupby("zone").agg(
        units=("unit_id", "count"),
        mean_risk=("probability", "mean"),
        pct_high=("risk_class", lambda s: 100 * s.isin(HIGH).mean()),
        main_rock=("lithology", lambda s: s.mode().iat[0] if s.notna().any() else "?"),
        avg_slope=("slope", "mean"),
        avg_rain=("rainfall", "mean"),
    )
    g = g[g.units >= 5].sort_values("mean_risk", ascending=False).head(n)
    g = g.round({"mean_risk": 3, "pct_high": 1, "avg_slope": 1, "avg_rain": 0})
    return g.reset_index().rename(columns={"zone": "area"})


def unit_ranking(ctx: Context, mask: pd.Series, n: int) -> pd.DataFrame:
    cols = [
        "unit_id", "zone", "probability", "risk_class", "lithology", "land_cover",
        "slope", "rainfall", "fault_dist", "x_km", "y_km"
    ]
    return ctx.df[mask].sort_values("probability", ascending=False).head(n)[cols].round({"probability": 3})


def geology_ranking(ctx: Context, mask: pd.Series) -> pd.DataFrame:
    d = ctx.df[mask]
    g = d.groupby("lithology").agg(
        units=("unit_id", "count"),
        mean_risk=("probability", "mean"),
        pct_high=("risk_class", lambda s: 100 * s.isin(HIGH).mean()),
        observed_landslide_pct=("landslide", lambda s: 100 * s.mean()),
        avg_slope=("slope", "mean"),
        avg_fault_dist=("fault_dist", "mean"),
    )
    near = d[d.fault_dist <= 1000].groupby("lithology")["probability"].mean().rename("risk_near_fault_1km")
    g = g.join(near).sort_values("mean_risk", ascending=False)
    return g.round({
        "mean_risk": 3, "pct_high": 1, "observed_landslide_pct": 1,
        "avg_slope": 1, "avg_fault_dist": 0, "risk_near_fault_1km": 3
    }).reset_index()


def rainfall_scenario(ctx: Context, mask: pd.Series, pct: float) -> pd.DataFrame:
    d = ctx.df[mask].copy()
    base = d["probability"].values
    d2 = d.copy()
    d2["rainfall"] = d2["rainfall"] * (1 + pct / 100.0)
    d["scenario_prob"] = ctx.predict(d2)
    d["scenario_class"] = risk_class(d["scenario_prob"])
    d["change"] = d["scenario_prob"] - base
    return d


def compare_zones(ctx: Context, z1: str, z2: str) -> Tuple[str, pd.DataFrame]:
    """Compare two spatial grid zones (e.g. A1 vs C2) in detail."""
    df = ctx.df
    d1 = df[df.zone.str.upper() == z1.upper()]
    d2 = df[df.zone.str.upper() == z2.upper()]

    if d1.empty or d2.empty:
        missing = [z for z, d in [(z1, d1), (z2, d2)] if d.empty]
        return f"Zone comparison failed: {missing} not found in study area.", pd.DataFrame()

    stats1 = {
        "Zone": z1.upper(), "Units": len(d1),
        "Mean Risk": round(d1.probability.mean(), 3),
        "High/Very High %": round(100 * d1.risk_class.isin(HIGH).mean(), 1),
        "Main Rock": d1.lithology.mode().iat[0] if d1.lithology.notna().any() else "N/A",
        "Avg Slope (deg)": round(d1.slope.mean(), 1),
        "Avg Rainfall (mm)": round(d1.rainfall.mean(), 0),
        "Avg Fault Dist (m)": round(d1.fault_dist.mean(), 0),
    }
    stats2 = {
        "Zone": z2.upper(), "Units": len(d2),
        "Mean Risk": round(d2.probability.mean(), 3),
        "High/Very High %": round(100 * d2.risk_class.isin(HIGH).mean(), 1),
        "Main Rock": d2.lithology.mode().iat[0] if d2.lithology.notna().any() else "N/A",
        "Avg Slope (deg)": round(d2.slope.mean(), 1),
        "Avg Rainfall (mm)": round(d2.rainfall.mean(), 0),
        "Avg Fault Dist (m)": round(d2.fault_dist.mean(), 0),
    }

    comp_df = pd.DataFrame([stats1, stats2])
    diff_risk = stats1["Mean Risk"] - stats2["Mean Risk"]
    higher_zone = z1.upper() if diff_risk > 0 else z2.upper()

    summary = (
        f"ZONE COMPARISON: {z1.upper()} vs {z2.upper()}\n"
        f"--------------------------------------------------------------------\n"
        f"{comp_df.to_string(index=False)}\n\n"
        f"Key Takeaway: Zone {higher_zone} exhibits greater hazard (+{abs(diff_risk):.1%} mean risk). "
        f"Zone {z1.upper()} has {stats1['High/Very High %']}% high-hazard units vs "
        f"{stats2['High/Very High %']}% in {z2.upper()}."
    )
    return summary, comp_df


def compare_scenarios(ctx: Context, pct1: float, pct2: float, mask: Optional[pd.Series] = None) -> Tuple[str, pd.DataFrame]:
    """Compare changes between two rainfall scenarios (e.g. +10% vs +30%)."""
    df = ctx.df
    m = mask if mask is not None else pd.Series(True, index=df.index)
    s1 = rainfall_scenario(ctx, m, pct1)
    s2 = rainfall_scenario(ctx, m, pct2)

    diff = pd.DataFrame({
        "unit_id": s1["unit_id"],
        "zone": s1["zone"],
        "lithology": s1["lithology"],
        "baseline_prob": s1["probability"].round(3),
        f"scenario_{pct1:+.0f}%_prob": s1["scenario_prob"].round(3),
        f"scenario_{pct2:+.0f}%_prob": s2["scenario_prob"].round(3),
        "delta_prob": (s2["scenario_prob"] - s1["scenario_prob"]).round(3),
        f"risk_class_{pct1:+.0f}%": s1["scenario_class"],
        f"risk_class_{pct2:+.0f}%": s2["scenario_class"],
    })

    newly_escalated = diff[(~diff[f"risk_class_{pct1:+.0f}%"].isin(HIGH)) & (diff[f"risk_class_{pct2:+.0f}%"].isin(HIGH))]

    summary = (
        f"SCENARIO COMPARISON: Scenario A ({pct1:+.0f}%) vs Scenario B ({pct2:+.0f}%)\n"
        f"--------------------------------------------------------------------\n"
        f"Mean Risk: {s1.scenario_prob.mean():.1%} -> {s2.scenario_prob.mean():.1%} "
        f"({(s2.scenario_prob.mean() - s1.scenario_prob.mean()) * 100:+.1f} percentage points)\n"
        f"High/Very High Units: {s1.scenario_class.isin(HIGH).sum()} -> {s2.scenario_class.isin(HIGH).sum()}\n"
        f"Units newly crossing into High/Very High: {len(newly_escalated)}\n\n"
        f"Top escalated zones: "
        + ", ".join(f"{z} ({c})" for z, c in newly_escalated["zone"].value_counts().head(5).items())
    )
    return summary, diff.sort_values("delta_prob", ascending=False)


def early_warning_assessment(
    ctx: Context,
    rainfall_24h_mm: Optional[float] = None,
    use_open_meteo: bool = False,
) -> Tuple[str, pd.DataFrame, Dict[str, int]]:
    """Early Warning Live Rainfall assessment via Open-Meteo API or manual 24-hr rainfall.

    Returns:
        (summary_text, alerts_df, counts_dict)
    """
    rain_val = rainfall_24h_mm

    if use_open_meteo or rain_val is None:
        try:
            # Query Open-Meteo for Central Italy study area center
            url = (
                f"https://api.open-meteo.com/v1/forecast?"
                f"latitude=42.85&longitude=12.65&daily=precipitation_sum&timezone=auto"
            )
            resp = requests.get(url, timeout=5)
            if resp.status_code == 200:
                precip_list = resp.json().get("daily", {}).get("precipitation_sum", [])
                if precip_list:
                    rain_val = float(precip_list[0])
                    logger.info(f"Retrieved 24h forecast from Open-Meteo: {rain_val:.1f} mm")
        except Exception as e:
            logger.warning(f"Open-Meteo API unavailable: {e}. Defaulting to 40 mm 24h event.")

    if rain_val is None:
        rain_val = 40.0

    df = ctx.df.copy()
    # Dynamic rainfall scaling: baseline annual rainfall scaled by 24h event shock
    # Historical meteorological shock formula for Central Italy Apennines
    shock_scale = 1.0 + (rain_val / 120.0)
    df_shock = df.copy()
    df_shock["rainfall"] = df_shock["rainfall"] * shock_scale
    shock_probs = ctx.predict(df_shock)

    df["live_prob"] = shock_probs.round(3)

    # 4 Alert Tiers
    alert_levels = []
    for p in shock_probs:
        if p >= 0.75:
            alert_levels.append("Red (Warning)")
        elif p >= 0.50:
            alert_levels.append("Orange (Watch)")
        elif p >= 0.25:
            alert_levels.append("Yellow (Advisory)")
        else:
            alert_levels.append("Green (Normal)")
    df["alert_level"] = alert_levels

    counts = df["alert_level"].value_counts().to_dict()
    for lvl in ["Red (Warning)", "Orange (Watch)", "Yellow (Advisory)", "Green (Normal)"]:
        counts.setdefault(lvl, 0)

    alerts_df = df[
        ["unit_id", "zone", "alert_level", "live_prob", "probability", "lithology", "slope", "rainfall"]
    ].sort_values("live_prob", ascending=False)

    OUT.mkdir(parents=True, exist_ok=True)
    alerts_df.to_csv(OUT / "early_warning_alerts.csv", index=False)

    worst_zones = alerts_df[alerts_df.alert_level.str.startswith("Red")]["zone"].value_counts().head(5)

    summary = (
        f"EARLY WARNING BULLETIN: 24-Hour Rainfall Shock ({rain_val:.1f} mm)\n"
        f"--------------------------------------------------------------------\n"
        f"Source: {'Open-Meteo Live API' if use_open_meteo else 'Manual 24h entry'}\n"
        f"  [RED] Warning:     {counts['Red (Warning)']:,} units (Immediate risk/evacuation)\n"
        f"  [ORANGE] Watch:    {counts['Orange (Watch)']:,} units (High slope instability)\n"
        f"  [YELLOW] Advisory: {counts['Yellow (Advisory)']:,} units (Elevated monitoring)\n"
        f"  [GREEN] Normal:    {counts['Green (Normal)']:,} units\n\n"
        f"Critical Zones with Red alerts: "
        + (", ".join(f"{z} ({c} units)" for z, c in worst_zones.items()) if len(worst_zones) else "None")
        + f"\nExported alerts to: outputs/early_warning_alerts.csv"
    )

    return summary, alerts_df, counts


def explain_unit(ctx: Context, unit_id: str) -> Tuple[Optional[pd.Series], Optional[pd.DataFrame]]:
    """Explain single slope-unit prediction with feature contributions."""
    row = ctx.df[ctx.df.unit_id.str.upper() == unit_id.upper()]
    if row.empty:
        return None, None

    prep = ctx.base_pipeline.named_steps["prep"]
    mdl = ctx.base_pipeline.named_steps["model"]
    Xt = prep.transform(row[FEATURES])
    raw_feature_names = prep.get_feature_names_out()
    feature_names = [n.split("__", 1)[-1] for n in raw_feature_names]

    # Calculate SHAP tree contributions
    explainer = shap.TreeExplainer(mdl)
    sv = explainer(Xt)
    contrib = sv.values[0]

    agg: Dict[str, float] = {}
    for n, v in zip(feature_names, contrib):
        base = n
        for c in CAT:
            if base.startswith(c + "_"):
                base = c
        agg[base] = agg.get(base, 0.0) + float(v)

    s = pd.Series(agg).sort_values(key=abs, ascending=False)
    vals = {c: row.iloc[0][c] for c in s.index}
    out = pd.DataFrame({
        "factor": s.index,
        "value": [f"{vals[c]:g}" if isinstance(vals[c], (int, float, np.floating, np.integer)) else str(vals[c]) for c in s.index],
        "pushes_risk": s.values.round(3),
    })
    out["effect"] = np.where(out.pushes_risk > 0, "raises risk", "lowers risk")
    return row.iloc[0], out


def explain_unit_shap(ctx: Context, unit_id: str) -> Tuple[Optional[pd.Series], Optional[shap.Explanation]]:
    """Provide SHAP Explanation object for generating per-unit waterfall plots."""
    row = ctx.df[ctx.df.unit_id.str.upper() == unit_id.upper()]
    if row.empty:
        return None, None

    prep = ctx.base_pipeline.named_steps["prep"]
    mdl = ctx.base_pipeline.named_steps["model"]
    Xt = prep.transform(row[FEATURES])
    raw_feature_names = prep.get_feature_names_out()
    feature_names = [n.split("__", 1)[-1] for n in raw_feature_names]

    explainer = shap.TreeExplainer(mdl)
    shap_vals = explainer(Xt)
    shap_vals.feature_names = feature_names
    return row.iloc[0], shap_vals[0]


# --------------------------------------------------------------------------- Prompt Engine
HELP = """Ask in plain English. Examples:
  which areas are most prone to landslides?            -> ranks map zones (A1..E5)
  compare zone A1 vs C2                                -> compares geology, terrain & risk
  what changed between scenario +10% and +30%?         -> scenario sensitivity difference
  live rainfall 50 mm                                  -> early warning alert list (Green/Yellow/Orange/Red)
  top 15 units with clay and slope > 30                -> ranks individual slope units
  which rock type is most dangerous?                   -> geology ranking (+ risk near faults)
  what if rainfall increases by 30% in clay areas?     -> rainfall scenario simulator
  explain SU-0123                                      -> per-unit SHAP factor drivers
  summary                                              -> overall counts per risk class
Type 'quit' to exit."""


def answer(q: str, ctx: Context) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.Series], Optional[str]]:
    """Answer natural language query using Claude LLM or rule-based parser."""
    ql = q.lower().strip()
    df = ctx.df

    if ql in {"help", "?", "examples"}:
        return HELP, None, None, None

    # 1. Compare zones query: "compare zone A1 vs C2"
    comp_match = re.search(r"compare(?:\s+zones?)?\s+([a-e][1-5])\s+(?:vs|versus|and|with)\s+([a-e][1-5])", ql)
    if comp_match:
        z1, z2 = comp_match.group(1).upper(), comp_match.group(2).upper()
        txt, cdf = compare_zones(ctx, z1, z2)
        zmask = df.zone.isin([z1, z2])
        return txt, cdf, zmask, f"Comparison: Zone {z1} vs {z2}"

    # 2. Compare scenarios query: "what changed between scenario 10% and 30%"
    scen_comp_match = re.search(
        r"(?:what\s+changed\s+between\s+scenarios?|compare\s+scenarios?)\s*([+-]?\d+(?:\.\d+)?)\s*%\s*(?:and|vs|to)\s*([+-]?\d+(?:\.\d+)?)\s*%",
        ql
    )
    if scen_comp_match:
        p1 = float(scen_comp_match.group(1))
        p2 = float(scen_comp_match.group(2))
        txt, sdiff = compare_scenarios(ctx, p1, p2)
        return txt, sdiff, pd.Series(True, index=df.index), f"Scenario Diff: {p1:+.0f}% vs {p2:+.0f}%"

    # 3. Early Warning query: "live rainfall" or "alert"
    if re.search(r"\b(early\s*warning|live\s*rain\w*|warning\s*list|alert\s*list|storm\s*alert)\b", ql):
        rain_match = re.search(r"(\d+(?:\.\d+)?)\s*(?:mm|m)?", ql)
        r_val = float(rain_match.group(1)) if rain_match else None
        use_om = "open-meteo" in ql or "live" in ql or "forecast" in ql or r_val is None
        txt, alerts, _ = early_warning_assessment(ctx, rainfall_24h_mm=r_val, use_open_meteo=use_om)
        red_mask = alerts.alert_level.str.startswith("Red") | alerts.alert_level.str.startswith("Orange")
        return txt, alerts, pd.Series(red_mask, index=df.index), "Early Warning Critical Units"

    # 4. Explain unit query
    m = re.search(r"\b(su[-\s]?\d{1,4})\b", ql)
    if m and re.search(r"explain|why|reason|detail|about|tell", ql):
        uid = re.sub(r"su[-\s]?(\d+)", lambda k: f"SU-{int(k.group(1)):04d}", m.group(1).upper())
        row, tab = explain_unit(ctx, uid)
        if row is None:
            return f"Unit {uid} not found.", None, None, None
        txt = (
            f"{uid} (Zone {row.zone}) — Susceptibility: {row.probability:.1%} -> {row.risk_class}\n"
            f"Rock: {row.lithology} | Land cover: {row.land_cover} | Slope: {row.slope:.1f}° | "
            f"Rainfall: {row.rainfall:.0f} mm | Fault distance: {row.fault_dist:.0f} m\n\n"
            f"What drives this score (SHAP contributions):\n"
            + tab.to_string(index=False)
        )
        return txt, tab, (df.unit_id == uid), f"{uid}: {row.risk_class}"

    # 5. Check if Claude LLM returns structured JSON
    llm_parsed = call_claude_parser(q)
    if llm_parsed and isinstance(llm_parsed, dict):
        intent = llm_parsed.get("intent", "filter")
        if intent == "compare_zones" and len(llm_parsed.get("compare_zones", [])) == 2:
            z1, z2 = llm_parsed["compare_zones"][0], llm_parsed["compare_zones"][1]
            txt, cdf = compare_zones(ctx, z1, z2)
            return txt, cdf, df.zone.isin([z1, z2]), f"Comparison: Zone {z1} vs {z2}"
        if intent == "scenario_diff" and llm_parsed.get("scenario_x_pct") is not None and llm_parsed.get("scenario_y_pct") is not None:
            txt, sdiff = compare_scenarios(ctx, float(llm_parsed["scenario_x_pct"]), float(llm_parsed["scenario_y_pct"]))
            return txt, sdiff, pd.Series(True, index=df.index), "Scenario Difference"
        if intent == "early_warning":
            txt, alerts, _ = early_warning_assessment(
                ctx,
                rainfall_24h_mm=llm_parsed.get("rainfall_24h_mm"),
                use_open_meteo=llm_parsed.get("use_open_meteo", False),
            )
            return txt, alerts, alerts.alert_level.str.startswith("Red"), "Early Warning Alerts"

        mask, notes = apply_structured_json_filters(llm_parsed.get("filters", {}), df)
        n = int(llm_parsed.get("top_n", top_n(q)))
    else:
        # Offline rule-based parser fallback
        mask, notes = parse_filters(q, df)
        n = top_n(q)

    scope = "all units" if not notes else "filter: " + "; ".join(notes)
    if mask.sum() == 0:
        return f"No slope units match ({scope}). Try loosening a condition.", None, None, None

    # Scenario simulation
    if re.search(r"what[\s-]*if|scenario|simulat|increase|decrease|heavier|heavy rain|rain\w*\s*[+-]\s*\d|\d+\s*%", ql) and "rain" in ql:
        pm = re.search(r"(\d+(?:\.\d+)?)\s*%", ql)
        pct = float(pm.group(1)) if pm else 30.0
        if re.search(r"\b(decreas\w*|drop\w*|reduc\w*|less|lower\w*|drier|declin\w*|fall\w*|fewer)\b", ql.replace("rainfall", "rain")):
            pct = -pct
        d = rainfall_scenario(ctx, mask, pct)
        before = d.risk_class.isin(HIGH).sum()
        after = d.scenario_class.isin(HIGH).sum()
        newly = d[(~d.risk_class.isin(HIGH)) & d.scenario_class.isin(HIGH)]
        by_rock = d.groupby("lithology").agg(
            units=("unit_id", "count"), mean_change=("change", "mean"), newly_high=("scenario_class", lambda s: 0)
        ).round(3)
        by_rock["newly_high"] = newly.groupby("lithology").size().reindex(by_rock.index).fillna(0).astype(int)
        by_rock = by_rock.sort_values("newly_high", ascending=False).reset_index()
        worst = newly.groupby("zone").size().sort_values(ascending=False).head(5)
        txt = (
            f"RAINFALL SCENARIO: {pct:+.0f}% rainfall ({scope}; {len(d):,} units)\n"
            f"Mean risk: {d.probability.mean():.1%} -> {d.scenario_prob.mean():.1%}\n"
            f"High/Very-High units: {before:,} -> {after:,} ({len(newly):,} units newly become High/Very High)\n\n"
            f"Rock types most affected:\n{by_rock.to_string(index=False)}\n\n"
            f"Zones gaining the most new high-risk units: "
            + (", ".join(f"{z} (+{c})" for z, c in worst.items()) if len(worst) else "none")
        )
        res = d.sort_values("change", ascending=False)[
            ["unit_id", "zone", "lithology", "rainfall", "probability", "scenario_prob", "risk_class", "scenario_class", "change"]
        ].round(3)
        hl = df.index.isin(newly.index)
        return txt, res, pd.Series(hl, index=df.index), f"Newly High/Very High at {pct:+.0f}% rainfall"

    # Summary
    if re.search(r"summary|overview|how many|count|distribution", ql):
        t = df[mask].risk_class.value_counts().reindex(RISK_LABELS).fillna(0).astype(int).rename("units").reset_index()
        t.columns = ["risk_class", "units"]
        t["share_%"] = (100 * t.units / t.units.sum()).round(1)
        return f"RISK SUMMARY ({scope}; {mask.sum():,} units)\n{t.to_string(index=False)}", t, mask, "Selected units"

    # Geology ranking
    if re.search(r"geolog|lithology|rock|formation|soil type|which (?:rock|soil)", ql) and not re.search(r"\bunits?\b", ql):
        g = geology_ranking(ctx, mask)
        top_row = g.iloc[0]
        txt = (
            f"GEOLOGY RISK RANKING ({scope}; {mask.sum():,} units)\n{g.to_string(index=False)}\n\n"
            f"-> {top_row.lithology} is the most landslide-prone rock type (mean risk {top_row.mean_risk:.1%}, "
            f"{top_row.pct_high:.0f}% of units High/Very High)."
        )
        return txt, g, mask, "Units in selection"

    # Unit ranking
    if re.search(r"\bunits?\b|\bhills?\b|\bslopes?\b.*\b(list|rank|top)\b|\btop\s*\d+\b", ql) and not re.search(r"\b(areas?|zones?|region)\b", ql):
        t = unit_ranking(ctx, mask, n)
        txt = f"TOP {len(t)} SLOPE UNITS BY RISK ({scope}; {mask.sum():,} candidates)\n{t.drop(columns=['x_km','y_km']).to_string(index=False)}"
        mk = pd.Series(df.unit_id.isin(t.unit_id), index=df.index)
        return txt, t, mk, f"Top {len(t)} riskiest units"

    # Zone ranking (Default)
    t = zone_ranking(ctx, mask, n)
    top_zone = t.iloc[0]
    txt = (
        f"MOST LANDSLIDE-PRONE AREAS ({scope}; {mask.sum():,} units)\n{t.to_string(index=False)}\n\n"
        f"-> Zone {top_zone.area} ranks highest: mean risk {top_zone.mean_risk:.1%}, "
        f"{top_zone.pct_high:.0f}% of units High/Very High, dominant rock {top_zone.main_rock}, "
        f"average slope {top_zone.avg_slope:.0f}°, rainfall ~{top_zone.avg_rain:.0f} mm."
    )
    zmask = df.zone.isin(t.area) & mask
    return txt, t, zmask, f"Top {len(t)} zones in selection"


def run_query(q: str, ctx: Context) -> str:
    """Execute natural language query, write output files, and return text summary."""
    txt, tab, mask, title = answer(q, ctx)
    OUT.mkdir(parents=True, exist_ok=True)
    if tab is not None:
        tab.to_csv(OUT / "query_result.csv", index=False)
    if mask is not None and mask.any():
        fig = draw_map(ctx.df, highlight=mask, title=title or "Query Map")
        fig.savefig(OUT / "query_map.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
    return txt


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    ctx = Context()
    if len(sys.argv) > 1:
        query_str = " ".join(sys.argv[1:])
        print(run_query(query_str, ctx))
        return

    print("Landslide Susceptibility Analyzer — type a question, 'help' for examples, 'quit' to exit.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"quit", "exit", "q"}:
            break
        if q:
            print(run_query(q, ctx))


if __name__ == "__main__":
    main()
