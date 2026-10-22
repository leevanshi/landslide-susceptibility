"""Streamlit dashboard for landslide susceptibility prediction, interactive geospatial exploration,
per-unit SHAP waterfall explanations, and live rainfall early warning.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import shap
import streamlit as st
import streamlit.components.v1 as components

matplotlib.use("Agg")

from analyze import (Context, HELP, answer, compare_scenarios, compare_zones,
                    early_warning_assessment, explain_unit, explain_unit_shap,
                    geology_ranking, rainfall_scenario)
from common import (CONFIG, FEATURES, OUT, RISK_COLORS, RISK_LABELS, risk_class)
from map_utils import create_interactive_map, create_plotly_map

st.set_page_config(
    page_title="AI Landslide Susceptibility & Early Warning",
    page_icon="⛰️",
    layout="wide",
)


@st.cache_resource
def get_ctx() -> Context:
    return Context()


ctx = get_ctx()
df = ctx.df
is_synthetic = df.attrs.get("is_synthetic", True)
provenance = df.attrs.get("provenance", "Synthetic Data")

# Top Banner & Metrics
st.title("⛰️ AI Landslide Susceptibility & Early Warning System")
if is_synthetic:
    st.info(
        f"ℹ️ **Data Provenance:** {provenance}. "
        "Illustrative synthetic simulation following Central Italy Apennines topography. "
        "Provide Alvioli et al. (2024) benchmark CSV at `data/alvioli_2024_benchmark.csv` for verified real data."
    )
else:
    st.success(f"✅ **Verified Benchmark:** {provenance} (Alvioli et al., 2024).")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total Slope Units", f"{len(df):,}")
c2.metric("High / Very High", f"{df.risk_class.isin(['High', 'Very High']).mean():.1%}")
c3.metric("Most Prone Rock", geology_ranking(ctx, pd.Series(True, index=df.index)).iloc[0].lithology)
c4.metric("Most Prone Zone", df.groupby("zone").probability.mean().idxmax())
c5.metric("Model Calibration", "Isotonic (Active)" if ctx.is_calibrated else "Standard XGBoost")

# Tabs
tab_map, tab_early, tab_exp, tab_ask, tab_rain, tab_geo = st.tabs([
    "🗺️ Interactive Risk Map",
    "🚨 Live Early Warning",
    "🔍 Per-Unit SHAP Explanation",
    "💬 Ask the Data (LLM / Rules)",
    "🌧️ Rainfall What-If",
    "🪨 Geology & Zones",
])

# --------------------------------------------------------------------------- TAB 1: Real Map
with tab_map:
    st.subheader("Geospatial Susceptibility Map")
    map_mode = st.radio(
        "Map Render Engine:",
        ["Leaflet / Folium (Layer Toggles & Fault Lines)", "Plotly High-Performance Scatter"],
        horizontal=True,
    )

    map_html_file = OUT / "map.html"
    if not map_html_file.exists():
        create_interactive_map(df, output_path=map_html_file, sample_size=3000)

    if map_mode.startswith("Leaflet"):
        st.caption("Layer toggles in top-right: Landslide Risk Tiers, Geological Lithology, Fault Lines & Corridors.")
        try:
            with open(map_html_file, "r", encoding="utf-8") as f:
                html_code = f.read()
            components.html(html_code, height=620, scrolling=False)
        except Exception as e:
            st.error(f"Error loading map.html: {e}")

        with open(map_html_file, "rb") as f:
            st.download_button(
                label="📥 Download Interactive map.html",
                data=f,
                file_name="landslide_map.html",
                mime="text/html",
            )
    else:
        color_choice = st.selectbox("Color slope units by:", ["risk_class", "lithology"])
        plotly_fig = create_plotly_map(df, color_by=color_choice, title="Central Italy Slope-Unit Risk")
        st.plotly_chart(plotly_fig, use_container_width=True)

    if (OUT / "model_comparison.csv").exists():
        st.subheader("Model Performance & Calibration (Spatial Test Set)")
        st.dataframe(pd.read_csv(OUT / "model_comparison.csv"), hide_index=True)


# --------------------------------------------------------------------------- TAB 2: Early Warning
with tab_early:
    st.subheader("🚨 Live Rainfall Early Warning Mode")
    st.write(
        "Computes dynamic 24-hour storm shock impact across all slope units, "
        "categorizing units into Green (Normal), Yellow (Advisory), Orange (Watch), and Red (Warning) evacuation tiers."
    )

    ew_col1, ew_col2 = st.columns([1, 2])
    with ew_col1:
        use_om = st.checkbox("Fetch Live Forecast from Open-Meteo API", value=True)
        if use_om:
            st.caption("Coordinates: 42.85°N, 12.65°E (Central Italy Apennines)")
            rain_input = None
        else:
            rain_input = st.number_input("24-Hour Rainfall Shock (mm):", min_value=0.0, max_value=300.0, value=65.0, step=5.0)

        run_ew = st.button("Run Early Warning Assessment", type="primary")

    if run_ew or "last_ew" in st.session_state:
        if run_ew:
            summary_txt, alerts_df, counts = early_warning_assessment(ctx, rainfall_24h_mm=rain_input, use_open_meteo=use_om)
            st.session_state["last_ew"] = (summary_txt, alerts_df, counts)
        else:
            summary_txt, alerts_df, counts = st.session_state["last_ew"]

        with ew_col2:
            st.code(summary_txt, language="text")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("🔴 Red (Warning)", f"{counts.get('Red (Warning)', 0):,}", help="Immediate risk of failure; evacuation recommended")
        m2.metric("🟠 Orange (Watch)", f"{counts.get('Orange (Watch)', 0):,}", help="High slope instability potential")
        m3.metric("🟡 Yellow (Advisory)", f"{counts.get('Yellow (Advisory)', 0):,}", help="Elevated monitoring required")
        m4.metric("🟢 Green (Normal)", f"{counts.get('Green (Normal)', 0):,}", help="Baseline equilibrium")

        st.subheader("Critical Warning Units (Red / Orange Alert List)")
        critical = alerts_df[alerts_df.alert_level.str.startswith(("Red", "Orange"))]
        st.dataframe(critical.head(100), hide_index=True)

        st.download_button(
            "📥 Download Complete Early Warning CSV",
            alerts_df.to_csv(index=False),
            file_name="early_warning_alerts.csv",
            mime="text/csv",
        )


# --------------------------------------------------------------------------- TAB 3: SHAP Waterfall
with tab_exp:
    st.subheader("🔍 Explain a Slope Unit (Per-Unit SHAP Waterfall)")
    st.write("TreeSHAP attribution breaks down the exact base rate and feature pushes (positive or negative) driving the score.")

    top_units = df.sort_values("probability", ascending=False).unit_id.head(300).tolist()
    selected_uid = st.selectbox("Select Slope Unit ID:", top_units)

    row, shap_exp = explain_unit_shap(ctx, selected_uid)
    if row is not None and shap_exp is not None:
        u_col1, u_col2 = st.columns([1, 2])
        with u_col1:
            st.write(f"### Unit: **{selected_uid}**")
            st.markdown(f"- **Zone:** `{row.zone}`")
            st.markdown(f"- **Susceptibility:** **{row.probability:.1%}** ({row.risk_class})")
            st.markdown(f"- **Lithology:** `{row.lithology}`")
            st.markdown(f"- **Land Cover:** `{row.land_cover}`")
            st.markdown(f"- **Slope Gradient:** `{row.slope:.1f}°`")
            st.markdown(f"- **Elevation:** `{row.elevation:.0f} m`")
            st.markdown(f"- **Rainfall:** `{row.rainfall:.0f} mm`")
            st.markdown(f"- **Fault Proximity:** `{row.fault_dist:.0f} m`")

        with u_col2:
            st.subheader("SHAP Waterfall Attribution Plot")
            plt.clf()
            fig, ax = plt.subplots(figsize=(9, 5.5))
            shap.plots.waterfall(shap_exp, max_display=10, show=False)
            plt.title(f"SHAP Waterfall: Feature Contribution to {selected_uid}", fontsize=11)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close("all")


# --------------------------------------------------------------------------- TAB 4: Ask the Data
with tab_ask:
    st.subheader("💬 Ask the Data in Natural Language")
    st.caption("Powered by Claude API structured JSON translation with offline rule-based parser fallback.")

    col_q1, col_q2 = st.columns([3, 1])
    with col_q1:
        preset = st.selectbox(
            "Example questions:",
            [
                "Custom query...",
                "compare zone A1 vs C2",
                "what changed between scenario 10% and 30%",
                "which areas are most prone to landslides on clay or marl rock with slope > 25?",
                "top 15 units with clay and slope > 30",
                "which rock type is most dangerous?",
                "what if rainfall increases by 30% in clay areas?",
            ],
        )
    with col_q2:
        api_status = "Claude API Active" if os.environ.get("ANTHROPIC_API_KEY") else "Offline Rule Engine Active"
        st.metric("Parser Engine", api_status)

    default_text = preset if preset != "Custom query..." else "compare zone A1 vs C2"
    q_input = st.text_input("Your question:", value=default_text)

    if q_input:
        txt, tab_res, mask, q_title = answer(q_input, ctx)
        st.code(txt, language="text")

        if tab_res is not None:
            st.dataframe(tab_res.head(50), hide_index=True)
            st.download_button(
                "📥 Download Result CSV",
                tab_res.to_csv(index=False),
                file_name="query_result.csv",
                mime="text/csv",
            )


# --------------------------------------------------------------------------- TAB 5: Rainfall What-If
with tab_rain:
    st.subheader("🌧️ Rainfall Climate & Storm Scenario Simulator")
    pct_val = st.slider("Rainfall percentage change (%):", -50, 100, 30, 5)
    rock_filter = st.multiselect("Limit scenario to rock types:", sorted(df.lithology.dropna().unique()))
    scen_mask = df.lithology.isin(rock_filter) if rock_filter else pd.Series(True, index=df.index)

    d_scen = rainfall_scenario(ctx, scen_mask, pct_val)
    newly_high = d_scen[(~d_scen.risk_class.isin(["High", "Very High"])) & d_scen.scenario_class.isin(["High", "Very High"])]

    s1, s2, s3 = st.columns(3)
    s1.metric("Mean Susceptibility", f"{d_scen.scenario_prob.mean():.1%}", f"{(d_scen.scenario_prob.mean() - d_scen.probability.mean()) * 100:+.1f} pts")
    s2.metric("Newly Escalated High/Very-High Units", f"{len(newly_high):,}")
    s3.metric("Evaluated Slope Units", f"{len(d_scen):,}")

    p_scen_fig = create_plotly_map(
        d_scen,
        color_by="scenario_class",
        highlight_mask=d_scen.index.isin(newly_high.index),
        title=f"Newly High/Very High Slope Units at {pct_val:+d}% Rainfall",
    )
    st.plotly_chart(p_scen_fig, use_container_width=True)


# --------------------------------------------------------------------------- TAB 6: Geology & Zones
with tab_geo:
    st.subheader("🪨 Geological Formations & Fault Proximity Analysis")
    g_tab = geology_ranking(ctx, pd.Series(True, index=df.index))
    st.dataframe(g_tab, hide_index=True)
    st.bar_chart(g_tab.set_index("lithology")["mean_risk"])

    st.subheader("Spatial Grid Zone Overview (A1 to E5)")
    z_tab = df.groupby("zone").agg(
        units=("unit_id", "count"),
        mean_risk=("probability", "mean"),
        pct_high=("risk_class", lambda s: 100 * s.isin(["High", "Very High"]).mean()),
        main_rock=("lithology", lambda s: s.mode().iat[0] if s.notna().any() else "N/A"),
        avg_slope=("slope", "mean"),
        avg_rainfall=("rainfall", "mean"),
    ).round({"mean_risk": 3, "pct_high": 1, "avg_slope": 1, "avg_rainfall": 0}).reset_index()
    st.dataframe(z_tab, hide_index=True)
