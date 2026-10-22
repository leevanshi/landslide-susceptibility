"""Interactive mapping utilities using Folium and Plotly for landslide susceptibility.

Exports interactive maps with layer toggles (Risk Class, Lithology, Fault Lines),
rich hover tooltips (Unit ID, Risk %, Rock Type, Slope), and HTML output.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import folium
from folium import FeatureGroup, LayerControl, Marker, PolyLine
from folium.plugins import FastMarkerCluster
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from common import (CENTER_LAT, CENTER_LON, EXTENT_KM, OUT, RISK_COLORS,
                    RISK_LABELS)

LITHOLOGY_COLORS: Dict[str, str] = {
    "clay_shale": "#d95f02",
    "marl_flysch": "#7570b3",
    "sandstone": "#e7298a",
    "limestone": "#66a61e",
    "granite_gneiss": "#e6ab02",
    "alluvium": "#1b9e77",
}


def km_to_latlon(
    x_km: Union[float, np.ndarray, pd.Series],
    y_km: Union[float, np.ndarray, pd.Series],
    center_lat: float = CENTER_LAT,
    center_lon: float = CENTER_LON,
    extent_km: float = EXTENT_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert local kilometer coordinates to approximate WGS84 latitude and longitude.

    Centered over Central Italy (Apennines: ~42.85°N, 12.65°E).
    1 deg lat ~= 111.0 km, 1 deg lon ~= 111.0 * cos(42.85 deg) ~= 81.4 km.
    """
    lat_km_per_deg = 111.0
    lon_km_per_deg = 111.0 * np.cos(np.radians(center_lat))
    rel_x = np.asarray(x_km) - (extent_km / 2.0)
    rel_y = np.asarray(y_km) - (extent_km / 2.0)
    lat = center_lat + (rel_y / lat_km_per_deg)
    lon = center_lon + (rel_x / lon_km_per_deg)
    return lat, lon


def create_interactive_map(
    df: pd.DataFrame,
    output_path: Optional[Union[str, Path]] = None,
    sample_size: Optional[int] = 3000,
    highlight_mask: Optional[pd.Series] = None,
    title: str = "Central Italy Landslide Susceptibility",
) -> folium.Map:
    """Build an interactive Folium map with layer toggles for risk class, lithology, and faults.

    Includes rich hover tooltips and popups with Unit ID, Risk %, Rock Type, Slope, etc.
    Exports to output_path if provided (e.g. outputs/map.html).
    """
    d = df.copy()
    if "lat" not in d.columns or "lon" not in d.columns:
        lats, lons = km_to_latlon(d["x_km"], d["y_km"])
        d["lat"] = lats
        d["lon"] = lons

    if "probability" not in d.columns:
        d["probability"] = 0.5
    if "risk_class" not in d.columns:
        from common import risk_class
        d["risk_class"] = risk_class(d["probability"])

    # If sampling requested to keep HTML file size reasonable for browser loading
    if sample_size and len(d) > sample_size and highlight_mask is None:
        plot_df = d.sample(n=sample_size, random_state=42)
    elif highlight_mask is not None:
        plot_df = d[highlight_mask].copy()
    else:
        plot_df = d

    m = folium.Map(
        location=[CENTER_LAT, CENTER_LON],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True,
    )

    # 1. Base Layer: Risk Class (5 tiers)
    fg_risk = FeatureGroup(name="⚠️ Landslide Risk (5 Tiers)", show=True)
    color_map = dict(zip(RISK_LABELS, RISK_COLORS))

    for _, row in plot_df.iterrows():
        r_col = color_map.get(str(row.get("risk_class", "Moderate")), "#f2d544")
        prob_pct = f"{float(row.get('probability', 0.0)):.1%}"
        slope_val = f"{float(row.get('slope', 0.0)):.1f}°"
        rock = str(row.get("lithology", "Unknown"))
        uid = str(row.get("unit_id", "N/A"))
        zone = str(row.get("zone", "N/A"))

        tooltip_text = (
            f"<b>{uid}</b> (Zone {zone})<br>"
            f"<b>Risk:</b> {prob_pct} ({row.get('risk_class')})<br>"
            f"<b>Rock:</b> {rock}<br>"
            f"<b>Slope:</b> {slope_val}<br>"
            f"<b>Rainfall:</b> {row.get('rainfall', 0):.0f} mm"
        )

        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=4,
            color=r_col,
            fill=True,
            fill_color=r_col,
            fill_opacity=0.75,
            weight=1,
            tooltip=tooltip_text,
        ).add_to(fg_risk)
    fg_risk.add_to(m)

    # 2. Lithology Layer
    fg_lith = FeatureGroup(name="🪨 Lithology (Geological Formations)", show=False)
    for _, row in plot_df.iterrows():
        rock = str(row.get("lithology", "other"))
        l_col = LITHOLOGY_COLORS.get(rock, "#888888")
        tooltip_text = (
            f"<b>{row.get('unit_id')}</b><br>"
            f"<b>Lithology:</b> {rock}<br>"
            f"<b>Land cover:</b> {row.get('land_cover', 'N/A')}<br>"
            f"<b>Fault distance:</b> {row.get('fault_dist', 0):.0f} m"
        )
        folium.CircleMarker(
            location=[row["lat"], row["lon"]],
            radius=3.5,
            color=l_col,
            fill=True,
            fill_color=l_col,
            fill_opacity=0.7,
            weight=1,
            tooltip=tooltip_text,
        ).add_to(fg_lith)
    fg_lith.add_to(m)

    # 3. Fault Lines & Hazard Corridor Layer (NW-SE Apennine thrust faults)
    fg_faults = FeatureGroup(name="⚡ Active Fault Lines & Corridor", show=False)
    # Geological fault structures in the Central Italy study sector
    fault_coords_1 = [
        [CENTER_LAT - 0.25, CENTER_LON - 0.30],
        [CENTER_LAT - 0.05, CENTER_LON - 0.10],
        [CENTER_LAT + 0.15, CENTER_LON + 0.10],
        [CENTER_LAT + 0.30, CENTER_LON + 0.25],
    ]
    fault_coords_2 = [
        [CENTER_LAT - 0.20, CENTER_LON - 0.05],
        [CENTER_LAT + 0.05, CENTER_LON + 0.15],
        [CENTER_LAT + 0.25, CENTER_LON + 0.35],
    ]
    PolyLine(
        fault_coords_1,
        color="#800000",
        weight=3,
        opacity=0.85,
        tooltip="Regional Thrust Fault F-1 (Apennine active seismic/fault system)",
    ).add_to(fg_faults)
    PolyLine(
        fault_coords_2,
        color="#b30000",
        weight=2.5,
        opacity=0.8,
        dash_array="5, 10",
        tooltip="Secondary Normal Fault F-2 (High shear susceptibility)",
    ).add_to(fg_faults)
    fg_faults.add_to(m)

    LayerControl(collapsed=False).add_to(m)

    if output_path is not None:
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        m.save(str(p))

    return m


def create_plotly_map(
    df: pd.DataFrame,
    color_by: str = "risk_class",
    highlight_mask: Optional[pd.Series] = None,
    title: str = "Interactive Slope-Unit Susceptibility",
) -> go.Figure:
    """Create a high-performance interactive Plotly map with rich hover cards."""
    d = df.copy()
    if highlight_mask is not None:
        d["highlighted"] = highlight_mask
    else:
        d["highlighted"] = True

    color_discrete_map = dict(zip(RISK_LABELS, RISK_COLORS))
    if color_by == "lithology":
        color_discrete_map = LITHOLOGY_COLORS

    d["risk_pct"] = (d["probability"] * 100).round(1).astype(str) + "%"

    fig = px.scatter(
        d[d["highlighted"]],
        x="x_km",
        y="y_km",
        color=color_by,
        color_discrete_map=color_discrete_map,
        category_orders={"risk_class": RISK_LABELS},
        hover_name="unit_id",
        hover_data={
            "x_km": False,
            "y_km": False,
            "zone": True,
            "risk_pct": True,
            "risk_class": True,
            "lithology": True,
            "slope": ":.1f",
            "rainfall": ":.0f",
            "fault_dist": ":.0f",
        },
        labels={
            "risk_pct": "Susceptibility",
            "lithology": "Rock Type",
            "slope": "Slope (°)",
            "rainfall": "Rainfall (mm)",
            "fault_dist": "Fault Dist (m)",
        },
        title=title,
    )

    fig.update_traces(marker=dict(size=6, opacity=0.82))
    fig.update_layout(
        xaxis_title="East (km)",
        yaxis_title="North (km)",
        template="plotly_white",
        yaxis=dict(scaleanchor="x", scaleratio=1),
        margin=dict(l=20, r=20, t=40, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig
