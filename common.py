"""Shared constants, configuration, schema validation, and helpers for landslide susceptibility."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# Base project paths (dynamic, no hard-coded absolute paths)
ROOT: Path = Path(__file__).resolve().parent
CONFIG_PATH: Path = ROOT / "config.yaml"


def load_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load configuration from YAML file or return robust defaults."""
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to parse config.yaml: {e}. Using internal defaults.")
    return {}


CONFIG: Dict[str, Any] = load_config()

# Paths
paths_cfg = CONFIG.get("paths", {})
DATA_PATH: Path = ROOT / paths_cfg.get("synthetic_data", "data/landslide_data.csv")
BENCHMARK_PATH: Path = ROOT / paths_cfg.get("benchmark_data", "data/alvioli_2024_benchmark.csv")
MODEL_PATH: Path = ROOT / paths_cfg.get("model_path", "models/xgb_pipeline.joblib")
CALIBRATED_MODEL_PATH: Path = ROOT / paths_cfg.get("calibrated_model_path", "models/xgb_calibrated.joblib")
OUT: Path = ROOT / paths_cfg.get("outputs_dir", "outputs")

# Project settings
SEED: int = int(CONFIG.get("project", {}).get("seed", 42))

feat_cfg = CONFIG.get("features", {})
NUM: List[str] = feat_cfg.get(
    "numeric",
    ["slope", "elevation", "aspect", "curvature", "rainfall", "river_dist", "road_dist", "fault_dist"],
)
CAT: List[str] = feat_cfg.get("categorical", ["land_cover", "lithology"])
FEATURES: List[str] = NUM + CAT
TARGET: str = feat_cfg.get("target", "landslide")

# Risk classification
rc_cfg = CONFIG.get("risk_classes", {})
RISK_LABELS: List[str] = rc_cfg.get("labels", ["Very Low", "Low", "Moderate", "High", "Very High"])
RISK_BINS: List[float] = rc_cfg.get("bins", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0001])
RISK_COLORS: List[str] = rc_cfg.get("colors", ["#2e8b57", "#9acd32", "#f2d544", "#f28c28", "#c0392b"])

# Spatial layout
spatial_cfg = CONFIG.get("spatial", {})
GRID_N: int = int(spatial_cfg.get("grid_n", 5))
EXTENT_KM: float = float(spatial_cfg.get("extent_km", 64.0))
BLOCK_KM: float = float(spatial_cfg.get("block_km", 8.0))
SPLIT_FRACTIONS: Tuple[float, float, float] = tuple(spatial_cfg.get("split_fractions", [0.70, 0.15, 0.15]))  # type: ignore
CENTER_LAT: float = float(spatial_cfg.get("center_lat", 42.85))
CENTER_LON: float = float(spatial_cfg.get("center_lon", 12.65))

# Alvioli et al. (2024) Central Italy benchmark mapping dictionary
ALVIOLI_COLUMN_MAP: Dict[str, str] = {
    # Unique unit IDs
    "su_id": "unit_id",
    "suid": "unit_id",
    "id": "unit_id",
    "objectid": "unit_id",
    "slope_unit": "unit_id",
    "unit_id": "unit_id",
    # Spatial coordinates
    "x": "x_km",
    "x_coord": "x_km",
    "coord_x": "x_km",
    "east": "x_km",
    "easting": "x_km",
    "lon": "x_km",
    "longitude": "x_km",
    "x_km": "x_km",
    "y": "y_km",
    "y_coord": "y_km",
    "coord_y": "y_km",
    "north": "y_km",
    "northing": "y_km",
    "lat": "y_km",
    "latitude": "y_km",
    "y_km": "y_km",
    # Morphometric features
    "slope_mean": "slope",
    "mean_slope": "slope",
    "slope_deg": "slope",
    "slope_avg": "slope",
    "slope": "slope",
    "elev_mean": "elevation",
    "elevation_mean": "elevation",
    "mean_elev": "elevation",
    "dem_mean": "elevation",
    "altitude": "elevation",
    "elevation": "elevation",
    "aspect_mean": "aspect",
    "mean_aspect": "aspect",
    "aspect": "aspect",
    "curv_mean": "curvature",
    "mean_curv": "curvature",
    "plan_curvature": "curvature",
    "profile_curvature": "curvature",
    "curvature": "curvature",
    # Climate / trigger
    "precip_mean": "rainfall",
    "rain_mean": "rainfall",
    "precipitation": "rainfall",
    "annual_rain": "rainfall",
    "precip": "rainfall",
    "rainfall": "rainfall",
    # Distances
    "dist_river": "river_dist",
    "river_distance": "river_dist",
    "stream_dist": "river_dist",
    "dist_stream": "river_dist",
    "hydro_dist": "river_dist",
    "river_dist": "river_dist",
    "dist_road": "road_dist",
    "road_distance": "road_dist",
    "roads_dist": "road_dist",
    "dist_to_road": "road_dist",
    "road_dist": "road_dist",
    "dist_fault": "fault_dist",
    "fault_distance": "fault_dist",
    "faults_dist": "fault_dist",
    "dist_to_fault": "fault_dist",
    "fault_dist": "fault_dist",
    # Thematic / geology
    "landcover": "land_cover",
    "land_use": "land_cover",
    "landuse": "land_cover",
    "corine": "land_cover",
    "clc": "land_cover",
    "land_cover": "land_cover",
    "litho": "lithology",
    "geology": "lithology",
    "rock_type": "lithology",
    "geo_unit": "lithology",
    "lithology": "lithology",
    # Target landslide label
    "landslide_presence": "landslide",
    "landslides": "landslide",
    "ls_presence": "landslide",
    "target": "landslide",
    "label": "landslide",
    "class": "landslide",
    "landslide": "landslide",
}


class SchemaValidationError(ValueError):
    """Raised when an input dataset fails the required landslide schema validation."""
    pass


class Winsorizer(BaseEstimator, TransformerMixin):
    """Cap extreme values at the 1st / 99th percentile (outlier handling)."""

    def fit(self, X: Sequence[Sequence[float]] | np.ndarray, y: Optional[Any] = None) -> Winsorizer:
        X_arr = np.asarray(X, dtype=float)
        self.lo_ = np.nanpercentile(X_arr, 1, axis=0)
        self.hi_ = np.nanpercentile(X_arr, 99, axis=0)
        return self

    def transform(self, X: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(X, dtype=float), self.lo_, self.hi_)

    def get_feature_names_out(self, names: Optional[Sequence[str]] = None) -> np.ndarray:
        return np.asarray(names) if names is not None else np.array([])


def make_preprocessor(num: Sequence[str] = NUM, cat: Sequence[str] = CAT) -> ColumnTransformer:
    """Construct column preprocessor pipeline with imputation, winsorization, and scaling."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap", Winsorizer()),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    parts = []
    if num:
        parts.append(("num", numeric, list(num)))
    if cat:
        parts.append(("cat", categorical, list(cat)))
    return ColumnTransformer(parts)


def zone_of(x: Union[pd.Series, np.ndarray, Sequence[float]], y: Union[pd.Series, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Name the map cell of a point: column letter (west->east A..E) + row number (south->north 1..5)."""
    step = EXTENT_KM / GRID_N
    col = np.clip((np.asarray(x) // step).astype(int), 0, GRID_N - 1)
    row = np.clip((np.asarray(y) // step).astype(int), 0, GRID_N - 1)
    letters = np.array(list("ABCDE"[:GRID_N]))
    return letters[col] + (row + 1).astype(str)


def risk_class(p: Union[pd.Series, np.ndarray, Sequence[float]]) -> np.ndarray:
    """Map probabilities into 5 qualitative risk classes."""
    return pd.cut(pd.Series(p), RISK_BINS, labels=RISK_LABELS, right=False).astype(str).values


def validate_dataset_schema(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """Validate that the DataFrame conforms to the required landslide schema.

    Returns:
        (is_valid, error_messages)
    """
    errors: List[str] = []
    required_cols = ["unit_id", "x_km", "y_km", TARGET] + FEATURES
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        errors.append(
            f"Missing required columns: {missing}. Available columns: {list(df.columns)}"
        )
        return False, errors

    # Check target
    target_vals = set(df[TARGET].dropna().unique())
    if not target_vals.issubset({0, 1, 0.0, 1.0}):
        errors.append(f"Target column '{TARGET}' must be binary 0/1. Found unique values: {target_vals}")

    # Check numeric features
    for col in ["x_km", "y_km"] + NUM:
        if not pd.api.types.is_numeric_dtype(df[col]):
            # Try coercion
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() == 0 and len(df) > 0:
                errors.append(f"Feature column '{col}' could not be parsed as numeric.")

    # Check categorical features
    for col in CAT:
        if df[col].dropna().empty:
            errors.append(f"Categorical column '{col}' contains only null values.")

    return len(errors) == 0, errors


def map_alvioli_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map Alvioli et al. (2024) Central Italy benchmark columns to internal schema."""
    df = df.copy()
    rename_dict: Dict[str, str] = {}
    lower_map = {k.lower(): v for k, v in ALVIOLI_COLUMN_MAP.items()}

    for col in df.columns:
        col_clean = str(col).strip()
        cl = col_clean.lower()
        if cl in lower_map:
            rename_dict[col] = lower_map[cl]

    if rename_dict:
        df = df.rename(columns=rename_dict)

    # If coordinates are in meters (e.g. UTM projected > 1000 m), normalize to km extent
    if "x_km" in df.columns and "y_km" in df.columns:
        if pd.api.types.is_numeric_dtype(df["x_km"]) and pd.api.types.is_numeric_dtype(df["y_km"]):
            x_max, y_max = df["x_km"].max(), df["y_km"].max()
            if x_max > 1000 or y_max > 1000:
                logger.info("Projected coordinates detected in meters; scaling to relative km extent.")
                df["x_km"] = (df["x_km"] - df["x_km"].min()) / 1000.0
                df["y_km"] = (df["y_km"] - df["y_km"].min()) / 1000.0
                # Scale into EXTENT_KM if broader
                if df["x_km"].max() > EXTENT_KM:
                    df["x_km"] = df["x_km"] * (EXTENT_KM / df["x_km"].max())
                if df["y_km"].max() > EXTENT_KM:
                    df["y_km"] = df["y_km"] * (EXTENT_KM / df["y_km"].max())

    return df


def load_data(path: Optional[Union[str, Path]] = None) -> pd.DataFrame:
    """Load landslide susceptibility dataset with benchmark validation and synthetic fallback.

    If path is None:
      1. Tries the Alvioli et al. (2024) Central Italy benchmark at `BENCHMARK_PATH`.
      2. If missing or invalid, falls back to the synthetic dataset at `DATA_PATH`.
    """
    is_benchmark = False
    chosen_path: Path

    if path is not None:
        chosen_path = Path(path)
        if not chosen_path.exists():
            raise FileNotFoundError(f"Requested dataset path not found: {chosen_path}")
    else:
        # Check if benchmark file exists
        if BENCHMARK_PATH.exists():
            chosen_path = BENCHMARK_PATH
            is_benchmark = True
        else:
            chosen_path = DATA_PATH

    if not chosen_path.exists():
        raise SystemExit(
            f"Dataset not found at: {chosen_path}\n"
            f"Please run 'python generate_data.py' or provide the Alvioli benchmark at {BENCHMARK_PATH}."
        )

    try:
        raw_df = pd.read_csv(chosen_path)
    except Exception as e:
        if is_benchmark and DATA_PATH.exists():
            logger.warning(f"Error reading benchmark file ({e}). Falling back to synthetic data: {DATA_PATH}")
            raw_df = pd.read_csv(DATA_PATH)
            chosen_path = DATA_PATH
            is_benchmark = False
        else:
            raise

    # Apply column mapping
    df = map_alvioli_columns(raw_df)
    is_valid, errors = validate_dataset_schema(df)

    if not is_valid:
        if is_benchmark and DATA_PATH.exists():
            logger.warning(
                f"Benchmark file schema validation failed:\n  - " + "\n  - ".join(errors) +
                f"\nFalling back to synthetic dataset: {DATA_PATH}"
            )
            df = pd.read_csv(DATA_PATH)
            is_benchmark = False
            is_valid, errors = validate_dataset_schema(df)
            if not is_valid:
                raise SchemaValidationError(f"Synthetic dataset validation failed: {errors}")
        else:
            raise SchemaValidationError(
                f"Dataset at '{chosen_path}' failed schema validation:\n  - " + "\n  - ".join(errors)
            )

    df = df.drop_duplicates("unit_id").reset_index(drop=True)
    df["zone"] = zone_of(df.x_km, df.y_km)

    # Attach dataset metadata
    df.attrs["is_synthetic"] = not is_benchmark
    df.attrs["source_file"] = str(chosen_path)
    df.attrs["provenance"] = (
        "Alvioli et al. (2024) Central Italy Benchmark"
        if is_benchmark
        else "Synthetic Simulation (Central Italy parameters - non-real data)"
    )

    return df


def spatial_split(
    df: pd.DataFrame,
    seed: int = SEED,
    fractions: Tuple[float, float, float] = SPLIT_FRACTIONS,
    block_km: float = BLOCK_KM,
) -> Tuple[np.ndarray, np.ndarray]:
    """Spatial block partition (70/15/15) ensuring neighboring units remain strictly grouped.

    Returns:
        (split, block): split array (0: train, 1: val, 2: test) and block group array.
    """
    bx = (df.x_km // block_km).astype(int)
    by = (df.y_km // block_km).astype(int)
    block = (bx * 1000 + by).values
    rng = np.random.default_rng(seed)
    unique_blocks = np.unique(block)
    blocks = rng.permutation(unique_blocks)
    sizes = pd.Series(block).value_counts()
    target = np.array(fractions) * len(df)
    assign: Dict[int, int] = {}
    filled = np.zeros(3)

    for b in blocks:
        # Assign to split furthest below quota
        k = int(np.argmax((target - filled) / target))
        assign[b] = k
        filled[k] += sizes[b]

    split = np.array([assign[b] for b in block])
    return split, block
