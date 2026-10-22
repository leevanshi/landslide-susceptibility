"""Create a realistic synthetic slope-unit dataset (Central-Italy style, 7,360 units, ~4,100 km2).

The real benchmark (Alvioli et al., 2024) can be used instead: put your CSV at
data/landslide_data.csv with the same column names (see README).
"""
import numpy as np
import pandas as pd

from common import DATA_PATH, EXTENT_KM, SEED

N = 7360
LITHOLOGIES = {  # name: (landslide-proneness effect on the logit, share of map)
    "clay_shale": (1.30, 0.20),
    "marl_flysch": (0.80, 0.22),
    "sandstone": (0.05, 0.18),
    "limestone": (-0.45, 0.20),
    "granite_gneiss": (-0.95, 0.08),
    "alluvium": (-0.25, 0.12),
}
LC_EFFECT = {"forest": -0.60, "farmland": 0.10, "grassland": 0.0, "bare_soil": 0.90, "urban": 0.20}


def smooth_field(rng, x, y, k=7):
    f = np.zeros_like(x)
    for _ in range(k):
        fx, fy = rng.uniform(0.04, 0.25, 2)
        f += rng.uniform(0.4, 1.0) * np.sin(fx * x + rng.uniform(0, 6.3)) * np.cos(fy * y + rng.uniform(0, 6.3))
    return f


def main():
    rng = np.random.default_rng(SEED)
    x = rng.uniform(0, EXTENT_KM, N)
    y = rng.uniform(0, EXTENT_KM, N)

    # terrain: smooth relief -> elevation, slope from local gradient
    seed_state = rng.bit_generator.state
    def field(px, py):
        r = np.random.default_rng(11)
        return smooth_field(r, px, py)
    f = field(x, y)
    h = 0.05
    gx = (field(x + h, y) - field(x - h, y)) / (2 * h)
    gy = (field(x, y + h) - field(x, y - h)) / (2 * h)
    grad = np.hypot(gx, gy)
    fn = (f - f.min()) / (f.max() - f.min())
    elevation = 150 + 1350 * fn + rng.normal(0, 40, N)
    slope = np.clip(4 + 42 * (grad / np.percentile(grad, 98)) + rng.normal(0, 4.5, N), 1, 58)
    aspect = rng.uniform(0, 360, N)
    curvature = rng.normal(0, 1, N) + 0.4 * (rng.normal(size=N))
    rainfall = np.clip(650 + 0.55 * elevation + 120 * np.sin(x / 9) + rng.normal(0, 90, N), 500, 2400)

    # geology: patchy lithology zones (nearest seed point)
    names = list(LITHOLOGIES)
    seeds = rng.uniform(0, EXTENT_KM, (16, 2))
    seed_lith = rng.choice(names, 16, p=[LITHOLOGIES[n][1] for n in names])
    d = np.hypot(x[:, None] - seeds[:, 0], y[:, None] - seeds[:, 1])
    lithology = seed_lith[d.argmin(1)]
    lith_eff = np.array([LITHOLOGIES[l][0] for l in lithology])

    # distance to 3 fault lines (metres)
    fault_d = np.full(N, np.inf)
    for _ in range(3):
        p0, p1 = rng.uniform(0, EXTENT_KM, 2), rng.uniform(0, EXTENT_KM, 2)
        v = p1 - p0
        t = np.clip(((x - p0[0]) * v[0] + (y - p0[1]) * v[1]) / (v @ v), 0, 1)
        dist = np.hypot(x - (p0[0] + t * v[0]), y - (p0[1] + t * v[1]))
        fault_d = np.minimum(fault_d, dist)
    fault_dist = fault_d * 1000 + rng.uniform(0, 80, N)

    river_dist = rng.exponential(600, N) * (1.2 - 0.5 * fn)
    road_dist = rng.exponential(900, N)
    road_dist[rng.choice(N, 8, replace=False)] *= 12          # a few outliers

    # land cover depends on slope / elevation
    lc = np.empty(N, dtype=object)
    for i in range(N):
        if slope[i] < 10 and elevation[i] < 500:
            pr = [0.10, 0.50, 0.15, 0.02, 0.23]
        elif slope[i] < 22:
            pr = [0.35, 0.30, 0.25, 0.07, 0.03]
        else:
            pr = [0.50, 0.04, 0.22, 0.24, 0.0]
        lc[i] = rng.choice(list(LC_EFFECT), p=pr)
    lc_eff = np.array([LC_EFFECT[c] for c in lc])

    weak_rock = np.isin(lithology, ["clay_shale", "marl_flysch"])
    logit = (0.085 * (slope - 22) + 0.0013 * (rainfall - 1100) + lith_eff + lc_eff
             + 1.1 * np.exp(-fault_dist / 1500) + 0.45 * np.exp(-road_dist / 300)
             + 0.30 * np.exp(-river_dist / 200) - 0.18 * curvature
             # non-linear interactions (what tree models can learn and a straight line cannot)
             + 0.9 * ((slope > 30) & (rainfall > 1300)) + 0.8 * (weak_rock & (slope > 20))
             + 0.7 * ((lc == "bare_soil") & (rainfall > 1200)) - 0.8 * ((slope < 8) & ~weak_rock))
    logit = 1.7 * logit + rng.normal(0, 0.45, N)
    lo, hi = -10, 10                                           # intercept so classes ~50/50
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > 0.5:
            hi = mid
        else:
            lo = mid
    label = (rng.uniform(size=N) < 1 / (1 + np.exp(-(logit + mid)))).astype(int)

    df = pd.DataFrame({
        "unit_id": [f"SU-{i + 1:04d}" for i in range(N)],
        "x_km": x.round(3), "y_km": y.round(3),
        "slope": slope.round(1), "elevation": elevation.round(0), "aspect": aspect.round(0),
        "curvature": curvature.round(2), "rainfall": rainfall.round(0),
        "land_cover": lc, "river_dist": river_dist.round(0), "road_dist": road_dist.round(0),
        "fault_dist": fault_dist.round(0), "lithology": lithology, "landslide": label,
    })
    for col in ["rainfall", "curvature", "river_dist", "land_cover", "lithology"]:   # ~2% gaps
        df.loc[rng.choice(N, int(0.02 * N), replace=False), col] = np.nan
    DATA_PATH.parent.mkdir(exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Saved {len(df)} slope units -> {DATA_PATH}  (landslide share: {label.mean():.0%})")


if __name__ == "__main__":
    main()
