# AI-Powered Landslide Susceptibility Prediction — working model

Implements the pipeline from the presentation: 8 conditioning factors (+ geology) → preprocessing →
XGBoost (grid-searched) → probability → 5-class risk map, compared with LR / Decision Tree / SVM / Random Forest.

## Run in VS Code
1. Open this folder (File ▸ Open Folder), open a terminal (Ctrl+`).
2. `python -m venv .venv` then activate it
   (Windows: `.venv\Scripts\activate`, Mac/Linux: `source .venv/bin/activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py`   — creates `data/landslide_data.csv` (skip if you use the real dataset)
5. `python train.py`          — trains, evaluates, writes `outputs/` (metrics, ROC, confusion matrix, map, sensitivity)
6. `python analyze.py`        — **the prompt**: ask questions in plain English
7. `streamlit run app.py`     — optional dashboard in the browser

## The prompt (analyze.py)
    > which areas are most prone to landslides?
    > which rock type is most dangerous?
    > top 10 units with clay and slope > 30
    > limestone areas within 500 m of a fault
    > what if rainfall increases by 30% in clay areas?
    > explain SU-0123
    > summary of bare soil above 1200 mm rainfall
Filters you can mix: rock type (clay/shale, marl/flysch, sandstone, limestone, granite/gneiss, alluvium),
land cover, slope, elevation, rainfall, distance to fault/river/road, zone (A1…E5).
Each answer also saves `outputs/query_result.csv` and a highlighted map `outputs/query_map.png`.

## Added features beyond the slides
1. **Geology ranking** – which rock types are most prone, incl. risk within 1 km of faults (new factors: `lithology`, `fault_dist`).
2. **Rainfall what-if simulator** – re-predicts with ±X% rainfall; shows units/zones that become High/Very High.
3. **Explain a unit** – per-unit factor contributions (XGBoost SHAP-style `pred_contribs`).
4. Zone-level "which area" ranking on a named A1…E5 grid, plus sensitivity analysis (threshold + factor-group removal).

## Using the real benchmark dataset
Save it as `data/landslide_data.csv` with columns:
`unit_id, x_km, y_km, slope, elevation, aspect, curvature, rainfall, land_cover, river_dist, road_dist,
fault_dist, lithology, landslide` (x_km/y_km in 0–64 km; landslide = 0/1). Rename/derive columns as needed,
then rerun `python train.py`. Lithology/land-cover words in `analyze.py` (`LITH_WORDS`, `LC_WORDS`) should match your category names.

## Notes
- The bundled data is **synthetic** (generated from plausible geology/terrain rules) so the project runs anywhere;
  numbers in `outputs/` come from it, not from the published benchmark. Replace the slide's "illustrative" values with your own run.
- Splits are spatial (whole 8 km map blocks go to one set) to avoid leakage between neighbouring units; seed = 42.
