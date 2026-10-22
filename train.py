"""Train + evaluate models with spatial CV, isotonic calibration, SHAP explainability, and interactive maps.

Run:  python train.py
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import (CALIBRATED_MODEL_PATH, CAT, CONFIG, FEATURES, MODEL_PATH,
                    NUM, OUT, RISK_COLORS, RISK_LABELS, SEED, TARGET,
                    load_data, make_preprocessor, risk_class, spatial_split)
from map_utils import create_interactive_map

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


def xgb(**kw: Any) -> XGBClassifier:
    """Create XGBoost classifier instance with fixed random seed."""
    return XGBClassifier(eval_metric="logloss", random_state=SEED, n_jobs=-1, **kw)


def metrics(y: Sequence[int] | np.ndarray, p: Sequence[float] | np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Calculate key performance metrics including Brier score."""
    pred = (np.asarray(p) >= thr).astype(int)
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    return {
        "Accuracy": float(accuracy_score(y_arr, pred)),
        "Precision": float(precision_score(y_arr, pred, zero_division=0)),
        "Recall": float(recall_score(y_arr, pred, zero_division=0)),
        "F1": float(f1_score(y_arr, pred, zero_division=0)),
        "AUC": float(roc_auc_score(y_arr, p_arr)),
        "Brier": float(brier_score_loss(y_arr, p_arr)),
    }


def spatial_cv_evaluation(
    models: Dict[str, Pipeline],
    X: pd.DataFrame,
    y: pd.Series,
    groups: np.ndarray,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Evaluate models using strict spatial cross-validation to prevent block leakage."""
    gkf = GroupKFold(n_splits=n_splits)
    rows: List[Dict[str, Any]] = []

    for name, model_template in models.items():
        fold_scores: Dict[str, List[float]] = {
            "Accuracy": [], "Precision": [], "Recall": [], "F1": [], "AUC": [], "Brier": []
        }
        for tr_idx, val_idx in gkf.split(X, y, groups=groups):
            # Clone and fit
            m = joblib.load(joblib.dump(model_template, (OUT / "temp_m.joblib"))[0])
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p_val = m.predict_proba(X.iloc[val_idx])[:, 1]
            fold_m = metrics(y.iloc[val_idx], p_val)
            for k, v in fold_m.items():
                fold_scores[k].append(v)

        if (OUT / "temp_m.joblib").exists():
            (OUT / "temp_m.joblib").unlink(missing_ok=True)

        row = {"Model": name}
        for k, v_list in fold_scores.items():
            row[f"{k}_mean"] = round(float(np.mean(v_list)), 3)
            row[f"{k}_std"] = round(float(np.std(v_list)), 3)
        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_data()
    provenance = df.attrs.get("provenance", "Synthetic Data")
    is_synthetic = df.attrs.get("is_synthetic", True)
    print("=" * 80)
    print(f"LANDSLIDE SUSCEPTIBILITY PIPELINE")
    print(f"Dataset Provenance: {provenance}")
    if is_synthetic:
        print("[!] Note: Running on synthetic simulation benchmark. Results are illustrative.")
    else:
        print("[*] Benchmark verified: Alvioli et al. (2024) Central Italy slope units.")
    print("=" * 80)

    split, groups = spatial_split(df)
    X, y = df[FEATURES], df[TARGET]
    tr, va, te = split == 0, split == 1, split == 2
    print(f"Units: {len(df):,} | Train/Val/Test = {tr.sum()}/{va.sum()}/{te.sum()} (Spatial block isolation)")

    # 1. Hyperparameter tuning for XGBoost via Spatial CV (GroupKFold)
    train_cfg = CONFIG.get("training", {})
    grid = train_cfg.get("grid_search", {
        "model__n_estimators": [100, 300],
        "model__max_depth": [3, 5],
        "model__learning_rate": [0.05, 0.1],
    })
    search = GridSearchCV(
        Pipeline([("prep", make_preprocessor()), ("model", xgb())]),
        grid,
        scoring="roc_auc",
        cv=GroupKFold(n_splits=train_cfg.get("cv_folds", 5)),
        n_jobs=-1,
    )
    search.fit(X[tr], y[tr], groups=groups[tr])
    best_params = {k.split("__")[1]: v for k, v in search.best_params_.items()}
    print(f"Best XGBoost settings: {best_params} (Spatial CV AUC: {search.best_score_:.3f})")
    best_xgb: Pipeline = search.best_estimator_

    # Baseline models comparison
    models: Dict[str, Pipeline] = {
        "Logistic Regression": Pipeline([("prep", make_preprocessor()), ("model", LogisticRegression(max_iter=1000))]),
        "Decision Tree": Pipeline([("prep", make_preprocessor()), ("model", DecisionTreeClassifier(max_depth=6, random_state=SEED))]),
        "SVM": Pipeline([("prep", make_preprocessor()), ("model", SVC(probability=True, random_state=SEED))]),
        "Random Forest": Pipeline([("prep", make_preprocessor()), ("model", RandomForestClassifier(300, random_state=SEED, n_jobs=-1))]),
        "Proposed (XGBoost)": best_xgb,
    }

    # 2. Strict Spatial Cross-Validation for ALL reported metrics
    print("\nRunning 5-fold Spatial Cross-Validation across all baseline models...")
    cv_summary = spatial_cv_evaluation(models, X[tr | va], y[tr | va], groups[tr | va], n_splits=5)
    cv_summary.to_csv(OUT / "spatial_cv_metrics.csv", index=False)
    print("\nSpatial CV Metrics (Mean +/- Std):")
    print(cv_summary[["Model", "AUC_mean", "AUC_std", "F1_mean", "Brier_mean", "Brier_std"]].to_string(index=False))

    # 3. Model Training on Train set, Evaluation on Held-out Test set
    rows = []
    probs: Dict[str, np.ndarray] = {}
    for name, m in models.items():
        if name != "Proposed (XGBoost)":
            m.fit(X[tr], y[tr])
        p_te = m.predict_proba(X[te])[:, 1]
        probs[name] = p_te
        rows.append({"Model": name, **metrics(y[te], p_te)})
        print(f"  {name:<22} test AUC: {rows[-1]['AUC']:.3f} | Recall: {rows[-1]['Recall']:.3f} | Brier: {rows[-1]['Brier']:.3f}")

    # 4. Probability Calibration (Isotonic Regression)
    # Fit calibrator on validation fold (va) - NEVER touching the test set (te)
    try:
        from sklearn.frozen import FrozenEstimator
        calibrator = CalibratedClassifierCV(FrozenEstimator(best_xgb), method="isotonic")
    except ImportError:
        calibrator = CalibratedClassifierCV(best_xgb, method="isotonic", cv="prefit")

    calibrator.fit(X[va], y[va])
    p_cal_te = calibrator.predict_proba(X[te])[:, 1]
    cal_metrics = metrics(y[te], p_cal_te)
    probs["Proposed (XGBoost Calibrated)"] = p_cal_te
    rows.append({"Model": "Proposed (XGBoost Calibrated)", **cal_metrics})

    print(
        f"  {'Calibrated XGBoost':<22} test AUC: {cal_metrics['AUC']:.3f} | "
        f"Recall: {cal_metrics['Recall']:.3f} | Brier: {cal_metrics['Brier']:.3f} (Isotonic)"
    )

    res = pd.DataFrame(rows).round(3)
    res.to_csv(OUT / "model_comparison.csv", index=False)
    print("\nTest Set Comparison Table:\n", res.to_string(index=False))

    # Save models
    joblib.dump(best_xgb, MODEL_PATH)
    joblib.dump(calibrator, CALIBRATED_MODEL_PATH)

    # 5. Predictions for all units using Calibrated probabilities
    df["probability"] = calibrator.predict_proba(X)[:, 1]
    df["uncalibrated_probability"] = best_xgb.predict_proba(X)[:, 1]
    df["risk_class"] = risk_class(df["probability"])
    df["split"] = np.array(["train", "val", "test"])[split]
    df.to_csv(OUT / "predictions.csv", index=False)

    # 6. Global SHAP Explainability Plot
    print("\nComputing TreeSHAP values for global explainability...")
    prep = best_xgb.named_steps["prep"]
    booster = best_xgb.named_steps["model"]
    X_te_trans = prep.transform(X[te])
    raw_feature_names = prep.get_feature_names_out()
    feature_names = [n.split("__", 1)[-1] for n in raw_feature_names]

    explainer = shap.TreeExplainer(booster)
    shap_values = explainer(X_te_trans)

    # SHAP Summary Plot
    plt.figure(figsize=(9, 6.5))
    shap.summary_plot(shap_values.values, X_te_trans, feature_names=feature_names, show=False)
    plt.title("Global TreeSHAP Summary (Test Set Slope Units)", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Save SHAP feature importances
    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    shap_imp = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)
    shap_imp.to_csv(OUT / "shap_importance.csv", header=["mean_abs_shap"])

    # 7. Model Diagnostic Plots (ROC, Calibration, Confusion Matrix, Feature Importance)
    cmap = plt.get_cmap("tab10")
    ax = res.set_index("Model")[["Accuracy", "F1", "AUC"]].plot.bar(figsize=(9, 4.5), rot=15, ylim=(0.5, 1))
    ax.set_title("Model comparison (test set)")
    plt.tight_layout()
    plt.savefig(OUT / "model_comparison.png", dpi=130)
    plt.close()

    # Calibration Curve (Reliability Diagram)
    plt.figure(figsize=(5.5, 5))
    prob_true_raw, prob_pred_raw = calibration_curve(y[te], probs["Proposed (XGBoost)"], n_bins=10)
    prob_true_cal, prob_pred_cal = calibration_curve(y[te], probs["Proposed (XGBoost Calibrated)"], n_bins=10)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")
    plt.plot(prob_pred_raw, prob_true_raw, "s-", color="#c0392b", label=f"Uncalibrated (Brier={rows[-2]['Brier']:.3f})")
    plt.plot(prob_pred_cal, prob_true_cal, "o-", color="#2e8b57", label=f"Isotonic Calibrated (Brier={rows[-1]['Brier']:.3f})")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Observed Landslide Fraction")
    plt.title("Probability Calibration Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUT / "calibration_curve.png", dpi=130)
    plt.close()

    # ROC Curves
    plt.figure(figsize=(6, 5.2))
    for i, (n, p) in enumerate(probs.items()):
        fpr, tpr, _ = roc_curve(y[te], p)
        lw = 2.5 if "Calibrated" in n else (2.0 if "Proposed" in n else 1.2)
        plt.plot(fpr, tpr, color=cmap(i), lw=lw, label=f"{n} (AUC {roc_auc_score(y[te], p):.3f})")
    plt.plot([0, 1], [0, 1], "k:")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.legend(loc="lower right", fontsize=8)
    plt.title("ROC curves (Held-out Spatial Test Set)")
    plt.tight_layout()
    plt.savefig(OUT / "roc_curve.png", dpi=130)
    plt.close()

    # Confusion matrix
    cm = confusion_matrix(y[te], (probs["Proposed (XGBoost Calibrated)"] >= 0.5).astype(int))
    plt.figure(figsize=(4, 3.6))
    plt.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, v, ha="center", va="center", fontsize=14, color="white" if v > cm.max() / 2 else "black")
    plt.xticks([0, 1], ["Safe", "Landslide"])
    plt.yticks([0, 1], ["Safe", "Landslide"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix (Calibrated XGBoost)")
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.png", dpi=130)
    plt.close()

    # Traditional Feature Importance
    names = best_xgb.named_steps["prep"].get_feature_names_out()
    imp = pd.Series(best_xgb.named_steps["model"].feature_importances_, index=names)
    agg: Dict[str, float] = {}
    for n, v in imp.items():
        base = n.split("__", 1)[1]
        for c in CAT:
            if base.startswith(c + "_"):
                base = c
        agg[base] = agg.get(base, 0.0) + float(v)
    agg_s = pd.Series(agg).sort_values()
    agg_s.plot.barh(figsize=(6, 4), color="#c0392b")
    plt.title("Feature importance (XGBoost Gain)")
    plt.tight_layout()
    plt.savefig(OUT / "feature_importance.png", dpi=130)
    plt.close()
    agg_s.sort_values(ascending=False).round(4).to_csv(OUT / "feature_importance.csv", header=["importance"])

    # 8. Interactive Map Export (map.html) & Static Map
    print("\nGenerating interactive map (outputs/map.html)...")
    create_interactive_map(df, output_path=OUT / "map.html", sample_size=3000)

    from analyze import draw_map
    fig = draw_map(df, title="Landslide susceptibility map (5 risk classes)")
    fig.savefig(OUT / "susceptibility_map.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 9. Factor-group sensitivity
    p = probs["Proposed (XGBoost)"]
    thr = np.arange(0.2, 0.81, 0.1)
    pr = [precision_score(y[te], p >= t, zero_division=0) for t in thr]
    rc = [recall_score(y[te], p >= t, zero_division=0) for t in thr]
    groups_drop = {
        "None (all factors)": [],
        "Slope + terrain": ["slope", "elevation", "aspect", "curvature"],
        "Rainfall": ["rainfall"],
        "Land cover": ["land_cover"],
        "River + road": ["river_dist", "road_dist"],
        "Geology (lithology + fault)": ["lithology", "fault_dist"],
    }
    sens = {}
    for g, drop in groups_drop.items():
        num_sub = [c for c in NUM if c not in drop]
        cat_sub = [c for c in CAT if c not in drop]
        m = Pipeline([("prep", make_preprocessor(num_sub, cat_sub)), ("model", xgb(**best_params))]).fit(
            X[tr][num_sub + cat_sub], y[tr]
        )
        sens[g] = roc_auc_score(y[te], m.predict_proba(X[te][num_sub + cat_sub])[:, 1])

    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    axs[0].plot(thr, pr, "o-", label="Precision")
    axs[0].plot(thr, rc, "s-", label="Recall")
    axs[0].set_xlabel("Decision threshold")
    axs[0].legend()
    axs[0].set_title("Threshold sensitivity")
    pd.Series(sens).iloc[::-1].plot.barh(ax=axs[1], color="#2c7fb8", xlim=(0.5, 1))
    axs[1].set_title("AUC when a factor group is removed")
    plt.tight_layout()
    plt.savefig(OUT / "sensitivity.png", dpi=130)
    plt.close()
    pd.Series(sens, name="AUC").round(3).to_csv(OUT / "sensitivity_factor_groups.csv")

    print(f"\nDone! Outputs generated in {OUT}:\n  - {MODEL_PATH}\n  - {CALIBRATED_MODEL_PATH}\n  - {OUT / 'map.html'}\n  - {OUT / 'shap_summary.png'}\n  - {OUT / 'calibration_curve.png'}")


if __name__ == "__main__":
    main()
