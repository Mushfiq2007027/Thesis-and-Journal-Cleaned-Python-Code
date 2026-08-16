from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    csv_path: str = "tabular_data.csv"  # BEHSOF clinical records, 113 rows
    output_dir: str = "outputs_fibrosis_tabular"
    seed: int = 42
    n_folds: int = 5
    inner_val_size: float = 0.20  # inner validation share of each outer-train split
    bootstrap_iterations: int = 1000  
    shap_patient_id: str = "BEH01121"  
    shap_top_n: int = 10
    dpi: int = 300


# Task constants 
TASK_NAME = "Fibrosis"
LABEL_COL = "fibrosis_binary"
STAGE_COL = "Fibroscan F"
CLASS_NAMES = {0: "F0", 1: "F1-F2"}
DERIVED_FEATURES = ["AST_ALT_ratio", "FIB4", "APRI", "GPR", "Forns_Index"]

# Conventional upper-limit-of-normal constants applied uniformly across the
# cohort: patient- or sex-specific reference ranges
# were not provided in the BEHSOF metadata.
ULN_AST = 40.0  # U/L, used in APRI (Eq. 7)
ULN_GGT = 40.0  # U/L, used in GPR (Eq. 8)

# Columns excluded from the predictor set to prevent target leakage
# patient identifier, steatosis stage, Fibroscan S,
# Fibroscan F, CAP score, E score. Label columns are excluded as well.
LEAKAGE_COLS = [
    "Patient ID",
    "patient_id",
    "Steatosis stage",
    "Fibroscan S",
    "Fibroscan F",
    "CAP score",
    "E score",
    "steatosis_binary",
    "fibrosis_binary",
]

# Metrics
METRIC_NAMES = [
    "accuracy",
    "balanced_accuracy",
    "f1",
    "auc_roc",
    "auc_pr",
    "precision",
    "recall",
    "specificity",
]

SEX_MAP = {"M": 0, "F": 1, "m": 0, "f": 1,
           "Male": 0, "Female": 1, "male": 0, "female": 1}


def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


# ---------------------------------------------------------------------------
# 1. Cohort preparation 
# ---------------------------------------------------------------------------


def load_cohort(cfg: Config) -> pd.DataFrame:
    """Load BEHSOF clinical records and build the binary fibrosis target."""
    df = pd.read_csv(cfg.csv_path)
    df.columns = df.columns.str.strip()

    # Drop a trailing fully-empty row, if present.
    if df.iloc[-1].isna().all():
        df = df.iloc[:-1].copy()

    # Standardized alphanumeric patient identifier.
    df["patient_id"] = (
        df["Patient ID"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.replace(" ", "", regex=False)
        .str.strip()
    )

    # Binary encoding of sex .
    df["sex"] = df["sex"].astype(str).str.strip().map(SEX_MAP)
    assert df["sex"].notna().all(), "Unmapped sex value found."

    # Binary target: F0 -> 0 ("absent"), F1-F2 -> 1 ("mild-moderate").
    df[STAGE_COL] = pd.to_numeric(df[STAGE_COL], errors="coerce")
    assert df[STAGE_COL].notna().all(), "Invalid Fibroscan F values."
    df[LABEL_COL] = (df[STAGE_COL].astype(int) >= 1).astype(int)

    assert len(df) == 113, f"Expected 113 patients, got {len(df)}."
    assert df["patient_id"].nunique() == len(df), "Patient IDs must be unique."

    counts = df[LABEL_COL].value_counts().sort_index()
    print(f"Patients: {len(df)} | Class 0 (F0): {counts[0]}, "
          f"Class 1 (F1-F2): {counts[1]}")
    return df


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric non-leakage raw predictors: 62 variables - 6 leakage = 56."""
    cols = [
        c for c in df.columns
        if c not in LEAKAGE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    assert len(cols) == 56, f"Expected 56 raw predictors, got {len(cols)}."
    return cols


# ---------------------------------------------------------------------------
# 2. Engineered fibrosis indices 
# ---------------------------------------------------------------------------


def add_fibrosis_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Append the five fibrosis-specific derived features.

    Applied to median-imputed values inside each fold, matching the paper's
    pipeline order (imputation -> feature engineering -> scaling). No
    zero/negative flooring: the paper states platelets, GGT and age were
    strictly positive for every patient, so positivity is asserted instead.
    """
    df = df.copy()
    age = df["Age"].astype(float)              # years
    ast = df["AST (SGOT)"].astype(float)       # U/L
    alt = df["ALT (SGPT)"].astype(float)       # U/L
    platelets = df["Platelets"].astype(float)  # x10^9/L
    ggt = df["GGT"].astype(float)              # U/L
    cholesterol = df["Cholestrol"].astype(float)  # mg/dL (dataset spelling)

    for name, series in {"Age": age, "AST": ast, "ALT": alt,
                         "Platelets": platelets, "GGT": ggt}.items():
        assert (series > 0).all(), (
            f"{name} must be strictly positive (paper: no flooring required)."
        )

    # Eq. (1): AST/ALT ratio (shared with the steatosis feature set).
    df["AST_ALT_ratio"] = ast / alt

    # Eq. (6): FIB-4 = Age * AST / (Platelets * sqrt(ALT)).
    df["FIB4"] = (age * ast) / (platelets * np.sqrt(alt))

    # Eq. (7): APRI = ((AST / ULN_AST) / Platelets) * 100, ULN_AST = 40 U/L.
    df["APRI"] = ((ast / ULN_AST) / platelets) * 100.0

    # Eq. (8): GPR = ((GGT / 40) / Platelets) * 100.
    df["GPR"] = ((ggt / ULN_GGT) / platelets) * 100.0

    # Eq. (9): Forns = 7.811 - 3.131 ln(Platelets) + 0.781 ln(GGT)
    #                  + 3.467 ln(Age) - 0.014 Cholesterol.
    df["Forns_Index"] = (
        7.811
        - 3.131 * np.log(platelets)
        + 0.781 * np.log(ggt)
        + 3.467 * np.log(age)
        - 0.014 * cholesterol
    )
    return df


# ---------------------------------------------------------------------------
# 3. Leakage-safe per-fold preprocessing 
# ---------------------------------------------------------------------------


def fit_imputer(train_df: pd.DataFrame, raw_cols: list[str]) -> SimpleImputer:
    """Median imputation fitted on the training partition only."""
    return SimpleImputer(strategy="median").fit(train_df[raw_cols])


def transform_features(
    df: pd.DataFrame, raw_cols: list[str], imputer: SimpleImputer
) -> pd.DataFrame:
    """Impute, then append engineered indices -> 61-dimensional frame."""
    X = pd.DataFrame(
        imputer.transform(df[raw_cols]), columns=raw_cols, index=df.index
    )
    X = add_fibrosis_indices(X)
    feature_cols = raw_cols + DERIVED_FEATURES
    assert X.shape[1] == 61, f"Expected 61 features, got {X.shape[1]}."
    return X[feature_cols]


def make_model(seed: int) -> LogisticRegression:
    """Tabular Logistic Regression classifier."""
    return LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        class_weight="balanced",
        max_iter=50000,
        tol=1e-8,
        random_state=seed,
    )


# ---------------------------------------------------------------------------
# 4. Metrics and threshold selection 
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else np.nan,
    }


def select_threshold(y_true, y_prob) -> float:
    """Threshold maximizing balanced accuracy on the inner validation split.

    The chosen threshold is fixed and applied unchanged to the outer test
    fold, which is never used for selection.
    """
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [
        balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# 5. Outer 5-fold cross-validation (patient-level, stratified)
# ---------------------------------------------------------------------------


def run_cross_validation(df: pd.DataFrame, cfg: Config):
    raw_cols = raw_feature_columns(df)
    indexed = df.set_index("patient_id")

    # Deterministic patient ordering: with a shared seed, all configuration
    # scripts (tabular / image / multimodal) produce matched outer folds
    patients = (
        df[["patient_id", LABEL_COL]]
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )

    fold_rows, oof_rows, fold_assignments = [], [], {}

    for fold, (tr_idx, te_idx) in enumerate(
        skf.split(patients["patient_id"], patients[LABEL_COL]), start=1
    ):
        train_pids = patients["patient_id"].iloc[tr_idx].tolist()
        test_pids = patients["patient_id"].iloc[te_idx].tolist()
        fold_assignments[fold] = {"train": train_pids, "test": test_pids}

        train_df = indexed.loc[train_pids]
        test_df = indexed.loc[test_pids]
        y_train = train_df[LABEL_COL].to_numpy()
        y_test = test_df[LABEL_COL].to_numpy()

        # --- Inner split: threshold selection only ---
        inner_train, inner_val = train_test_split(
            train_df,
            test_size=cfg.inner_val_size,
            stratify=train_df[LABEL_COL],
            random_state=cfg.seed + fold,
        )
        inner_imputer = fit_imputer(inner_train, raw_cols)
        X_inner = transform_features(inner_train, raw_cols, inner_imputer)
        X_val = transform_features(inner_val, raw_cols, inner_imputer)
        inner_scaler = StandardScaler().fit(X_inner)
        inner_model = make_model(cfg.seed + fold)
        inner_model.fit(inner_scaler.transform(X_inner),
                        inner_train[LABEL_COL].to_numpy())
        val_prob = inner_model.predict_proba(inner_scaler.transform(X_val))[:, 1]
        threshold = select_threshold(inner_val[LABEL_COL].to_numpy(), val_prob)

        # --- Final fold model: preprocessing refit on the full outer-train ---
        imputer = fit_imputer(train_df, raw_cols)
        X_train = transform_features(train_df, raw_cols, imputer)
        X_test = transform_features(test_df, raw_cols, imputer)
        scaler = StandardScaler().fit(X_train)

        model = make_model(cfg.seed + fold)
        model.fit(scaler.transform(X_train), y_train)

        test_prob = model.predict_proba(scaler.transform(X_test))[:, 1]
        test_pred = (test_prob >= threshold).astype(int)

        fold_metrics = compute_metrics(y_test, test_pred, test_prob)
        fold_metrics.update({"fold": fold, "threshold": threshold})
        fold_rows.append(fold_metrics)

        for pid, yt, yp, ypr in zip(test_pids, y_test, test_pred, test_prob):
            oof_rows.append({
                "patient_id": pid,
                "fold": fold,
                "true_label": int(yt),
                "predicted_probability": float(ypr),
                "predicted_label": int(yp),
                "threshold": threshold,
            })

        print(f"Fold {fold}: threshold={threshold:.2f}, "
              f"AUC-ROC={fold_metrics['auc_roc']:.3f}, "
              f"BalAcc={fold_metrics['balanced_accuracy']:.3f}, "
              f"F1={fold_metrics['f1']:.3f}")

    fold_df = pd.DataFrame(fold_rows)
    oof_df = pd.DataFrame(oof_rows).sort_values("patient_id").reset_index(drop=True)

    # Pooled OOF integrity: exactly one prediction per patient (n = 113).
    assert len(oof_df) == 113 and oof_df["patient_id"].nunique() == 113
    oof_df["outcome_type"] = np.select(
        [
            (oof_df["true_label"] == 1) & (oof_df["predicted_label"] == 1),
            (oof_df["true_label"] == 0) & (oof_df["predicted_label"] == 0),
            (oof_df["true_label"] == 0) & (oof_df["predicted_label"] == 1),
            (oof_df["true_label"] == 1) & (oof_df["predicted_label"] == 0),
        ],
        ["TP", "TN", "FP", "FN"],
    )
    return fold_df, oof_df, fold_assignments


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
    """Fold-wise mean +/- SD for the eight paper metrics (Section V.B)."""
    rows = []
    for metric in METRIC_NAMES:
        rows.append({
            "metric": metric,
            "mean": fold_df[metric].mean(),
            "std": fold_df[metric].std(ddof=1),
        })
        print(f"{metric:<20} {rows[-1]['mean']:.3f} +/- {rows[-1]['std']:.3f}")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 6. Pooled OOF visualizations (confusion matrix, ROC, PR)
# ---------------------------------------------------------------------------


def plot_confusion_matrix(oof_df: pd.DataFrame, cfg: Config) -> None:
    y_true = oof_df["true_label"].to_numpy()
    y_pred = oof_df["predicted_label"].to_numpy()
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    assert cm.sum() == 113

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=cfg.dpi)
    image = ax.imshow(cm, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[row, col] > cm.max() / 2 else "black")
    labels = [CLASS_NAMES[0], CLASS_NAMES[1]]
    ax.set_xticks([0, 1], [f"Predicted {labels[0]}", f"Predicted {labels[1]}"])
    ax.set_yticks([0, 1], [f"True {labels[0]}", f"True {labels[1]}"])
    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")
    ax.set_title(f"Pooled OOF Confusion Matrix — Tabular LR ({TASK_NAME})\n"
                 "113 unique patients across five outer folds",
                 fontweight="bold", pad=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Patient count", fontweight="bold")
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "confusion_matrix_pooled_oof.png",
                dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)


def plot_roc_curve(oof_df: pd.DataFrame, cfg: Config) -> None:
    y_true = oof_df["true_label"].to_numpy()
    y_prob = oof_df["predicted_probability"].to_numpy()
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    auc_roc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=cfg.dpi)
    ax.plot(fpr, tpr, color="navy", linewidth=2.5,
            label=f"Pooled OOF ROC (AUC = {auc_roc:.3f})")
    ax.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2,
            label="No-discrimination reference")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate / sensitivity", fontweight="bold")
    ax.set_title(f"ROC Curve — Tabular LR ({TASK_NAME})\n"
                 "Pooled out-of-fold predictions, n=113",
                 fontweight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "roc_curve_pooled_oof.png",
                dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"false_positive_rate": fpr, "true_positive_rate": tpr,
                  "threshold": thresholds}).to_csv(
        Path(cfg.output_dir) / "roc_curve_pooled_oof.csv", index=False
    )


def plot_pr_curve(oof_df: pd.DataFrame, cfg: Config) -> None:
    y_true = oof_df["true_label"].to_numpy()
    y_prob = oof_df["predicted_probability"].to_numpy()
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    auc_pr = average_precision_score(y_true, y_prob)
    prevalence = y_true.mean()

    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=cfg.dpi)
    ax.plot(recall, precision, color="darkred", linewidth=2.5,
            label=f"Pooled OOF PR (AP = {auc_pr:.3f})")
    ax.axhline(prevalence, color="gray", linestyle="--", linewidth=1.2,
               label=f"Positive-class prevalence = {prevalence:.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall / sensitivity", fontweight="bold")
    ax.set_ylabel("Precision / positive predictive value", fontweight="bold")
    ax.set_title(f"Precision-Recall Curve — Tabular LR ({TASK_NAME})\n"
                 "Pooled out-of-fold predictions, n=113",
                 fontweight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(Path(cfg.output_dir) / "pr_curve_pooled_oof.png",
                dpi=cfg.dpi, bbox_inches="tight")
    plt.close(fig)

    pd.DataFrame({"recall": recall[:-1], "precision": precision[:-1],
                  "threshold": thresholds}).to_csv(
        Path(cfg.output_dir) / "pr_curve_pooled_oof.csv", index=False
    )


# ---------------------------------------------------------------------------
# 7. Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def stratified_bootstrap_indices(y_true: np.ndarray, rng) -> np.ndarray:
    """Patient-level stratified resampling: preserves the class distribution."""
    indices = []
    for cls in (0, 1):
        cls_idx = np.where(y_true == cls)[0]
        indices.append(rng.choice(cls_idx, size=len(cls_idx), replace=True))
    bootstrap_idx = np.concatenate(indices)
    rng.shuffle(bootstrap_idx)
    return bootstrap_idx


def bootstrap_confidence_intervals(
    oof_df: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """95% percentile bootstrap CIs from 1,000 stratified patient-level
    resamples of the pooled OOF predictions"""
    y_true = oof_df["true_label"].to_numpy()
    y_pred = oof_df["predicted_label"].to_numpy()
    y_prob = oof_df["predicted_probability"].to_numpy()

    observed = compute_metrics(y_true, y_pred, y_prob)
    rng = np.random.default_rng(cfg.seed)
    bootstrap_values = {m: [] for m in METRIC_NAMES}

    for _ in range(cfg.bootstrap_iterations):
        idx = stratified_bootstrap_indices(y_true, rng)
        resampled = compute_metrics(y_true[idx], y_pred[idx], y_prob[idx])
        for metric, value in resampled.items():
            if np.isfinite(value):
                bootstrap_values[metric].append(value)

    rows = []
    for metric in METRIC_NAMES:
        values = np.asarray(bootstrap_values[metric], dtype=float)
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append({
            "metric": metric,
            "pooled_oof_estimate": observed[metric],
            "ci95_lower": lower,
            "ci95_upper": upper,
            "n_valid_resamples": len(values),
        })
        print(f"{metric:<20} {observed[metric]:.3f} "
              f"(95% CI {lower:.3f}-{upper:.3f})")
    return pd.DataFrame(rows)


# ----------------------------------------------
# 8. Final full-data model, inference, and SHAP 
# ----------------------------------------------


def fit_final_model(df: pd.DataFrame, cfg: Config) -> dict:
    """Refit the pipeline on all 113 patients for deployment and SHAP.

    Used only for inference on new patients and for post hoc explanation;
    all performance estimates come from the leakage-safe OOF evaluation.
    The deployment threshold is selected with the same inner-validation rule.
    """
    raw_cols = raw_feature_columns(df)
    imputer = fit_imputer(df, raw_cols)
    X_full = transform_features(df, raw_cols, imputer)
    scaler = StandardScaler().fit(X_full)
    y_full = df[LABEL_COL].to_numpy()

    inner_train, inner_val = train_test_split(
        df, test_size=cfg.inner_val_size, stratify=df[LABEL_COL],
        random_state=cfg.seed,
    )
    inner_imputer = fit_imputer(inner_train, raw_cols)
    X_inner = transform_features(inner_train, raw_cols, inner_imputer)
    X_val = transform_features(inner_val, raw_cols, inner_imputer)
    inner_scaler = StandardScaler().fit(X_inner)
    inner_model = make_model(cfg.seed)
    inner_model.fit(inner_scaler.transform(X_inner),
                    inner_train[LABEL_COL].to_numpy())
    val_prob = inner_model.predict_proba(inner_scaler.transform(X_val))[:, 1]
    threshold = select_threshold(inner_val[LABEL_COL].to_numpy(), val_prob)

    model = make_model(cfg.seed)
    model.fit(scaler.transform(X_full), y_full)

    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "threshold": threshold,
        "raw_cols": raw_cols,
        "feature_cols": raw_cols + DERIVED_FEATURES,
    }


def transform_new_patients(bundle: dict, new_df: pd.DataFrame) -> np.ndarray:
    """Preprocess raw BEHSOF-style records with the fitted final pipeline."""
    X = transform_features(new_df, bundle["raw_cols"], bundle["imputer"])
    return bundle["scaler"].transform(X)


def predict_new_patients(bundle: dict, new_df: pd.DataFrame) -> pd.DataFrame:
    """Inference on new patients: returns probabilities and binary predictions
    at the deployment threshold selected on the inner validation split."""
    X_scaled = transform_new_patients(bundle, new_df)
    prob = bundle["model"].predict_proba(X_scaled)[:, 1]
    return pd.DataFrame({
        "predicted_probability": prob,
        "predicted_label": (prob >= bundle["threshold"]).astype(int),
        "threshold": bundle["threshold"],
    })


def run_shap(
    bundle: dict, df: pd.DataFrame, oof_df: pd.DataFrame, cfg: Config
) -> None:
    """Global SHAP summary and single-patient waterfall

    SHAP values are computed from the final full-data model; the patient's
    reported prediction and TP/TN/FP/FN status are the leakage-safe OOF
    values from the fold model that never saw this patient in training.
    """
    raw_cols = bundle["raw_cols"]
    X_full = transform_features(df.set_index("patient_id"), raw_cols,
                                bundle["imputer"])
    X_scaled = pd.DataFrame(
        bundle["scaler"].transform(X_full),
        columns=bundle["feature_cols"],
        index=X_full.index,
    )

    explainer = shap.LinearExplainer(bundle["model"], X_scaled)
    shap_values = explainer.shap_values(X_scaled)
    if isinstance(shap_values, list):
        shap_values = np.asarray(shap_values[-1])
    shap_values = np.asarray(shap_values)
    assert shap_values.shape == X_scaled.shape

    # --- Global importance ---
    global_df = pd.DataFrame({
        "feature": bundle["feature_cols"],
        "mean_abs_shap_log_odds": np.mean(np.abs(shap_values), axis=0),
        "lr_coefficient_log_odds_per_sd": bundle["model"].coef_.ravel(),
    }).sort_values("mean_abs_shap_log_odds", ascending=False)
    global_df.to_csv(Path(cfg.output_dir) / "shap_global_importance.csv",
                     index=False)

    plt.figure(figsize=(12, max(10, len(bundle["feature_cols"]) * 0.28)),
               dpi=cfg.dpi)
    shap.summary_plot(shap_values, X_scaled,
                      max_display=len(bundle["feature_cols"]), show=False)
    plt.title(f"Global SHAP Summary — Tabular LR ({TASK_NAME})",
              fontweight="bold", pad=14)
    plt.tight_layout()
    plt.savefig(Path(cfg.output_dir) / "shap_global_summary.png",
                dpi=cfg.dpi, bbox_inches="tight")
    plt.close("all")

    # --- Single-patient waterfall (default: BEH01121) ---
    patient_id = str(cfg.shap_patient_id).strip()
    assert patient_id in X_scaled.index, f"Patient not found: {patient_id}"
    pos = X_scaled.index.get_loc(patient_id)

    base_value = getattr(explainer, "expected_value",
                         bundle["model"].intercept_[0])
    base_value = float(np.asarray(base_value).reshape(-1)[-1])

    explanation = shap.Explanation(
        values=shap_values[pos],
        base_values=base_value,
        data=X_scaled.iloc[pos].to_numpy(),
        feature_names=bundle["feature_cols"],
    )

    patient_oof = oof_df[oof_df["patient_id"] == patient_id].iloc[0]
    true_text = CLASS_NAMES[int(patient_oof["true_label"])]
    pred_text = CLASS_NAMES[int(patient_oof["predicted_label"])]

    patient_df = pd.DataFrame({
        "patient_id": patient_id,
        "feature": bundle["feature_cols"],
        "scaled_feature_value": X_scaled.iloc[pos].to_numpy(),
        "shap_log_odds": shap_values[pos],
        "abs_shap_log_odds": np.abs(shap_values[pos]),
    }).sort_values("abs_shap_log_odds", ascending=False)
    patient_df.to_csv(
        Path(cfg.output_dir) / f"shap_waterfall_{patient_id}.csv", index=False
    )

    plt.figure(figsize=(13, 7.5), dpi=cfg.dpi)
    shap.plots.waterfall(explanation, max_display=cfg.shap_top_n + 1,
                         show=False)
    fig = plt.gcf()
    plt.title(
        f"Patient-Specific SHAP Waterfall — {patient_id} ({TASK_NAME})\n"
        f"True: {true_text} | OOF prediction: {pred_text} | "
        f"P({CLASS_NAMES[1]}): {patient_oof['predicted_probability']:.3f} | "
        f"{patient_oof['outcome_type']}",
        fontsize=13, fontweight="bold", pad=14,
    )

    # Color key: bar color encodes the patient's standardized feature value.
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    try:
        from shap.plots import colors as shap_colors
        cmap = shap_colors.red_blue
    except Exception:
        cmap = plt.cm.RdBu_r
    fig.subplots_adjust(right=0.84, top=0.82)
    cax = fig.add_axes([0.87, 0.18, 0.018, 0.60])
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Low", "High"])
    cbar.set_label("Feature value\n(standardized)", fontweight="bold")

    fig.savefig(Path(cfg.output_dir) / f"shap_waterfall_{patient_id}.png",
                dpi=cfg.dpi, bbox_inches="tight")
    plt.close("all")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Tabular LR baseline — fibrosis (F0 vs F1-F2)."
    )
    parser.add_argument("--csv-path", default=Config.csv_path)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--bootstrap-iterations", type=int,
                        default=Config.bootstrap_iterations)
    parser.add_argument("--shap-patient-id", default=Config.shap_patient_id)
    args = parser.parse_args()
    return Config(
        csv_path=args.csv_path,
        output_dir=args.output_dir,
        seed=args.seed,
        bootstrap_iterations=args.bootstrap_iterations,
        shap_patient_id=args.shap_patient_id,
    )


def main() -> None:
    cfg = parse_args()
    set_seeds(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 80)
    print(f"TABULAR LOGISTIC REGRESSION — {TASK_NAME.upper()} "
          f"({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})")
    print("=" * 80)

    df = load_cohort(cfg)

    print("\n[1/5] Patient-level stratified 5-fold cross-validation")
    fold_df, oof_df, fold_assignments = run_cross_validation(df, cfg)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    with open(out_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    print("\n[2/5] Fold-wise mean +/- SD (eight paper metrics)")
    summary_df = summarize_folds(fold_df)
    summary_df.to_csv(out_dir / "fold_metrics_summary.csv", index=False)

    print("\n[3/5] Pooled OOF confusion matrix, ROC curve, PR curve")
    plot_confusion_matrix(oof_df, cfg)
    plot_roc_curve(oof_df, cfg)
    plot_pr_curve(oof_df, cfg)

    print("\n[4/5] Bootstrap 95% CIs "
          f"({cfg.bootstrap_iterations} stratified patient-level resamples)")
    ci_df = bootstrap_confidence_intervals(oof_df, cfg)
    ci_df.to_csv(out_dir / "bootstrap_95ci.csv", index=False)

    print("\n[5/5] Final full-data model + SHAP "
          f"(case-study patient {cfg.shap_patient_id})")
    bundle = fit_final_model(df, cfg)
    run_shap(bundle, df, oof_df, cfg)

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
