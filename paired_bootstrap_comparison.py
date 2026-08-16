from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    steatosis_tabular: str = "outputs_steatosis_tabular/oof_predictions.csv"
    steatosis_image: str = "outputs_steatosis_image/oof_predictions.csv"
    steatosis_multimodal: str = "outputs_steatosis_multimodal/oof_predictions.csv"
    fibrosis_tabular: str = "outputs_fibrosis_tabular/oof_predictions.csv"
    fibrosis_image: str = "outputs_fibrosis_image/oof_predictions.csv"
    fibrosis_multimodal: str = "outputs_fibrosis_multimodal/oof_predictions.csv"
    output_dir: str = "outputs_statistical_testing"
    n_bootstrap: int = 2000
    seed: int = 42
    alpha: float = 0.05


TASKS = ["steatosis", "fibrosis"]
CONFIGS = ["tabular", "image", "multimodal"]

# Multimodal is always the first member of each compared pair, so a positive
# delta means the multimodal model scores higher.
COMPARISONS = [("multimodal", "image"), ("multimodal", "tabular")]

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
PROB_METRICS = {"auc_roc", "auc_pr"}  # computed from probabilities, not labels

# The single prespecified primary comparison per task.
PRIMARY_METRIC = "auc_roc"
PRIMARY_COMPARISON = ("multimodal", "image")


# ---------------------------------------------------------------------------
# OOF loading and validation
# ---------------------------------------------------------------------------


def load_oof(path: str, name: str) -> pd.DataFrame:
    """Load one configuration's pooled OOF predictions and validate them."""
    df = pd.read_csv(path)
    required = {"patient_id", "fold", "true_label",
                "predicted_probability", "predicted_label"}
    missing = required - set(df.columns)
    assert not missing, f"{name}: missing columns {missing} in {path}"

    df["patient_id"] = df["patient_id"].astype(str)
    assert len(df) == 113, f"{name}: expected 113 rows, got {len(df)}"
    assert df["patient_id"].nunique() == 113, (
        f"{name}: each patient must appear exactly once in the OOF table."
    )
    return df


def align_pair(oof_a: pd.DataFrame, oof_b: pd.DataFrame,
               name_a: str, name_b: str):
    """Align two OOF tables on patient ID and verify they describe the same
    cohort with identical ground-truth labels."""
    a = oof_a.set_index("patient_id")
    b = oof_b.set_index("patient_id")
    assert set(a.index) == set(b.index), (
        f"Patient sets differ between {name_a} and {name_b}."
    )
    b = b.loc[a.index]
    assert (a["true_label"].to_numpy() == b["true_label"].to_numpy()).all(), (
        f"Ground-truth labels disagree between {name_a} and {name_b}."
    )
    return a, b


def check_matched_folds(oof_map: dict, task: str) -> None:
    """Informational check: the three configurations of a task should assign
    each patient to the same outer fold (fold-matched comparison)."""
    folds = {
        name: oof_map[name].set_index("patient_id")["fold"]
        for name in CONFIGS
    }
    matched = all(
        (folds[CONFIGS[0]].sort_index().to_numpy()
         == folds[name].sort_index().to_numpy()).all()
        for name in CONFIGS[1:]
    )
    note = "IDENTICAL" if matched else "DIFFERENT — check seeds/ordering"
    print(f"  Fold assignments across the three {task} configurations: {note}")


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def metric_value(metric: str, y_true, y_pred, y_prob) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    if metric == "accuracy":
        return accuracy_score(y_true, y_pred)
    if metric == "balanced_accuracy":
        return balanced_accuracy_score(y_true, y_pred)
    if metric == "f1":
        return f1_score(y_true, y_pred, zero_division=0)
    if metric == "auc_roc":
        return roc_auc_score(y_true, y_prob)
    if metric == "auc_pr":
        return average_precision_score(y_true, y_prob)
    if metric == "precision":
        return precision_score(y_true, y_pred, zero_division=0)
    if metric == "recall":
        return recall_score(y_true, y_pred, zero_division=0)
    if metric == "specificity":
        return tn / (tn + fp) if (tn + fp) > 0 else np.nan
    raise ValueError(f"Unknown metric: {metric}")


# ---------------------------------------------------------------------------
# Paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(oof_a: pd.DataFrame, oof_b: pd.DataFrame, metric: str,
                     n_bootstrap: int, seed: int) -> dict:
    """Paired bootstrap on the pooled OOF predictions of two configurations.

    Each iteration draws one set of patient indices with replacement and
    applies it to both models simultaneously, so the metric difference is
    computed on identical resampled cohorts. The p-value is two-sided:
    twice the smaller tail proportion of the delta distribution.
    """
    a, b = align_pair(oof_a, oof_b, "model A", "model B")
    y_true = a["true_label"].to_numpy()
    prob_a = a["predicted_probability"].to_numpy()
    prob_b = b["predicted_probability"].to_numpy()
    pred_a = a["predicted_label"].to_numpy()
    pred_b = b["predicted_label"].to_numpy()

    uses_prob = metric in PROB_METRICS
    scores_a = prob_a if uses_prob else pred_a
    scores_b = prob_b if uses_prob else pred_b

    point_delta = metric_value(metric, y_true, pred_a, prob_a) - \
        metric_value(metric, y_true, pred_b, prob_b)

    rng = np.random.default_rng(seed)
    deltas = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        yt = y_true[idx]
        # Probability-based metrics are undefined for single-class resamples.
        if uses_prob and len(np.unique(yt)) < 2:
            continue
        delta = metric_value(metric, yt, pred_a[idx], prob_a[idx]) - \
            metric_value(metric, yt, pred_b[idx], prob_b[idx])
        if np.isfinite(delta):
            deltas.append(delta)

    deltas = np.asarray(deltas, dtype=float)
    ci_lo, ci_hi = np.percentile(deltas, [2.5, 97.5])
    p_value = 2.0 * min((deltas > 0).mean(), (deltas < 0).mean())
    p_value = min(p_value, 1.0)

    return {
        "point_delta": float(point_delta),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "p_value": float(p_value),
        "n_valid_resamples": int(len(deltas)),
    }


# ---------------------------------------------------------------------------
# Holm-Bonferroni multiplicity correction
# ---------------------------------------------------------------------------


def holm_bonferroni(p_values) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values, preserving the input order."""
    p_values = np.asarray(p_values, dtype=float)
    n = len(p_values)
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = np.empty(n)
    for i in range(n):
        adjusted[i] = min(1.0, ranked[i] * (n - i))
    for i in range(1, n):  # enforce monotonicity of the step-down procedure
        adjusted[i] = max(adjusted[i], adjusted[i - 1])
    result = np.empty(n)
    result[order] = adjusted
    return result


# ---------------------------------------------------------------------------
# Per-task analysis
# ---------------------------------------------------------------------------


def run_task(task: str, oof_map: dict, cfg: Config) -> pd.DataFrame:
    """All paired comparisons for one task: multimodal vs image and
    multimodal vs tabular, across all eight metrics."""
    rows = []
    for name_a, name_b in COMPARISONS:
        oof_a, oof_b = oof_map[name_a], oof_map[name_b]
        for metric in METRIC_NAMES:
            result = paired_bootstrap(
                oof_a, oof_b, metric, cfg.n_bootstrap, cfg.seed
            )
            rows.append({
                "task": task,
                "comparison": f"{name_a}_vs_{name_b}",
                "metric": metric,
                **result,
            })
            print(f"  {task:<10} {name_a} vs {name_b:<8} {metric:<18} "
                  f"delta={result['point_delta']:+.4f} "
                  f"[{result['ci_lo']:+.4f}, {result['ci_hi']:+.4f}] "
                  f"p={result['p_value']:.4f}")
    return pd.DataFrame(rows)


def split_primary_secondary(results_df: pd.DataFrame, cfg: Config):
    """Primary rows are reported uncorrected; every remaining row is
    Holm-Bonferroni corrected within its task."""
    is_primary = (
        (results_df["comparison"]
         == f"{PRIMARY_COMPARISON[0]}_vs_{PRIMARY_COMPARISON[1]}")
        & (results_df["metric"] == PRIMARY_METRIC)
    )
    primary = results_df[is_primary].copy()

    secondary = results_df[~is_primary].copy()
    for task in TASKS:
        mask = secondary["task"] == task
        secondary.loc[mask, "p_holm"] = holm_bonferroni(
            secondary.loc[mask, "p_value"].to_numpy()
        )
    secondary["significant_holm"] = secondary["p_holm"] < cfg.alpha
    return primary, secondary


def format_cell(row) -> str:
    return (f"{row['point_delta']:+.3f} "
            f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}] "
            f"p={row['p_value']:.3f}")


def build_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    """One compact cell per metric: delta [95% CI] p-value."""
    rows = []
    for task in TASKS:
        for name_a, name_b in COMPARISONS:
            comparison = f"{name_a}_vs_{name_b}"
            row_data = {"task": task, "comparison": comparison}
            for metric in METRIC_NAMES:
                match = results_df[
                    (results_df["task"] == task)
                    & (results_df["comparison"] == comparison)
                    & (results_df["metric"] == metric)
                ]
                if not match.empty:
                    row_data[metric] = format_cell(match.iloc[0])
            rows.append(row_data)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Paired bootstrap significance testing: multimodal vs "
                    "unimodal baselines, per task."
    )
    for task in TASKS:
        for config_name in CONFIGS:
            parser.add_argument(
                f"--{task}-{config_name}",
                default=getattr(Config, f"{task}_{config_name}"),
                help=f"Path to the {task} {config_name} oof_predictions.csv",
            )
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--n-bootstrap", type=int, default=Config.n_bootstrap)
    parser.add_argument("--seed", type=int, default=Config.seed)
    args = parser.parse_args()

    cfg_kwargs = {
        f"{task}_{config_name}": getattr(args, f"{task}_{config_name}")
        for task in TASKS for config_name in CONFIGS
    }
    cfg_kwargs.update({
        "output_dir": args.output_dir,
        "n_bootstrap": args.n_bootstrap,
        "seed": args.seed,
    })
    return Config(**cfg_kwargs)


def main() -> None:
    cfg = parse_args()
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 90)
    print("PAIRED BOOTSTRAP SIGNIFICANCE TESTING")
    print(f"Resamples: {cfg.n_bootstrap} | seed: {cfg.seed}")
    print("=" * 90)

    # Load and validate the six OOF tables.
    oof = {}
    for task in TASKS:
        oof[task] = {}
        print(f"\n{task.capitalize()} OOF inputs:")
        for config_name in CONFIGS:
            path = getattr(cfg, f"{task}_{config_name}")
            oof[task][config_name] = load_oof(path, f"{task}/{config_name}")
            print(f"  {config_name:<11} <- {path}")
        check_matched_folds(oof[task], task)

    # Run all paired comparisons for both tasks.
    all_results = []
    for task in TASKS:
        print(f"\n--- {task.capitalize()} paired comparisons ---")
        all_results.append(run_task(task, oof[task], cfg))
    results_df = pd.concat(all_results, ignore_index=True)
    results_df.to_csv(out_dir / "paired_bootstrap_all_results.csv",
                      index=False)

    # Primary (uncorrected) vs secondary (Holm-Bonferroni within each task).
    primary, secondary = split_primary_secondary(results_df, cfg)
    primary.to_csv(out_dir / "primary_comparisons.csv", index=False)
    secondary.to_csv(out_dir / "secondary_comparisons_holm.csv", index=False)

    print("\n" + "=" * 90)
    print(f"PRIMARY COMPARISONS (multimodal vs image, {PRIMARY_METRIC}, "
          "uncorrected)")
    print("=" * 90)
    print(primary[["task", "metric", "point_delta", "ci_lo", "ci_hi",
                   "p_value"]].to_string(index=False))

    print("\n" + "=" * 90)
    print("SECONDARY COMPARISONS (Holm-Bonferroni corrected within each task)")
    print("=" * 90)
    print(secondary[["task", "comparison", "metric", "point_delta",
                     "ci_lo", "ci_hi", "p_value", "p_holm",
                     "significant_holm"]].to_string(index=False))

    n_significant = int(secondary["significant_holm"].sum())
    print(f"\nSignificant after correction: {n_significant} "
          f"of {len(secondary)} secondary comparisons")
    if n_significant:
        print(secondary[secondary["significant_holm"]][
            ["task", "comparison", "metric", "point_delta", "ci_lo",
             "ci_hi", "p_holm"]
        ].to_string(index=False))

    summary_df = build_summary_table(results_df)
    summary_df.to_csv(out_dir / "manuscript_summary_table.csv", index=False)
    print("\n" + "=" * 90)
    print("SUMMARY TABLE (delta [95% CI] p-value)")
    print("=" * 90)
    print(summary_df.to_string(index=False))

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
