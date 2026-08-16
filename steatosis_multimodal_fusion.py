from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
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
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    tabular_csv: str = "tabular_data.csv"   # clinical records, one row per patient
    image_root: str = "ultrasound_image"    # one subfolder per patient ID
    output_dir: str = "outputs_steatosis_multimodal"
    seed: int = 42
    n_folds: int = 5
    inner_val_size: float = 0.20   # share of training patients for early stopping + threshold
    img_size: int = 224
    backbone: str = "efficientnet_b0"
    img_feature_dim: int = 1280
    tab_feature_dim: int = 61
    tab_hidden_dim: int = 64
    tab_embed_dim: int = 32
    fusion_dim: int = 64
    epochs: int = 80
    patients_per_batch_train: int = 2
    patients_per_batch_eval: int = 4
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5      # reduced learning rate after unfreezing
    weight_decay: float = 5e-3
    patience: int = 15
    min_epochs_before_stop: int = 5   # warm-up epochs before early stopping can trigger
    unfreeze_epoch: int = 17
    dropout: float = 0.3
    num_workers: int = 0
    bootstrap_iterations: int = 1000
    tsne_patient_id: str = "BEH01121"
    tsne_perplexity: int = 15
    apply_cleaning: bool = False   # set True only for raw, uncleaned frames
    dpi: int = 300


TASK_NAME = "Steatosis"
CLASS_NAMES = {0: "S0-S1", 1: "S2-S3"}
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
DERIVED_FEATURES = ["AST_ALT_ratio", "TyG_index", "HSI", "FLI", "LAP"]

# Columns excluded from the predictor set to prevent target leakage.
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

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
    "font.size": 12,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})


def set_seeds(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# 1. Cohort, labels, and frame index
# ---------------------------------------------------------------------------


def load_cohort(tabular_csv: str) -> pd.DataFrame:
    """Clinical records with cleaned patient IDs, binary-encoded sex, and the
    binary steatosis target (S0-S1 -> 0, S2-S3 -> 1)."""
    df = pd.read_csv(tabular_csv)
    df.columns = df.columns.str.strip()
    if df.iloc[-1].isna().all():
        df = df.iloc[:-1].copy()

    df["patient_id"] = (
        df["Patient ID"]
        .astype(str)
        .str.replace("\\", "/", regex=False)
        .str.split("/")
        .str[-1]
        .str.replace(" ", "", regex=False)
        .str.strip()
    )
    df["sex"] = df["sex"].astype(str).str.strip().map(SEX_MAP)
    assert df["sex"].notna().all(), "Unmapped sex value found."

    stage = pd.to_numeric(df["Steatosis stage"], errors="coerce")
    assert stage.notna().all(), "Invalid steatosis stage values."
    df["steatosis_binary"] = (stage.astype(int) >= 2).astype(int)

    assert len(df) == 113, f"Expected 113 patients, got {len(df)}."
    assert df["patient_id"].nunique() == len(df), "Patient IDs must be unique."

    counts = df["steatosis_binary"].value_counts().sort_index()
    print(f"Patients: {len(df)} | Class 0 (S0-S1): {counts[0]}, "
          f"Class 1 (S2-S3): {counts[1]}")
    return df


def build_frame_index(image_root: str, labels: pd.DataFrame) -> pd.DataFrame:
    """One row per ultrasound frame, with the parent patient's label attached."""
    rows = []
    for _, row in labels.iterrows():
        folder = Path(image_root) / row["patient_id"]
        if not folder.is_dir():
            continue
        frames = sorted(
            p for p in folder.glob("*")
            if p.suffix.lower() in IMG_EXTENSIONS
        )
        for frame_path in frames:
            rows.append({
                "patient_id": row["patient_id"],
                "img_path": str(frame_path),
                "label": int(row["label"]),
            })

    frame_df = pd.DataFrame(rows)
    n_patients = frame_df["patient_id"].nunique()
    assert n_patients == len(labels), (
        f"Only {n_patients} of {len(labels)} patients have image folders."
    )
    print(f"Frames indexed: {len(frame_df)} across {n_patients} patients")
    return frame_df


# ---------------------------------------------------------------------------
# 2. Tabular feature engineering (56 raw + 5 derived = 61)
# ---------------------------------------------------------------------------


def raw_feature_columns(df: pd.DataFrame) -> list[str]:
    """Numeric non-leakage raw predictors: 62 variables - 6 leakage = 56."""
    cols = [
        c for c in df.columns
        if c not in LEAKAGE_COLS and pd.api.types.is_numeric_dtype(df[c])
    ]
    assert len(cols) == 56, f"Expected 56 raw predictors, got {len(cols)}."
    return cols


def add_steatosis_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Append the five steatosis-specific derived features. Applied to
    median-imputed values inside each fold (imputation -> engineering ->
    scaling). All log inputs are strictly positive in this cohort, so
    positivity is asserted instead of flooring values."""
    df = df.copy()
    ast = df["AST (SGOT)"].astype(float)
    alt = df["ALT (SGPT)"].astype(float)
    tg = df["Triglyceride"].astype(float)          # mg/dL
    fbs = df["Fasting Blood sugar"].astype(float)  # mg/dL
    bmi = df["BMI"].astype(float)
    waist = df["waist"].astype(float)              # cm
    ggt = df["GGT"].astype(float)                  # U/L

    for name, series in {"AST": ast, "ALT": alt, "TG": tg,
                         "FBS": fbs, "GGT": ggt}.items():
        assert (series > 0).all(), f"{name} must be strictly positive."

    # AST/ALT ratio (shared with the fibrosis feature set).
    df["AST_ALT_ratio"] = ast / alt

    # TyG = ln(TG[mg/dL] * FBS[mg/dL] / 2).
    df["TyG_index"] = np.log(tg * fbs / 2.0)

    # HSI = 8*(ALT/AST) + BMI + 2*I(Female) + 2*I(Diabetes); the diabetes
    # indicator is derived from fasting blood sugar >= 126 mg/dL.
    female = (df["sex"] == 1).astype(float)
    diabetes = (fbs >= 126.0).astype(float)
    df["HSI"] = 8.0 * (alt / ast) + bmi + 2.0 * female + 2.0 * diabetes

    # FLI = expit(L) * 100 with
    # L = 0.953 ln(TG) + 0.139 BMI + 0.718 ln(GGT) + 0.053 Waist - 15.745.
    fli_linear = (
        0.953 * np.log(tg)
        + 0.139 * bmi
        + 0.718 * np.log(ggt)
        + 0.053 * waist
        - 15.745
    )
    df["FLI"] = expit(fli_linear) * 100.0

    # LAP with TG converted to mmol/L (TG[mg/dL] / 88.5);
    # males: (Waist - 65) * TG, females: (Waist - 58) * TG.
    tg_mmol = tg / 88.5
    df["LAP"] = np.where(
        df["sex"] == 0, (waist - 65.0) * tg_mmol, (waist - 58.0) * tg_mmol
    )
    return df


def fit_imputer(train_df: pd.DataFrame, raw_cols: list[str]) -> SimpleImputer:
    """Median imputation fitted on training patients only."""
    return SimpleImputer(strategy="median").fit(train_df[raw_cols])


def transform_features(
    df: pd.DataFrame, raw_cols: list[str], imputer: SimpleImputer
) -> pd.DataFrame:
    """Impute, then append engineered indices -> 61-dimensional frame."""
    X = pd.DataFrame(
        imputer.transform(df[raw_cols]), columns=raw_cols, index=df.index
    )
    X = add_steatosis_indices(X)
    feature_cols = raw_cols + DERIVED_FEATURES
    assert X.shape[1] == 61, f"Expected 61 features, got {X.shape[1]}."
    return X[feature_cols]


def build_fold_tabular(cohort_df, raw_cols, fit_pids, all_pids):
    """Fit imputer + scaler on the inner training patients, then transform
    every patient of the fold. Returns one 61-d vector per patient."""
    indexed = cohort_df.set_index("patient_id")
    imputer = fit_imputer(indexed.loc[fit_pids], raw_cols)
    X = transform_features(indexed.loc[all_pids], raw_cols, imputer)
    scaler = StandardScaler().fit(X.loc[fit_pids])
    X_scaled = scaler.transform(X).astype(np.float32)
    tab_by_pid = {pid: X_scaled[i] for i, pid in enumerate(all_pids)}
    return tab_by_pid, imputer, scaler


# ---------------------------------------------------------------------------
# 3. Optional frame cleaning (for raw scanner exports)
# ---------------------------------------------------------------------------


def clean_ultrasound_frame(img: Image.Image, margin: int = 5) -> Image.Image:
    """4-stage cleaning: text removal -> ROI crop -> per-channel min-max
    normalization -> 8-bit rescale. Only needed when the on-disk frames are
    raw exports that still contain on-screen annotations."""
    import cv2

    arr = np.array(img.convert("RGB"))

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.zeros_like(gray)
    for contour in contours:
        area = cv2.contourArea(contour)
        _, _, w, h = cv2.boundingRect(contour)
        aspect = max(w, h) / max(1, min(w, h))
        if area < 2000 and (aspect > 2.0 or area < 500):
            cv2.drawContours(mask, [contour], -1, 255, thickness=-1)
    if mask.sum() > 0:
        arr = cv2.inpaint(arr, mask, 3, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1 = min(arr.shape[1], x + w + margin)
        y1 = min(arr.shape[0], y + h + margin)
        arr = arr[y0:y1, x0:x1]

    arr = arr.astype(np.float32)
    for channel in range(arr.shape[2]):
        cmin = arr[..., channel].min()
        cmax = arr[..., channel].max()
        arr[..., channel] = (arr[..., channel] - cmin) / (cmax - cmin + 1e-8)

    return Image.fromarray((arr * 255.0).astype(np.uint8))


# ---------------------------------------------------------------------------
# 4. Transforms, dataset, collate, samplers
# ---------------------------------------------------------------------------


def build_train_transform(img_size: int) -> T.Compose:
    """Augmentation applied to training frames only."""
    return T.Compose([
        T.Resize((img_size + 20, img_size + 20)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(degrees=10),
        T.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def build_eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


class MultimodalPatientDataset(Dataset):
    """Frame-level dataset; each item returns one frame plus its parent
    patient's preprocessed 61-d tabular vector and label."""

    def __init__(self, frame_df, tab_by_pid, transform=None,
                 apply_cleaning=False):
        self.df = frame_df.reset_index(drop=True)
        self.tab_by_pid = tab_by_pid
        self.transform = transform
        self.apply_cleaning = apply_cleaning

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row["img_path"]).convert("RGB")
        if self.apply_cleaning:
            img = clean_ultrasound_frame(img)
        if self.transform:
            img = self.transform(img)
        tabular = torch.tensor(self.tab_by_pid[row["patient_id"]],
                               dtype=torch.float32)
        return {
            "image": img,
            "tabular": tabular,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
        }


def multimodal_collate(batch):
    """Stack frames, map each frame to its patient's position in the batch,
    and keep exactly one tabular vector per patient."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    patient_ids = [b["patient_id"] for b in batch]
    unique_pids = sorted(set(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    patient_index = torch.tensor(
        [pid_to_idx[pid] for pid in patient_ids], dtype=torch.long
    )
    tab_by_pid = {}
    label_by_pid = {}
    for b in batch:
        tab_by_pid[b["patient_id"]] = b["tabular"]
        label_by_pid[b["patient_id"]] = b["label"]
    tabular = torch.stack([tab_by_pid[pid] for pid in unique_pids])
    labels = torch.stack([label_by_pid[pid] for pid in unique_pids])
    return {
        "image": images,
        "tabular": tabular,
        "patient_index": patient_index,
        "label": labels,
        "patient_ids": unique_pids,
    }


class PatientGroupedBatchSampler(Sampler):
    """Deterministic evaluation sampler: every patient's frames stay together
    in exactly one batch; every patient appears exactly once."""

    def __init__(self, frame_df, patients_per_batch):
        self.patient_to_indices = frame_df.groupby("patient_id").indices
        self.patient_ids = sorted(self.patient_to_indices.keys())
        self.patients_per_batch = patients_per_batch

    def __iter__(self):
        for start in range(0, len(self.patient_ids), self.patients_per_batch):
            batch_pids = self.patient_ids[start:start + self.patients_per_batch]
            indices = []
            for pid in batch_pids:
                indices.extend(self.patient_to_indices[pid].tolist())
            yield indices

    def __len__(self):
        return -(-len(self.patient_ids) // self.patients_per_batch)


class PatientWeightedBatchSampler(Sampler):
    """Class-balanced training sampler: draws whole patients with replacement
    (class weights inversely proportional to patient counts), keeping each
    sampled patient's full frame set together in one batch."""

    def __init__(self, frame_df, patients_per_batch, seed):
        self.patient_to_indices = frame_df.groupby("patient_id").indices
        patient_labels = (
            frame_df.drop_duplicates("patient_id")
            .set_index("patient_id")["label"]
        )
        self.patient_ids = list(patient_labels.index)
        counts = patient_labels.value_counts().to_dict()
        total = sum(counts.values())
        weight_by_class = {cls: total / cnt for cls, cnt in counts.items()}
        weights = np.array(
            [weight_by_class[patient_labels[pid]] for pid in self.patient_ids]
        )
        self.weights = weights / weights.sum()
        self.patients_per_batch = patients_per_batch
        self.n_batches = -(-len(self.patient_ids) // patients_per_batch)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        for _ in range(self.n_batches):
            sampled = self.rng.choice(
                self.patient_ids,
                size=self.patients_per_batch,
                replace=True,
                p=self.weights,
            )
            indices = []
            for pid in sampled:
                indices.extend(self.patient_to_indices[pid].tolist())
            yield indices

    def __len__(self):
        return self.n_batches


# ---------------------------------------------------------------------------
# 5. Model: image branch, tabular branch, gated residual fusion, head
# ---------------------------------------------------------------------------


class AttentionMILPool(nn.Module):
    """Learns a scalar attention weight per frame and returns the weighted
    sum of each patient's frame embeddings."""

    def __init__(self, in_dim=1280, hidden=64, dropout=0.3):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, frame_embeddings, patient_index):
        scores = self.attention(frame_embeddings).squeeze(-1)
        n_patients = int(patient_index.max().item()) + 1
        pooled = frame_embeddings.new_zeros(n_patients, frame_embeddings.size(1))
        weights = torch.zeros_like(scores)
        for p in range(n_patients):
            mask = patient_index == p
            w = torch.softmax(scores[mask], dim=0)
            pooled[p] = (frame_embeddings[mask] * w.unsqueeze(-1)).sum(dim=0)
            weights[mask] = w
        return pooled, weights


class TabularEncoder(nn.Module):
    """61 -> 64 -> 32 MLP with layer normalization, producing the compact
    clinical embedding."""

    def __init__(self, in_dim=61, hidden_dim=64, embed_dim=32, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class GatedResidualFusion(nn.Module):
    """Asymmetric gated residual fusion:
      - image residual pathway: linear projection 1280 -> 64, no activation;
      - a single learned scalar gate g = sigmoid(w_g), shared across patients,
        scales the tabular embedding uniformly;
      - the gated tabular embedding is concatenated with the full image
        embedding (32 + 1280 = 1312) and passed through a fusion MLP
        (1312 -> 64, ReLU, dropout) producing the multimodal correction;
      - fused representation = image residual + correction (unweighted add).
    """

    def __init__(self, img_dim=1280, tab_dim=32, fusion_dim=64, dropout=0.3):
        super().__init__()
        self.img_projection = nn.Linear(img_dim, fusion_dim)
        self.gate_logit = nn.Parameter(torch.zeros(1))
        self.fusion_mlp = nn.Sequential(
            nn.Linear(tab_dim + img_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_img, h_tab):
        h_res = self.img_projection(h_img)
        gate = torch.sigmoid(self.gate_logit)
        z = torch.cat([gate * h_tab, h_img], dim=1)
        h_corr = self.fusion_mlp(z)
        return h_res + h_corr, gate


class MultimodalFusionModel(nn.Module):
    """Shared EfficientNet-B0 encoder + attention-MIL pooling (image branch),
    tabular MLP encoder (clinical branch), gated residual fusion, and a
    classification head producing one logit per patient."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True, num_classes=0
        )
        self.mil_pool = AttentionMILPool(
            in_dim=cfg.img_feature_dim, hidden=64, dropout=cfg.dropout
        )
        self.tab_encoder = TabularEncoder(
            in_dim=cfg.tab_feature_dim,
            hidden_dim=cfg.tab_hidden_dim,
            embed_dim=cfg.tab_embed_dim,
            dropout=cfg.dropout,
        )
        self.fusion = GatedResidualFusion(
            img_dim=cfg.img_feature_dim,
            tab_dim=cfg.tab_embed_dim,
            fusion_dim=cfg.fusion_dim,
            dropout=cfg.dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.fusion_dim, 32),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(32, 1),
        )
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, tabular, patient_index):
        frame_embeddings = self.backbone(images)
        h_img, attn_weights = self.mil_pool(frame_embeddings, patient_index)
        h_tab = self.tab_encoder(tabular)
        h_fusion, gate = self.fusion(h_img, h_tab)
        logit = self.classifier(h_fusion).squeeze(-1)
        return logit, {
            "gate": gate,
            "h_fusion": h_fusion,
            "attn_weights": attn_weights,
        }


# ---------------------------------------------------------------------------
# 6. Metrics and threshold selection
# ---------------------------------------------------------------------------


def compute_metrics(y_true, y_pred, y_prob) -> dict:
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
    """Threshold maximizing balanced accuracy on the inner validation split."""
    candidates = np.round(np.arange(0.05, 0.951, 0.01), 2)
    scores = [
        balanced_accuracy_score(y_true, (y_prob >= t).astype(int))
        for t in candidates
    ]
    return float(candidates[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# 7. Training and evaluation
# ---------------------------------------------------------------------------


def make_loaders(frame_df, tab_by_pid, train_pids, val_pids, test_pids, cfg):
    train_transform = build_train_transform(cfg.img_size)
    eval_transform = build_eval_transform(cfg.img_size)

    subsets = {}
    for name, pids in (("train", train_pids), ("val", val_pids),
                       ("test", test_pids)):
        subsets[name] = frame_df[frame_df["patient_id"].isin(pids)] \
            .reset_index(drop=True)

    train_ds = MultimodalPatientDataset(subsets["train"], tab_by_pid,
                                        train_transform, cfg.apply_cleaning)
    val_ds = MultimodalPatientDataset(subsets["val"], tab_by_pid,
                                      eval_transform, cfg.apply_cleaning)
    test_ds = MultimodalPatientDataset(subsets["test"], tab_by_pid,
                                       eval_transform, cfg.apply_cleaning)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=PatientWeightedBatchSampler(
            subsets["train"], cfg.patients_per_batch_train, cfg.seed
        ),
        num_workers=cfg.num_workers,
        collate_fn=multimodal_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PatientGroupedBatchSampler(
            subsets["val"], cfg.patients_per_batch_eval
        ),
        num_workers=cfg.num_workers,
        collate_fn=multimodal_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=PatientGroupedBatchSampler(
            subsets["test"], cfg.patients_per_batch_eval
        ),
        num_workers=cfg.num_workers,
        collate_fn=multimodal_collate,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def predict_patients(model, loader, device, collect_embeddings=False):
    """One probability per patient, keyed by patient ID; optionally also the
    64-d fused embedding per patient."""
    model.eval()
    probs, labels, embeddings = {}, {}, {}
    for batch in loader:
        images = batch["image"].to(device)
        tabular = batch["tabular"].to(device)
        patient_index = batch["patient_index"].to(device)
        logits, aux = model(images, tabular, patient_index)
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        batch_labels = batch["label"].cpu().numpy()
        h_fusion = aux["h_fusion"].cpu().numpy()
        for i, pid in enumerate(batch["patient_ids"]):
            probs[pid] = float(batch_probs[i])
            labels[pid] = int(batch_labels[i])
            if collect_embeddings:
                embeddings[pid] = h_fusion[i]
    ordered = sorted(probs)
    y_prob = np.array([probs[pid] for pid in ordered])
    y_true = np.array([labels[pid] for pid in ordered])
    emb_matrix = (
        np.stack([embeddings[pid] for pid in ordered])
        if collect_embeddings else None
    )
    return y_true, y_prob, ordered, emb_matrix


def train_one_fold(fold, train_pids, test_pids, cohort_df, frame_df,
                   raw_cols, cfg, device, out_dir):
    """Train one outer fold. The inner validation split (carved out of the
    training patients) drives early stopping and threshold selection; the
    outer test fold stays untouched until the final evaluation pass.
    Tabular preprocessing is fitted on the inner training patients only."""
    label_by_pid = cohort_df.set_index("patient_id")["steatosis_binary"]
    train_labels = [label_by_pid[pid] for pid in train_pids]
    inner_train_pids, inner_val_pids = train_test_split(
        train_pids,
        test_size=cfg.inner_val_size,
        stratify=train_labels,
        random_state=cfg.seed + fold,
    )

    all_fold_pids = list(inner_train_pids) + list(inner_val_pids) \
        + list(test_pids)
    tab_by_pid, imputer, scaler = build_fold_tabular(
        cohort_df, raw_cols, inner_train_pids, all_fold_pids
    )

    train_loader, val_loader, test_loader = make_loaders(
        frame_df, tab_by_pid, inner_train_pids, inner_val_pids, test_pids, cfg
    )

    model = MultimodalFusionModel(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    head_params = (
        list(model.mil_pool.parameters())
        + list(model.tab_encoder.parameters())
        + list(model.fusion.parameters())
        + list(model.classifier.parameters())
    )
    optimizer = torch.optim.AdamW(
        [{"params": head_params, "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )

    history = {"epoch": [], "train_loss": [], "val_auc": []}
    best_val_auc, best_state, patience_counter = -np.inf, None, 0
    backbone_unfrozen = False

    for epoch in range(1, cfg.epochs + 1):
        # Staged fine-tuning: unfreeze the backbone at a reduced learning rate.
        if epoch == cfg.unfreeze_epoch and not backbone_unfrozen:
            model.unfreeze_backbone()
            optimizer.add_param_group({
                "params": model.backbone.parameters(),
                "lr": cfg.lr_backbone,
            })
            backbone_unfrozen = True

        model.train()
        epoch_loss, n_batches = 0.0, 0
        for batch in train_loader:
            images = batch["image"].to(device)
            tabular = batch["tabular"].to(device)
            patient_index = batch["patient_index"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _ = model(images, tabular, patient_index)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_true, val_prob, _, _ = predict_patients(model, val_loader, device)
        val_auc = roc_auc_score(val_true, val_prob)

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss / max(n_batches, 1))
        history["val_auc"].append(val_auc)
        print(f"  Fold {fold} | epoch {epoch:3d}/{cfg.epochs} | "
              f"train_loss={history['train_loss'][-1]:.4f} | "
              f"val_AUC={val_auc:.4f}")

        # Patience-based early stopping on inner validation AUC, after a
        # minimum number of warm-up epochs.
        if epoch >= cfg.min_epochs_before_stop:
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= cfg.patience:
                print(f"  Fold {fold}: early stop at epoch {epoch} "
                      f"(best val AUC {best_val_auc:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(model.state_dict(), out_dir / f"fold{fold}_best.pt")

    # Threshold from the inner validation split, then one final pass on the
    # untouched outer test fold (also collecting fused embeddings).
    val_true, val_prob, _, _ = predict_patients(model, val_loader, device)
    threshold = select_threshold(val_true, val_prob)

    test_true, test_prob, test_pids_out, test_emb = predict_patients(
        model, test_loader, device, collect_embeddings=True
    )
    assert set(test_pids_out) == set(test_pids)

    # Learned scalar gate value for this fold (global tabular weighting).
    gate_value = float(torch.sigmoid(model.fusion.gate_logit).item())

    # Preprocessing bundle needed to reproduce this fold's inference later.
    joblib.dump(
        {
            "imputer": imputer,
            "scaler": scaler,
            "threshold": threshold,
            "raw_cols": raw_cols,
            "feature_cols": raw_cols + DERIVED_FEATURES,
        },
        out_dir / f"fold{fold}_preprocessing.joblib",
    )

    return {
        "fold": fold,
        "history": history,
        "threshold": threshold,
        "best_val_auc": best_val_auc,
        "gate_value": gate_value,
        "y_true": test_true,
        "y_prob": test_prob,
        "patient_ids": test_pids_out,
        "embeddings": test_emb,
    }


def run_cross_validation(cohort_df, frame_df, cfg, device, out_dir):
    """Patient-level stratified 5-fold CV. Sorted patient IDs plus a fixed
    seed give the same folds as the other steatosis configuration scripts."""
    raw_cols = raw_feature_columns(cohort_df)
    labels = (
        cohort_df[["patient_id", "steatosis_binary"]]
        .rename(columns={"steatosis_binary": "label"})
        .sort_values("patient_id")
        .reset_index(drop=True)
    )
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )

    fold_results, oof_rows, emb_rows, fold_assignments = [], [], [], {}

    for fold, (tr_idx, te_idx) in enumerate(
        skf.split(labels["patient_id"], labels["label"]), start=1
    ):
        train_pids = labels["patient_id"].iloc[tr_idx].tolist()
        test_pids = labels["patient_id"].iloc[te_idx].tolist()
        fold_assignments[fold] = {"train": train_pids, "test": test_pids}

        print(f"\n--- Fold {fold}/{cfg.n_folds} "
              f"(train {len(train_pids)}, test {len(test_pids)} patients) ---")
        result = train_one_fold(
            fold, train_pids, test_pids, cohort_df, frame_df, raw_cols,
            cfg, device, out_dir,
        )
        fold_results.append(result)

        y_pred = (result["y_prob"] >= result["threshold"]).astype(int)
        for i, pid in enumerate(result["patient_ids"]):
            oof_rows.append({
                "patient_id": pid,
                "fold": fold,
                "true_label": int(result["y_true"][i]),
                "predicted_probability": float(result["y_prob"][i]),
                "predicted_label": int(y_pred[i]),
                "threshold": result["threshold"],
            })
            emb_rows.append({
                "patient_id": pid,
                **{f"h_fusion_{j}": float(result["embeddings"][i, j])
                   for j in range(result["embeddings"].shape[1])},
            })

        fold_metrics = compute_metrics(result["y_true"], y_pred,
                                       result["y_prob"])
        fold_metrics.update({
            "fold": fold,
            "threshold": result["threshold"],
            "gate_value": result["gate_value"],
        })
        result["metrics"] = fold_metrics
        print(f"Fold {fold}: threshold={result['threshold']:.2f}, "
              f"gate g={result['gate_value']:.3f}, "
              f"AUC-ROC={fold_metrics['auc_roc']:.3f}, "
              f"BalAcc={fold_metrics['balanced_accuracy']:.3f}, "
              f"F1={fold_metrics['f1']:.3f}")

    fold_df = pd.DataFrame([r["metrics"] for r in fold_results])
    oof_df = pd.DataFrame(oof_rows) \
        .sort_values("patient_id").reset_index(drop=True)
    emb_df = pd.DataFrame(emb_rows) \
        .sort_values("patient_id").reset_index(drop=True)

    # Exactly one out-of-fold prediction per patient.
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
    return fold_df, oof_df, emb_df, fold_results, fold_assignments


def summarize_folds(fold_df: pd.DataFrame) -> pd.DataFrame:
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
# 8. Visualizations
# ---------------------------------------------------------------------------


def plot_confusion_matrix(oof_df, cfg, out_dir):
    cm = confusion_matrix(oof_df["true_label"], oof_df["predicted_label"],
                          labels=[0, 1])
    assert cm.sum() == 113

    fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=cfg.dpi)
    image = ax.imshow(cm, cmap="Blues")
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha="center", va="center",
                    fontsize=18, fontweight="bold",
                    color="white" if cm[row, col] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], [f"Predicted {CLASS_NAMES[0]}",
                           f"Predicted {CLASS_NAMES[1]}"])
    ax.set_yticks([0, 1], [f"True {CLASS_NAMES[0]}",
                           f"True {CLASS_NAMES[1]}"])
    ax.set_xlabel("Predicted class", fontweight="bold")
    ax.set_ylabel("True class", fontweight="bold")
    ax.set_title(f"Pooled OOF Confusion Matrix — Multimodal ({TASK_NAME})\n"
                 "113 unique patients across five outer folds",
                 fontweight="bold", pad=14)
    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Patient count", fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix_pooled_oof.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


def plot_roc_curves(oof_df, cfg, out_dir):
    """Per-fold ROC curves plus the interpolated mean curve."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=cfg.dpi)
    mean_fpr = np.linspace(0, 1, 100)
    interp_tprs, fold_aucs = [], []

    for fold, sub in oof_df.groupby("fold"):
        fpr, tpr, _ = roc_curve(sub["true_label"],
                                sub["predicted_probability"])
        fold_auc = roc_auc_score(sub["true_label"],
                                 sub["predicted_probability"])
        fold_aucs.append(fold_auc)
        ax.plot(fpr, tpr, alpha=0.6, lw=1.5,
                label=f"Fold {fold} (AUC={fold_auc:.3f})")
        interp_tprs.append(np.interp(mean_fpr, fpr, tpr))

    ax.plot(mean_fpr, np.mean(interp_tprs, axis=0), color="black", lw=2.5,
            label=f"Mean ROC (AUC={np.mean(fold_aucs):.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", lw=1.2,
            label="No-discrimination reference")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("False positive rate", fontweight="bold")
    ax.set_ylabel("True positive rate / sensitivity", fontweight="bold")
    ax.set_title(f"ROC Curves — Multimodal Fusion ({TASK_NAME})",
                 fontweight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "roc_curves_per_fold.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


def plot_pr_curves(oof_df, cfg, out_dir):
    """Per-fold precision-recall curves plus the interpolated mean curve."""
    fig, ax = plt.subplots(figsize=(7.2, 6.2), dpi=cfg.dpi)
    mean_recall = np.linspace(0, 1, 100)
    interp_precisions, fold_aps = [], []

    for fold, sub in oof_df.groupby("fold"):
        precision, recall, _ = precision_recall_curve(
            sub["true_label"], sub["predicted_probability"]
        )
        fold_ap = average_precision_score(sub["true_label"],
                                          sub["predicted_probability"])
        fold_aps.append(fold_ap)
        ax.plot(recall, precision, alpha=0.6, lw=1.5,
                label=f"Fold {fold} (AP={fold_ap:.3f})")
        order = np.argsort(recall)
        interp_precisions.append(
            np.interp(mean_recall, recall[order], precision[order])
        )

    prevalence = oof_df["true_label"].mean()
    ax.plot(mean_recall, np.mean(interp_precisions, axis=0), color="black",
            lw=2.5, label=f"Mean PR (AP={np.mean(fold_aps):.3f})")
    ax.axhline(prevalence, color="gray", linestyle="--", lw=1.2,
               label=f"Positive-class prevalence = {prevalence:.3f}")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Recall / sensitivity", fontweight="bold")
    ax.set_ylabel("Precision / positive predictive value", fontweight="bold")
    ax.set_title(f"Precision-Recall Curves — Multimodal ({TASK_NAME})",
                 fontweight="bold", pad=14)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(out_dir / "pr_curves_per_fold.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(fold_results, cfg, out_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=cfg.dpi)
    for result in fold_results:
        history = result["history"]
        axes[0].plot(history["epoch"], history["train_loss"], alpha=0.8,
                     lw=1.5, label=f"Fold {result['fold']}")
        axes[1].plot(history["epoch"], history["val_auc"], alpha=0.8,
                     lw=1.5, label=f"Fold {result['fold']}")
    axes[0].set_title("Training loss per epoch", fontweight="bold", pad=14)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[1].set_title("Inner validation AUC-ROC per epoch",
                      fontweight="bold", pad=14)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUC-ROC")
    for ax in axes:
        ax.grid(alpha=0.25, linestyle="--")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


def plot_tsne(emb_df, oof_df, cfg, out_dir):
    """t-SNE of the 64-d fused out-of-fold embeddings, with the case-study
    patient highlighted by a star marker."""
    emb_df = emb_df.sort_values("patient_id").reset_index(drop=True)
    labels = (
        oof_df.sort_values("patient_id")["true_label"].to_numpy()
    )
    pids = emb_df["patient_id"].to_numpy()
    X = emb_df[[c for c in emb_df.columns if c.startswith("h_fusion_")]] \
        .to_numpy()

    tsne = TSNE(
        n_components=2,
        perplexity=cfg.tsne_perplexity,
        random_state=cfg.seed,
        init="pca",
        learning_rate="auto",
    )
    emb_2d = tsne.fit_transform(X)

    fig, ax = plt.subplots(figsize=(8, 6.5), dpi=cfg.dpi)
    target = cfg.tsne_patient_id
    target_idx = int(np.where(pids == target)[0][0]) \
        if target in pids else None

    for cls, color in ((0, "#1f77b4"), (1, "#d62728")):
        mask = labels == cls
        if target_idx is not None:
            mask = mask & (np.arange(len(pids)) != target_idx)
        ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1], c=color,
                   label=CLASS_NAMES[cls], alpha=0.55, s=55,
                   edgecolor="black", linewidth=0.4)

    if target_idx is not None:
        ax.scatter(emb_2d[target_idx, 0], emb_2d[target_idx, 1],
                   c="gold", marker="*", s=350, edgecolor="black",
                   linewidth=1.5, label=f"Patient {target}", zorder=5)

    ax.set_title(f"t-SNE of Fused Patient Embeddings — {TASK_NAME} (n=113)",
                 fontweight="bold", pad=14)
    ax.set_xlabel("t-SNE 1", fontweight="bold")
    ax.set_ylabel("t-SNE 2", fontweight="bold")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "tsne_fused_embeddings.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 9. Bootstrap confidence intervals
# ---------------------------------------------------------------------------


def bootstrap_confidence_intervals(oof_df, cfg) -> pd.DataFrame:
    """95% percentile CIs from patient-level resamples of the pooled
    out-of-fold predictions. Resampling is stratified by fold and class, so
    fold sizes and the class distribution are preserved in every resample;
    the bootstrapped quantity is the fold-mean of each metric."""
    rng = np.random.default_rng(cfg.seed)
    folds = sorted(oof_df["fold"].unique())

    def fold_mean_metrics(frame):
        per_fold = []
        for f in folds:
            sub = frame[frame["fold"] == f]
            per_fold.append(compute_metrics(
                sub["true_label"].to_numpy(),
                sub["predicted_label"].to_numpy(),
                sub["predicted_probability"].to_numpy(),
            ))
        return {m: float(np.mean([f[m] for f in per_fold]))
                for m in METRIC_NAMES}

    observed = fold_mean_metrics(oof_df)
    bootstrap_values = {m: [] for m in METRIC_NAMES}

    for _ in range(cfg.bootstrap_iterations):
        resampled_parts = []
        for f in folds:
            sub = oof_df[oof_df["fold"] == f]
            class_parts = []
            for cls in (0, 1):
                cls_sub = sub[sub["true_label"] == cls]
                draw = cls_sub.sample(
                    n=len(cls_sub), replace=True,
                    random_state=rng.integers(0, 2**31 - 1),
                )
                class_parts.append(draw)
            resampled_parts.append(pd.concat(class_parts))
        resampled = pd.concat(resampled_parts)
        values = fold_mean_metrics(resampled)
        for metric, value in values.items():
            if np.isfinite(value):
                bootstrap_values[metric].append(value)

    rows = []
    for metric in METRIC_NAMES:
        values = np.asarray(bootstrap_values[metric], dtype=float)
        lower, upper = np.percentile(values, [2.5, 97.5])
        rows.append({
            "metric": metric,
            "fold_mean_estimate": observed[metric],
            "ci95_lower": lower,
            "ci95_upper": upper,
            "n_valid_resamples": len(values),
        })
        print(f"{metric:<20} {observed[metric]:.3f} "
              f"(95% CI {lower:.3f}-{upper:.3f})")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 10. Inference
# ---------------------------------------------------------------------------


def load_fold_model(fold, cfg, device, out_dir):
    """Reload a saved fold checkpoint for inference or explanation."""
    model = MultimodalFusionModel(cfg).to(device)
    state = torch.load(out_dir / f"fold{fold}_best.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_new_patient(model, frame_paths, tabular_raw, prep, cfg, device):
    """Inference for a new patient.

    frame_paths: list of ultrasound frame paths for the patient.
    tabular_raw: one-row DataFrame with the raw clinical columns.
    prep: the fold's preprocessing bundle (imputer, scaler, threshold).
    """
    X = transform_features(tabular_raw, prep["raw_cols"], prep["imputer"])
    X_scaled = prep["scaler"].transform(X).astype(np.float32)
    tabular = torch.tensor(X_scaled, dtype=torch.float32, device=device)

    eval_transform = build_eval_transform(cfg.img_size)
    tensors = []
    for path in frame_paths:
        img = Image.open(path).convert("RGB")
        if cfg.apply_cleaning:
            img = clean_ultrasound_frame(img)
        tensors.append(eval_transform(img))
    images = torch.stack(tensors).to(device)
    patient_index = torch.zeros(len(tensors), dtype=torch.long, device=device)

    with torch.no_grad():
        logit, aux = model(images, tabular, patient_index)
        prob = torch.sigmoid(logit).item()
    return {
        "predicted_probability": prob,
        "predicted_label": int(prob >= prep["threshold"]),
        "threshold": prep["threshold"],
        "gate_value": float(aux["gate"].item()),
        "attention_weights": aux["attn_weights"].cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Multimodal gated residual fusion — steatosis "
                    "(S0-S1 vs S2-S3)."
    )
    parser.add_argument("--tabular-csv", default=Config.tabular_csv)
    parser.add_argument("--image-root", default=Config.image_root)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--bootstrap-iterations", type=int,
                        default=Config.bootstrap_iterations)
    parser.add_argument("--tsne-patient-id", default=Config.tsne_patient_id)
    parser.add_argument("--apply-cleaning", action="store_true",
                        help="Apply the 4-stage frame cleaning pipeline "
                             "(only for raw, uncleaned frames).")
    parser.add_argument("--num-workers", type=int, default=Config.num_workers)
    args = parser.parse_args()
    return Config(
        tabular_csv=args.tabular_csv,
        image_root=args.image_root,
        output_dir=args.output_dir,
        seed=args.seed,
        epochs=args.epochs,
        bootstrap_iterations=args.bootstrap_iterations,
        tsne_patient_id=args.tsne_patient_id,
        apply_cleaning=args.apply_cleaning,
        num_workers=args.num_workers,
    )


def main() -> None:
    cfg = parse_args()
    set_seeds(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(asdict(cfg), f, indent=2)

    print("=" * 80)
    print(f"MULTIMODAL GATED RESIDUAL FUSION — {TASK_NAME.upper()} "
          f"({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})")
    print(f"Device: {device}")
    print("=" * 80)

    cohort_df = load_cohort(cfg.tabular_csv)
    labels = (
        cohort_df[["patient_id", "steatosis_binary"]]
        .rename(columns={"steatosis_binary": "label"})
    )
    frame_df = build_frame_index(cfg.image_root, labels)

    print("\n[1/6] Patient-level stratified 5-fold cross-validation")
    fold_df, oof_df, emb_df, fold_results, fold_assignments = \
        run_cross_validation(cohort_df, frame_df, cfg, device, out_dir)
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    emb_df.to_csv(out_dir / "oof_fused_embeddings.csv", index=False)
    with open(out_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    print("\n[2/6] Fold-wise mean +/- SD (eight metrics)")
    summary_df = summarize_folds(fold_df)
    summary_df.to_csv(out_dir / "fold_metrics_summary.csv", index=False)
    print("Learned scalar gate g per fold: "
          + ", ".join(f"fold {r['fold']}: {r['gate_value']:.3f}"
                      for r in fold_results))

    print("\n[3/6] Confusion matrix, ROC curves, PR curves, training curves")
    plot_confusion_matrix(oof_df, cfg, out_dir)
    plot_roc_curves(oof_df, cfg, out_dir)
    plot_pr_curves(oof_df, cfg, out_dir)
    plot_training_curves(fold_results, cfg, out_dir)

    print("\n[4/6] Bootstrap 95% CIs "
          f"({cfg.bootstrap_iterations} patient-level resamples, "
          "stratified by fold and class)")
    ci_df = bootstrap_confidence_intervals(oof_df, cfg)
    ci_df.to_csv(out_dir / "bootstrap_95ci.csv", index=False)

    print("\n[5/6] t-SNE of fused embeddings "
          f"(case-study patient {cfg.tsne_patient_id})")
    plot_tsne(emb_df, oof_df, cfg, out_dir)

    print("\n[6/6] Fold checkpoints and preprocessing bundles saved "
          "for inference")
    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
