from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
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
from torch.utils.data import DataLoader, Dataset, Sampler
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class Config:
    tabular_csv: str = "tabular_data.csv"   # clinical records, one row per patient
    image_root: str = "ultrasound_image"    # one subfolder per patient ID
    output_dir: str = "outputs_fibrosis_image"
    seed: int = 42
    n_folds: int = 5
    inner_val_size: float = 0.20   # share of training patients used for early stopping + threshold
    img_size: int = 224
    backbone: str = "efficientnet_b0"
    img_feature_dim: int = 1280
    epochs: int = 80
    patients_per_batch_train: int = 4
    patients_per_batch_eval: int = 4
    minority_fraction_per_batch: float = 0.5  # target share of F0 patients per training batch
    lr_head: float = 1e-4
    lr_backbone: float = 1e-5      # reduced learning rate after unfreezing
    weight_decay: float = 5e-3
    patience: int = 15
    min_epochs_before_stop: int = 5   # warm-up epochs before early stopping can trigger
    unfreeze_epoch: int = 17
    dropout: float = 0.3
    grad_clip: float = 1.0
    num_workers: int = 0
    bootstrap_iterations: int = 1000
    gradcam_patient_id: str = "BEH01121"
    apply_cleaning: bool = False   # set True only for raw, uncleaned frames
    dpi: int = 300


TASK_NAME = "Fibrosis"
CLASS_NAMES = {0: "F0", 1: "F1-F2"}
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}

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
# 1. Labels and frame index
# ---------------------------------------------------------------------------


def load_patient_labels(tabular_csv: str) -> pd.DataFrame:
    """Binary fibrosis label per patient: F0 -> 0, F1-F2 -> 1."""
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
    stage = pd.to_numeric(df["Fibroscan F"], errors="coerce")
    assert stage.notna().all(), "Invalid Fibroscan F values."

    labels = pd.DataFrame({
        "patient_id": df["patient_id"],
        "label": (stage.astype(int) >= 1).astype(int),
    })
    assert labels["patient_id"].nunique() == 113, "Expected 113 unique patients."
    labels = labels.sort_values("patient_id").reset_index(drop=True)

    counts = labels["label"].value_counts().sort_index()
    print(f"Patients: {len(labels)} | Class 0 (F0): {counts[0]}, "
          f"Class 1 (F1-F2): {counts[1]}")
    return labels


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
# 2. Optional frame cleaning (for raw scanner exports)
# ---------------------------------------------------------------------------


def clean_ultrasound_frame(img: Image.Image, margin: int = 5) -> Image.Image:
    """4-stage cleaning: text removal -> ROI crop -> per-channel min-max
    normalization -> 8-bit rescale. Only needed when the on-disk frames are
    raw exports that still contain on-screen annotations."""
    import cv2

    arr = np.array(img.convert("RGB"))

    # Stage 1: remove bright text-like blobs by area/aspect-ratio filtering
    # and inpaint them (TELEA).
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

    # Stage 2: crop to the bounding box of the largest contour + small margin.
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

    # Stage 3: per-image, per-channel min-max normalization to [0, 1].
    arr = arr.astype(np.float32)
    for channel in range(arr.shape[2]):
        cmin = arr[..., channel].min()
        cmax = arr[..., channel].max()
        arr[..., channel] = (arr[..., channel] - cmin) / (cmax - cmin + 1e-8)

    # Stage 4: rescale to 8-bit.
    return Image.fromarray((arr * 255.0).astype(np.uint8))


# ---------------------------------------------------------------------------
# 3. Transforms and dataset
# ---------------------------------------------------------------------------


def build_train_transform(img_size: int) -> T.Compose:
    """Augmentation applied to training frames only; every draw of a patient
    (including oversampled repeats) goes through a fresh random pipeline."""
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


class PatientFrameDataset(Dataset):
    """Frame-level dataset; every frame carries its parent patient's label."""

    def __init__(self, frame_df, transform=None, apply_cleaning=False):
        self.df = frame_df.reset_index(drop=True)
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
        return {
            "image": img,
            "label": torch.tensor(float(row["label"]), dtype=torch.float32),
            "patient_id": row["patient_id"],
        }


def patient_collate(batch):
    """Stack frames and map each frame to its patient's position in the batch,
    so the attention pooling can group frames by patient."""
    images = torch.stack([b["image"] for b in batch], dim=0)
    patient_ids = [b["patient_id"] for b in batch]
    unique_pids = sorted(set(patient_ids))
    pid_to_idx = {pid: i for i, pid in enumerate(unique_pids)}
    patient_index = torch.tensor(
        [pid_to_idx[pid] for pid in patient_ids], dtype=torch.long
    )
    label_by_pid = {b["patient_id"]: b["label"] for b in batch}
    labels = torch.stack([label_by_pid[pid] for pid in unique_pids])
    return {
        "image": images,
        "patient_index": patient_index,
        "label": labels,
        "patient_ids": unique_pids,
    }


# ---------------------------------------------------------------------------
# 4. Whole-patient batch samplers
# ---------------------------------------------------------------------------


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


class MinorityOversamplingBatchSampler(Sampler):
    """Class-balanced training sampler for severe class imbalance: every
    training batch draws a fixed share of minority-class patients with
    replacement (fresh augmentation on each draw) and fills the rest with
    majority-class patients drawn without replacement within each epoch.
    Whole patients only, so the attention pooling always sees a patient's
    complete frame set."""

    def __init__(self, frame_df, patients_per_batch, minority_fraction, seed):
        self.patient_to_indices = frame_df.groupby("patient_id").indices
        patient_labels = (
            frame_df.drop_duplicates("patient_id")
            .set_index("patient_id")["label"]
        )
        counts = patient_labels.value_counts()
        self.minority_class = int(counts.idxmin())
        self.minority_pids = patient_labels[
            patient_labels == self.minority_class
        ].index.tolist()
        self.majority_pids = patient_labels[
            patient_labels != self.minority_class
        ].index.tolist()

        self.patients_per_batch = patients_per_batch
        self.n_minority_per_batch = max(
            1, int(round(patients_per_batch * minority_fraction))
        )
        self.n_majority_per_batch = patients_per_batch - self.n_minority_per_batch
        self.n_batches = -(-len(self.majority_pids)
                           // self.n_majority_per_batch)
        self.rng = np.random.RandomState(seed)

    def __iter__(self):
        # Majority patients cycle without replacement so each appears about
        # once per epoch; minority patients are oversampled with replacement.
        majority_order = self.rng.permutation(self.majority_pids).tolist()
        cursor = 0
        for _ in range(self.n_batches):
            if cursor + self.n_majority_per_batch > len(majority_order):
                majority_order = self.rng.permutation(self.majority_pids).tolist()
                cursor = 0
            batch_maj = majority_order[cursor:cursor + self.n_majority_per_batch]
            cursor += self.n_majority_per_batch
            batch_min = self.rng.choice(
                self.minority_pids, size=self.n_minority_per_batch,
                replace=True,
            ).tolist()

            indices = []
            for pid in batch_maj + batch_min:
                indices.extend(self.patient_to_indices[pid].tolist())
            yield indices

    def __len__(self):
        return self.n_batches


# ---------------------------------------------------------------------------
# 5. Model: shared EfficientNet-B0 encoder + attention-MIL pooling + head
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


class ImageOnlyClassifier(nn.Module):
    """EfficientNet-B0 encoder shared across frames -> attention-MIL pooling
    -> MLP head producing a single logit per patient."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone, pretrained=True, num_classes=0
        )
        self.mil_pool = AttentionMILPool(
            in_dim=cfg.img_feature_dim, hidden=64, dropout=cfg.dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(cfg.img_feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(128, 1),
        )
        self.freeze_backbone()

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, images, patient_index):
        frame_embeddings = self.backbone(images)
        pooled, attn_weights = self.mil_pool(frame_embeddings, patient_index)
        logits = self.classifier(pooled).squeeze(-1)
        return logits, attn_weights


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


def make_loaders(frame_df, train_pids, val_pids, test_pids, cfg):
    train_transform = build_train_transform(cfg.img_size)
    eval_transform = build_eval_transform(cfg.img_size)

    subsets = {}
    for name, pids in (("train", train_pids), ("val", val_pids),
                       ("test", test_pids)):
        subsets[name] = frame_df[frame_df["patient_id"].isin(pids)] \
            .reset_index(drop=True)

    train_ds = PatientFrameDataset(subsets["train"], train_transform,
                                   cfg.apply_cleaning)
    val_ds = PatientFrameDataset(subsets["val"], eval_transform,
                                 cfg.apply_cleaning)
    test_ds = PatientFrameDataset(subsets["test"], eval_transform,
                                  cfg.apply_cleaning)

    train_loader = DataLoader(
        train_ds,
        batch_sampler=MinorityOversamplingBatchSampler(
            subsets["train"], cfg.patients_per_batch_train,
            cfg.minority_fraction_per_batch, cfg.seed,
        ),
        num_workers=cfg.num_workers,
        collate_fn=patient_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_sampler=PatientGroupedBatchSampler(
            subsets["val"], cfg.patients_per_batch_eval
        ),
        num_workers=cfg.num_workers,
        collate_fn=patient_collate,
    )
    test_loader = DataLoader(
        test_ds,
        batch_sampler=PatientGroupedBatchSampler(
            subsets["test"], cfg.patients_per_batch_eval
        ),
        num_workers=cfg.num_workers,
        collate_fn=patient_collate,
    )
    return train_loader, val_loader, test_loader


@torch.no_grad()
def predict_patients(model, loader, device):
    """One probability per patient, keyed by patient ID."""
    model.eval()
    probs, labels = {}, {}
    for batch in loader:
        images = batch["image"].to(device)
        patient_index = batch["patient_index"].to(device)
        logits, _ = model(images, patient_index)
        batch_probs = torch.sigmoid(logits).cpu().numpy()
        batch_labels = batch["label"].cpu().numpy()
        for pid, prob, label in zip(batch["patient_ids"], batch_probs,
                                    batch_labels):
            probs[pid] = float(prob)
            labels[pid] = int(label)
    ordered = sorted(probs)
    y_prob = np.array([probs[pid] for pid in ordered])
    y_true = np.array([labels[pid] for pid in ordered])
    return y_true, y_prob, ordered


def train_one_fold(fold, train_pids, test_pids, frame_df, cfg, device,
                   out_dir):
    """Train one outer fold. The inner validation split (carved out of the
    training patients) drives early stopping and threshold selection; the
    outer test fold stays untouched until the final evaluation pass."""
    label_by_pid = (
        frame_df.drop_duplicates("patient_id").set_index("patient_id")["label"]
    )
    train_labels = [label_by_pid[pid] for pid in train_pids]
    inner_train_pids, inner_val_pids = train_test_split(
        train_pids,
        test_size=cfg.inner_val_size,
        stratify=train_labels,
        random_state=cfg.seed + fold,
    )

    train_loader, val_loader, test_loader = make_loaders(
        frame_df, inner_train_pids, inner_val_pids, test_pids, cfg
    )

    model = ImageOnlyClassifier(cfg).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        [
            {"params": model.mil_pool.parameters(), "lr": cfg.lr_head},
            {"params": model.classifier.parameters(), "lr": cfg.lr_head},
        ],
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
            patient_index = batch["patient_index"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            logits, _ = model(images, patient_index)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        val_true, val_prob, _ = predict_patients(model, val_loader, device)
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
    # untouched outer test fold.
    val_true, val_prob, _ = predict_patients(model, val_loader, device)
    threshold = select_threshold(val_true, val_prob)

    test_true, test_prob, test_pids_out = predict_patients(
        model, test_loader, device
    )
    assert set(test_pids_out) == set(test_pids)

    return {
        "fold": fold,
        "history": history,
        "threshold": threshold,
        "best_val_auc": best_val_auc,
        "y_true": test_true,
        "y_prob": test_prob,
        "patient_ids": test_pids_out,
    }


def run_cross_validation(labels, frame_df, cfg, device, out_dir):
    """Patient-level stratified 5-fold CV on the fibrosis label. Sorted
    patient IDs plus a fixed seed give the same folds as the other fibrosis
    configuration scripts."""
    skf = StratifiedKFold(
        n_splits=cfg.n_folds, shuffle=True, random_state=cfg.seed
    )
    fold_results, oof_rows, fold_assignments = [], [], {}

    for fold, (tr_idx, te_idx) in enumerate(
        skf.split(labels["patient_id"], labels["label"]), start=1
    ):
        train_pids = labels["patient_id"].iloc[tr_idx].tolist()
        test_pids = labels["patient_id"].iloc[te_idx].tolist()
        fold_assignments[fold] = {"train": train_pids, "test": test_pids}

        print(f"\n--- Fold {fold}/{cfg.n_folds} "
              f"(train {len(train_pids)}, test {len(test_pids)} patients) ---")
        result = train_one_fold(
            fold, train_pids, test_pids, frame_df, cfg, device, out_dir
        )
        fold_results.append(result)

        y_pred = (result["y_prob"] >= result["threshold"]).astype(int)
        for pid, yt, yp, ypr in zip(
            result["patient_ids"], result["y_true"], y_pred, result["y_prob"]
        ):
            oof_rows.append({
                "patient_id": pid,
                "fold": fold,
                "true_label": int(yt),
                "predicted_probability": float(ypr),
                "predicted_label": int(yp),
                "threshold": result["threshold"],
            })

        fold_metrics = compute_metrics(result["y_true"], y_pred,
                                       result["y_prob"])
        fold_metrics.update({"fold": fold, "threshold": result["threshold"]})
        result["metrics"] = fold_metrics
        print(f"Fold {fold}: threshold={result['threshold']:.2f}, "
              f"AUC-ROC={fold_metrics['auc_roc']:.3f}, "
              f"BalAcc={fold_metrics['balanced_accuracy']:.3f}, "
              f"F1={fold_metrics['f1']:.3f}")

    fold_df = pd.DataFrame([r["metrics"] for r in fold_results])
    oof_df = pd.DataFrame(oof_rows) \
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
    return fold_df, oof_df, fold_results, fold_assignments


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
    ax.set_title(f"Pooled OOF Confusion Matrix — Image-Only ({TASK_NAME})\n"
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
    ax.set_title(f"ROC Curves — Image-Only EfficientNet-B0 ({TASK_NAME})",
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
    ax.set_title(f"Precision-Recall Curves — Image-Only ({TASK_NAME})",
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
# 10. Grad-CAM
# ---------------------------------------------------------------------------


class GradCAM:
    """Grad-CAM from a target convolutional layer via forward/backward hooks."""

    def __init__(self, model, target_layer):
        self.gradients = None
        self.activations = None
        self.hook_fwd = target_layer.register_forward_hook(self._save_act)
        self.hook_bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, module, inputs, output):
        self.activations = output.detach()

    def _save_grad(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self):
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self.activations).sum(dim=1, keepdim=True))
        cam = cam.squeeze().cpu().numpy()
        return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

    def remove(self):
        self.hook_fwd.remove()
        self.hook_bwd.remove()


def load_fold_model(fold, cfg, device, out_dir):
    """Reload a saved fold checkpoint for inference or explanation."""
    model = ImageOnlyClassifier(cfg).to(device)
    state = torch.load(out_dir / f"fold{fold}_best.pt", map_location=device,
                       weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def most_attended_frame(model, frame_paths, cfg, device):
    """Frame with the highest attention weight under the MIL pooling."""
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
        embeddings = model.backbone(images)
        _, attn_weights = model.mil_pool(embeddings, patient_index)
    best = int(torch.argmax(attn_weights).item())
    return frame_paths[best], best


def run_gradcam(patient_id, oof_df, frame_df, cfg, device, out_dir):
    """Grad-CAM for one patient, using the fold model for which this patient
    was held out, computed on the patient's most-attended frame."""
    patient_rows = oof_df[oof_df["patient_id"] == patient_id]
    assert len(patient_rows) == 1, f"Patient not found in OOF: {patient_id}"
    patient_oof = patient_rows.iloc[0]
    fold = int(patient_oof["fold"])

    model = load_fold_model(fold, cfg, device, out_dir)
    frame_paths = frame_df[frame_df["patient_id"] == patient_id] \
        ["img_path"].tolist()
    best_path, _ = most_attended_frame(model, frame_paths, cfg, device)

    img = Image.open(best_path).convert("RGB")
    if cfg.apply_cleaning:
        img = clean_ultrasound_frame(img)
    tensor = build_eval_transform(cfg.img_size)(img).unsqueeze(0).to(device)
    tensor.requires_grad_(True)

    extractor = GradCAM(model, model.backbone.conv_head)
    patient_index = torch.zeros(1, dtype=torch.long, device=device)
    logits, _ = model(tensor, patient_index)
    model.zero_grad()
    logits[0].backward()
    cam = extractor.generate()
    extractor.remove()

    raw = np.array(img.resize((cfg.img_size, cfg.img_size)))
    cam_img = np.array(
        Image.fromarray((cam * 255).astype(np.uint8))
        .resize((cfg.img_size, cfg.img_size))
    ) / 255.0

    true_text = CLASS_NAMES[int(patient_oof["true_label"])]
    pred_text = CLASS_NAMES[int(patient_oof["predicted_label"])]
    prob = patient_oof["predicted_probability"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=cfg.dpi)
    axes[0].imshow(raw)
    axes[0].set_title("Original frame", fontweight="bold")
    axes[1].imshow(raw)
    axes[1].imshow(cam_img, cmap="jet", alpha=0.45)
    axes[1].set_title("Grad-CAM overlay", fontweight="bold")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(
        f"Grad-CAM — {patient_id} ({TASK_NAME}, fold {fold} model)\n"
        f"True: {true_text} | OOF prediction: {pred_text} | "
        f"P({CLASS_NAMES[1]}): {prob:.3f} | {patient_oof['outcome_type']}",
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(out_dir / f"gradcam_{patient_id}.png", dpi=cfg.dpi,
                bbox_inches="tight")
    plt.close(fig)
    print(f"Grad-CAM saved for patient {patient_id} "
          f"(fold {fold} model, frame {Path(best_path).name})")


def predict_new_patient(model, frame_paths, threshold, cfg, device):
    """Inference for a new patient: attention-pooled probability and binary
    prediction at the given threshold."""
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
        logits, attn_weights = model(images, patient_index)
        prob = torch.sigmoid(logits).item()
    return {
        "predicted_probability": prob,
        "predicted_label": int(prob >= threshold),
        "threshold": threshold,
        "attention_weights": attn_weights.detach().cpu().numpy().tolist(),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Image-only EfficientNet-B0 — fibrosis (F0 vs F1-F2)."
    )
    parser.add_argument("--tabular-csv", default=Config.tabular_csv)
    parser.add_argument("--image-root", default=Config.image_root)
    parser.add_argument("--output-dir", default=Config.output_dir)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--minority-fraction", type=float,
                        default=Config.minority_fraction_per_batch,
                        help="Target share of minority-class (F0) patients "
                             "in each training batch.")
    parser.add_argument("--bootstrap-iterations", type=int,
                        default=Config.bootstrap_iterations)
    parser.add_argument("--gradcam-patient-id", default=Config.gradcam_patient_id)
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
        minority_fraction_per_batch=args.minority_fraction,
        bootstrap_iterations=args.bootstrap_iterations,
        gradcam_patient_id=args.gradcam_patient_id,
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
    print(f"IMAGE-ONLY EFFICIENTNET-B0 — {TASK_NAME.upper()} "
          f"({CLASS_NAMES[0]} vs {CLASS_NAMES[1]})")
    print(f"Device: {device}")
    print(f"Training batches: {cfg.patients_per_batch_train} patients, "
          f"~{cfg.minority_fraction_per_batch:.0%} minority class")
    print("=" * 80)

    labels = load_patient_labels(cfg.tabular_csv)
    frame_df = build_frame_index(cfg.image_root, labels)

    print("\n[1/5] Patient-level stratified 5-fold cross-validation")
    fold_df, oof_df, fold_results, fold_assignments = run_cross_validation(
        labels, frame_df, cfg, device, out_dir
    )
    fold_df.to_csv(out_dir / "fold_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    with open(out_dir / "fold_assignments.json", "w") as f:
        json.dump(fold_assignments, f, indent=2)

    print("\n[2/5] Fold-wise mean +/- SD (eight metrics)")
    summary_df = summarize_folds(fold_df)
    summary_df.to_csv(out_dir / "fold_metrics_summary.csv", index=False)

    print("\n[3/5] Confusion matrix, ROC curves, PR curves, training curves")
    plot_confusion_matrix(oof_df, cfg, out_dir)
    plot_roc_curves(oof_df, cfg, out_dir)
    plot_pr_curves(oof_df, cfg, out_dir)
    plot_training_curves(fold_results, cfg, out_dir)

    print("\n[4/5] Bootstrap 95% CIs "
          f"({cfg.bootstrap_iterations} patient-level resamples, "
          "stratified by fold and class)")
    ci_df = bootstrap_confidence_intervals(oof_df, cfg)
    ci_df.to_csv(out_dir / "bootstrap_95ci.csv", index=False)

    print("\n[5/5] Grad-CAM for case-study patient "
          f"{cfg.gradcam_patient_id}")
    run_gradcam(cfg.gradcam_patient_id, oof_df, frame_df, cfg, device,
                out_dir)

    print(f"\nDone. Outputs written to: {out_dir}")


if __name__ == "__main__":
    main()
