# ==========================================================
# DTT + BBO + SAFE CHECKPOINTING + FINAL GRAPHS
# - Saves BEST checkpoint immediately when F1 improves
# - So crashes won't lose best model
# - Produces: confusion matrix + ROC/PR at end
# ==========================================================

import os
import re
import copy
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, confusion_matrix, classification_report,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
from PIL import Image, UnidentifiedImageError
from glob import glob
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.notebook import tqdm
import time

# ------------------------
# CONFIG
# ------------------------
KAGGLE_DATASET_PATH = r'/content/drive/MyDrive/alzheimers_project/Alzheimer_MRI_4_classes_dataset'
MODEL_SAVE_PATH = r'/content/drive/MyDrive/alzheimers_project/trained_models'  # persistent
FIGURE_SAVE_PATH = r'/content/drive/MyDrive/alzheimers_project/figures'

IMAGE_SIZE = 224
BATCH_SIZE = 2
NUM_EPOCHS = 50
LEARNING_RATE = 5e-6
WEIGHT_DECAY = 1e-3
EARLY_STOPPING_PATIENCE = 20
MIN_DELTA = 0.001
RANDOM_SEED = 42
MAX_TIMESTEPS = 32
NUM_FOLDS = 2

# BBO
BBO_POP_SIZE = 8
BBO_ITERATIONS = 8
BBO_CANDIDATE_EPOCHS = 5
DIM_RANGE = (128, 768)

# ------------------------
# REPRODUCIBILITY
# ------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ------------------------
# DATA
# ------------------------
def safe_open_image(path, retries=3, delay=0.5):
    for i in range(retries):
        try:
            return Image.open(path).convert('L').convert('RGB')
        except (OSError, UnidentifiedImageError):
            if i == retries - 1:
                raise
            time.sleep(delay)

def custom_collate_fn(batch):
    slices_list, labels_list = zip(*batch)
    labels = torch.tensor(labels_list, dtype=torch.long)

    padded_slices, masks = [], []
    for patient_slices in slices_list:
        slices_to_process = patient_slices[:MAX_TIMESTEPS]
        n = len(slices_to_process)

        mask = torch.ones(MAX_TIMESTEPS, dtype=torch.bool)
        mask[:n] = False
        masks.append(mask)

        if n < MAX_TIMESTEPS:
            pad_slice = torch.zeros_like(slices_to_process[0])
            slices_to_process.extend([pad_slice] * (MAX_TIMESTEPS - n))

        padded_slices.append(torch.stack(slices_to_process))

    return torch.stack(padded_slices), labels, torch.stack(masks)

def load_data_from_your_code(base_path):
    conditions = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    img_exts = ('.jpg', '.jpeg', '.png')
    fname_pattern = re.compile(r'^(\d+)\s*\((\d+)\)$')
    records = []

    for condition in conditions:
        folder = os.path.join(base_path, condition)
        if not os.path.isdir(folder):
            continue

        images = [p for p in glob(os.path.join(folder, '*')) if p.lower().endswith(img_exts)]
        patient_dict = {}

        for img_path in images:
            fname = os.path.splitext(os.path.basename(img_path))[0]
            m = fname_pattern.match(fname)

            if m:
                day, patient_num = int(m.group(1)), int(m.group(2))
            else:
                try:
                    day, patient_num = int(fname), 1
                except ValueError:
                    continue

            patient_dict.setdefault(patient_num, {})
            patient_dict[patient_num][day] = img_path

        for patient_num in sorted(patient_dict.keys()):
            row = {"Patient_ID": f"{condition}_P{patient_num}", "Condition": condition}
            days_map = patient_dict[patient_num]
            for d in range(1, 33):
                row[f"Day_{d}"] = days_map.get(d, None)
            records.append(row)

    df = pd.DataFrame(records)

    patient_data = []
    for _, row in df.iterrows():
        filepaths = [p for p in row["Day_1":"Day_32"] if p is not None]
        patient_data.append({
            'patient_id': row['Patient_ID'],
            'label': row['Condition'],
            'filepath': filepaths
        })

    return pd.DataFrame(patient_data).set_index('patient_id')

class PatientTimeSeries_Dataset(Dataset):
    def __init__(self, patient_df, transform=None):
        self.patient_df = patient_df
        self.transform = transform
        self.classes = sorted(patient_df['label'].unique().tolist())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def __len__(self):
        return len(self.patient_df)

    def __getitem__(self, idx):
        patient_id = self.patient_df.index[idx]
        image_paths = self.patient_df.loc[patient_id]['filepath']
        label = self.patient_df.loc[patient_id]['label']
        label_idx = self.class_to_idx[label]

        slices = []
        for p in image_paths:
            img = safe_open_image(p)
            if self.transform:
                img = self.transform(img)
            slices.append(img)

        return slices, label_idx

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_patient_level_data_loaders(dataset_df, batch_size, num_folds, random_seed):
    labels_numeric = dataset_df['label'].astype('category').cat.codes
    class_counts = dataset_df['label'].value_counts()
    min_class_size = class_counts.min()

    if min_class_size < 5:
        num_folds = min(num_folds, max(2, min_class_size))
        print(f"⚠️ Small class ({min_class_size}). Using n_splits={num_folds}")
    else:
        num_folds = min(num_folds, min_class_size)
        print(f"Using n_splits={num_folds}")

    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_seed)
    fold_loaders = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(dataset_df, labels_numeric), start=1):
        train_df = dataset_df.iloc[tr_idx]
        val_df = dataset_df.iloc[va_idx]

        train_ds = PatientTimeSeries_Dataset(train_df, transform=train_transform)
        val_ds = PatientTimeSeries_Dataset(val_df, transform=val_transform)

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=True,
            collate_fn=custom_collate_fn, num_workers=0, pin_memory=False
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            collate_fn=custom_collate_fn, num_workers=0, pin_memory=False
        )

        fold_loaders.append({'train': train_loader, 'val': val_loader, 'class_to_idx': train_ds.class_to_idx})

    return fold_loaders

# ------------------------
# MODEL
# ------------------------
class SliceLevelEncoder(nn.Module):
    def __init__(self, reduced_dim=256):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.projection = nn.Linear(resnet.fc.in_features, reduced_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.projection(x)

class TemporalProgressionTransformer(nn.Module):
    def __init__(self, feature_dim, num_heads=8, num_layers=4, num_classes=2, dropout=0.3):
        super().__init__()
        self.pos_encoder = nn.Parameter(torch.randn(1, MAX_TIMESTEPS + 1, feature_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, feature_dim))

        enc = nn.TransformerEncoderLayer(
            d_model=feature_dim, nhead=num_heads, dropout=dropout, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(enc, num_layers=num_layers)

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, x, src_key_padding_mask=None):
        b = x.size(0)
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_encoder[:, :x.size(1), :]

        if src_key_padding_mask is not None:
            cls_mask = torch.zeros((b, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = torch.cat([cls_mask, src_key_padding_mask], dim=1)

        out = self.transformer_encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.classifier(out[:, 0, :])

class DTT_System(nn.Module):
    def __init__(self, num_classes=2, reduced_dim=256, num_heads=8, num_layers=4, dropout=0.3):
        super().__init__()
        self.slice_encoder = SliceLevelEncoder(reduced_dim=reduced_dim)
        self.temporal_transformer = TemporalProgressionTransformer(
            feature_dim=reduced_dim, num_heads=num_heads, num_layers=num_layers,
            num_classes=num_classes, dropout=dropout
        )

    def forward(self, x, mask=None):
        B, T, C, H, W = x.size()
        x = x.view(B*T, C, H, W)
        feat = self.slice_encoder(x).view(B, T, -1)
        return self.temporal_transformer(feat, mask)

class EarlyStopperF1:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_f1 = -np.inf

    def early_stop(self, f1_val):
        if f1_val > self.best_f1 + self.min_delta:
            self.best_f1 = f1_val
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

class BrownBearOptimizer:
    def __init__(self, pop_size=8, iterations=8, dim_range=(128, 768)):
        self.pop_size = pop_size
        self.iterations = iterations
        self.dim_range = dim_range

    def _process_dim(self, dim):
        dim = int(round(dim))
        dim = max(self.dim_range[0], min(self.dim_range[1], dim))
        if dim % 8 != 0:
            dim -= dim % 8
        return dim

    def initialize_population(self):
        pop = []
        for _ in range(self.pop_size):
            d = np.random.randint(self.dim_range[0], self.dim_range[1] + 1)
            pop.append({"reduced_dim": self._process_dim(d)})
        return pop

    def update_population(self, population, best):
        new_pop = []
        for _ in population:
            d = np.random.randint(self.dim_range[0], self.dim_range[1] + 1)
            new_pop.append({"reduced_dim": self._process_dim(d)})
        return new_pop

# ------------------------
# TRAIN / EVAL
# ------------------------
def train_model(model, loader, criterion, optimizer, device):
    model.train()
    run_loss, correct, total = 0.0, 0, 0

    for x, y, m in tqdm(loader, desc='Training'):
        x, y, m = x.to(device), y.to(device), m.to(device)
        optimizer.zero_grad()
        out = model(x, m)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        run_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

    return run_loss / total, correct / total

@torch.no_grad()
def evaluate_model(model, loader, criterion, device):
    model.eval()
    run_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels, all_probs = [], [], []

    for x, y, m in tqdm(loader, desc='Evaluating'):
        x, y, m = x.to(device), y.to(device), m.to(device)
        out = model(x, m)
        loss = criterion(out, y)
        probs = torch.softmax(out, dim=1)[:, 1]

        run_loss += loss.item() * x.size(0)
        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(y.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    f1m = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return run_loss / total, correct / total, f1m, all_preds, all_labels, all_probs

# ------------------------
# PLOTS
# ------------------------
def plot_conf_matrix(labels, preds, class_names, save_path):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_roc_pr(labels, probs, save_path):
    fpr, tpr, _ = roc_curve(labels, probs)
    roc_auc = auc(fpr, tpr)

    precision, recall, _ = precision_recall_curve(labels, probs)
    ap = average_precision_score(labels, probs)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(fpr, tpr, label=f'AUC = {roc_auc:.4f}')
    ax1.plot([0, 1], [0, 1], '--')
    ax1.set_title('ROC Curve')
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(recall, precision, label=f'AP = {ap:.4f}')
    ax2.set_title('Precision-Recall Curve')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
    return roc_auc, ap

# ------------------------
# MAIN
# ------------------------
if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
    os.makedirs(FIGURE_SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # data
    full_df = load_data_from_your_code(KAGGLE_DATASET_PATH)
    binary_df = full_df.copy()
    binary_df['label'] = binary_df['label'].apply(lambda x: 'Demented' if x != 'NonDemented' else 'NonDemented')

    fold = get_patient_level_data_loaders(binary_df, BATCH_SIZE, NUM_FOLDS, RANDOM_SEED)[0]
    train_loader, val_loader = fold['train'], fold['val']

    criterion = nn.CrossEntropyLoss()

    # ---------------- BBO ----------------
    bbo = BrownBearOptimizer(pop_size=BBO_POP_SIZE, iterations=BBO_ITERATIONS, dim_range=DIM_RANGE)
    population = bbo.initialize_population()

    best_score = -np.inf
    best_params = None

    print("\nStarting Brown Bear Optimization...")
    for it in range(bbo.iterations):
        print(f"\nBBO Iteration {it+1}/{bbo.iterations}")

        for ind in population:
            rd = ind["reduced_dim"]
            print("Testing reduced_dim:", rd)

            model = DTT_System(num_classes=2, reduced_dim=rd, num_heads=8, num_layers=4, dropout=0.3).to(device)
            optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

            for _ in range(BBO_CANDIDATE_EPOCHS):
                train_model(model, train_loader, criterion, optimizer, device)

            _, _, f1m, _, _, _ = evaluate_model(model, val_loader, criterion, device)
            print("F1:", f1m)

            if f1m > best_score:
                best_score = f1m
                best_params = {"reduced_dim": rd}

        population = bbo.update_population(population, best_params)

    print("\nBest Reduced Dimension Found:", best_params)
    print("Best Validation F1:", best_score)

    # -------- Final training --------
    print("\nStarting Final Training...")
    model = DTT_System(
        num_classes=2,
        reduced_dim=best_params["reduced_dim"],
        num_heads=8,
        num_layers=4,
        dropout=0.3
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopper = EarlyStopperF1(patience=EARLY_STOPPING_PATIENCE, min_delta=MIN_DELTA)

    best_weights = None
    best_f1 = -np.inf
    best_preds, best_labels, best_probs = None, None, None

    best_ckpt_path = os.path.join(MODEL_SAVE_PATH, "stage1_best_model.pth")

    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS}")
        train_loss, train_acc = train_model(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, f1m, preds, labels, probs = evaluate_model(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val F1(macro): {f1m:.4f}")

        if f1m > best_f1:
            best_f1 = f1m
            best_weights = copy.deepcopy(model.state_dict())
            best_preds, best_labels, best_probs = preds, labels, probs

            # SAVE IMMEDIATELY on improvement
            torch.save({
                "epoch": epoch + 1,
                "best_f1": best_f1,
                "reduced_dim": best_params["reduced_dim"],
                "model_state_dict": best_weights
            }, best_ckpt_path)
            print(f"✅ Saved new best checkpoint at epoch {epoch+1} | F1={best_f1:.4f}")

        if early_stopper.early_stop(f1m):
            print("Early stopping triggered.")
            break

    # load from checkpoint (safe)
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nStage 1 Training Complete.")
    print("Final Best F1:", ckpt["best_f1"])
    print("Best reduced_dim:", ckpt["reduced_dim"])
    print("Checkpoint:", best_ckpt_path)

    # figures + report
    class_names = ['Demented', 'NonDemented']
    plot_conf_matrix(best_labels, best_preds, class_names, os.path.join(FIGURE_SAVE_PATH, 'confusion_matrix.png'))
    roc_auc, pr_ap = plot_roc_pr(best_labels, best_probs, os.path.join(FIGURE_SAVE_PATH, 'roc_pr_curves.png'))

    print("\nClassification Report:")
    print(classification_report(best_labels, best_preds, target_names=class_names))
    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC (AP): {pr_ap:.4f}")
    print("Figures saved in:", FIGURE_SAVE_PATH)
