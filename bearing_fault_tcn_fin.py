"""
Bearing Fault Classification - TCN
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt
import re

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import scipy.io as scio
from sklearn.utils.class_weight import compute_class_weight

# SECTION 1 - CONFIGURATION
# =========================

# Data Directories
ADAMS_DATA_DIR = "./data_adams"
CWRU_DATA_DIR = None
MODEL_SAVE_PATH = "./model_tcn"

# Signal Settings
ADAMS_FS = 10_000  # Hz
CWRU_FS = 12_000  # Hz
ADAMS_SHAFT_FREQ_HZ = 9.8  # Hz
LOWPASS_CUTOFF_HZ = 4999  # Hz
LOWPASS_ORDER = 8  # Butterworth filter order

# Windowing
WINDOW_SECONDS = 1.0  # Length of each window
STRIDE_SECONDS = 0.2  # Step between window starts

# Positional Split Ratios
TRAIN_SPLIT = 0.45
VAL_SPLIT = 0.10
TEST_SPLIT = 0.45

# Label Rules
TYPE_RULES = [
    ("bpfi", "inner_race_fault"),
    ("bpfo", "outer_race_fault"),
    ("IR", "inner_race_fault"),
    ("OR", "outer_race_fault"),
    ("inner", "inner_race_fault"),
    ("outer", "outer_race_fault"),
    ("healthy", "healthy"),
    ("normal", "healthy"),
    ("baseline", "healthy"),
    ("nominal", "healthy"),
    ("roller", "roller_fault"),
    ("spalling", "spalling"),
]
SIZE_RULES = [
    ("005", "005"),
    ("01", "01"),
    ("03", "03"),
    ("05", "05"),
]
MANUAL_LABEL_MAP = {}

# TCN Architecture
TCN_CHANNELS = [16, 32, 32, 64]
KERNEL_SIZE = 6
DROPOUT_RATE = 0.5

# Training Settings
BATCH_SIZE = 32
LEARNING_RATE = 5e-4
EPOCHS = 100
RANDOM_SEED = 42
WEIGHT_DECAY = 1e-4

# SECTION 2 - LABEL PARSING
# =========================

def parse_compound_label(filename_stem: str):
    """
    Infer a label from a filename stem based on updated logic.
    """
    if filename_stem in MANUAL_LABEL_MAP:
        return MANUAL_LABEL_MAP[filename_stem]

    stem_lower = filename_stem.lower()

    # Compound Fault Detection
    ir_sizes = [f"ir_{size}" for k, size in SIZE_RULES]
    or_sizes = [f"or_{size}" for k, size in SIZE_RULES]

    found_ir = None
    found_or = None

    for sz in ir_sizes:
        if sz in stem_lower:
            found_ir = sz.upper()
            break

    for sz in or_sizes:
        if sz in stem_lower:
            found_or = sz.upper()
            break

    # If both are present, treat as specific compound fault
    if found_ir and found_or:
        return f"{found_ir}_{found_or}"

    # Spalling Fault Detection
    if "spalling" in stem_lower:
        if "ir" in stem_lower or "inner" in stem_lower:
            return "spalling_inner_race"
        elif "or" in stem_lower or "outer" in stem_lower:
            return "spalling_outer_race"
        else:
            # Fallback if position is missing
            return "spalling"

    # Standard Detection
    fault_type = None
    for keyword, label in TYPE_RULES:
        if keyword.lower() in stem_lower:
            fault_type = label
            break

    if fault_type is None:
        return None

    if fault_type == "healthy":
        return "healthy"

    # Determine fault size
    fault_size = None
    for keyword, size in SIZE_RULES:
        if keyword.lower() in stem_lower:
            fault_size = size
            break

    if fault_size is None:
        print(f"  [WARNING] No fault size found in '{filename_stem}' — "
              f"using type label only: '{fault_type}'")
        return fault_type

    # Check for specific orientation keywords (225deg, 45deg)
    orientation = None
    if "225deg" in stem_lower:
        orientation = "_225deg"
    elif "45deg" in stem_lower:
        orientation = "_45deg"

    # Construct final label
    base_label = f"{fault_type}_{fault_size}"
    if orientation:
        return f"{base_label}{orientation}"

    return base_label

# SECTION 3 - DATA LOADING
# ========================

def detrend_signal(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def load_tab_file(filepath: str) -> np.ndarray:
    with open(filepath, "r") as f:
        lines = f.readlines()
    if len(lines) < 8:
        raise ValueError(f"File too short: {filepath}")
    raw_headers = lines[6].strip().split("\t")
    col_names = [h.strip().strip('"').split(".")[-1] for h in raw_headers]

    rows = []
    for line in lines[7:]:
        line = line.strip()
        if not line: continue
        try:
            values = [float(v) for v in line.split("\t")]
            if len(values) == len(col_names):
                rows.append(values)
        except ValueError:
            continue

    df = pd.DataFrame(rows, columns=col_names)
    df = df.drop_duplicates(subset=["TIME"], keep="last").reset_index(drop=True)
    return df["Q"].values.astype(np.float32)


def load_mat_file(filepath: str):
    """
    Load a .mat file. Supports standard CWRU keys and Chalmers/Experimental keys.
    Reads the Sample Rate directly from the file metadata if available.
    """
    data = scio.loadmat(filepath)
    keys = list(data.keys())
    fs_key = next((k for k in keys if "sample_rate" in k.lower()), None)

    if fs_key:
        fs = float(data[fs_key].flatten()[0])
        print(f"  [INFO] Read Sample Rate from file metadata: {fs} Hz")
    else:
        fs = CWRU_FS
        print(f"  [WARNING] No 'Sample_rate' key found. Defaulting to CWRU_FS ({fs} Hz).")

    # Detect Signal Key
    de_key = next((k for k in keys if k.endswith("_DE_time")), None)
    if de_key is None:
        # Fallback for Chalmers (Probe Y)
        de_key = next((k for k in keys if "AI_4_AI_4_minus_Probe_Y" in k and "time" not in k), None)
        if de_key is None:
            de_key = next((k for k in keys if "AI_1" in k and "time" not in k), None)

        if de_key:
            print(f"  [INFO] Using fallback data key: '{de_key}'")

    if de_key is None:
        raise ValueError(
            f"No suitable signal key found in '{filepath}'.\n"
            f"Available keys: {[k for k in keys if not k.startswith('__')]}")

    # Extract Signal
    signal = data[de_key].flatten().astype(np.float32)
    signal = signal * 9.81 * 1000  # g to mm/s²

    # Detect Shaft Frequency
    shaft_freq_hz = 0.0
    rpm_key = next((k for k in keys if k.endswith("RPM")), None)

    if rpm_key:
        rpm = float(data[rpm_key].flatten()[0])
        shaft_freq_hz = rpm / 60.0
    else:
        # Parse from filename (e.g., RS10Hz)
        filename = os.path.basename(filepath)
        match = re.search(r'RS(\d+(?:\.\d+)?)Hz', filename, re.IGNORECASE)
        if match:
            shaft_freq_hz = float(match.group(1))
            print(f"  [INFO] Shaft frequency parsed from filename: {shaft_freq_hz} Hz")
        else:
            # Fallback based on filename or default
            print(f"  [WARNING] No RPM found. Defaulting to 10.0 Hz.")
            shaft_freq_hz = 10.0

    return signal, shaft_freq_hz, fs

def apply_lowpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter using Second-Order Sections.
    """
    nyquist = fs / 2.0
    wn = min(LOWPASS_CUTOFF_HZ, nyquist * 0.9999) / nyquist

    # Use output='sos'
    sos = butter(LOWPASS_ORDER, wn, btype="lowpass", output='sos')

    return sosfiltfilt(sos, signal)

# SECTION 4 - SIGNAL PROCESSING
# =============================

def process_and_split_file_tcn(signal: np.ndarray, fs: float,
                               run_diagnostics: bool = False):
    signal_detrended = detrend_signal(signal)
    signal_filtered = apply_lowpass_filter(signal_detrended, fs)

    window_size = int(WINDOW_SECONDS * fs)
    stride = int(STRIDE_SECONDS * fs)
    N = len(signal_filtered)

    if N < window_size:
        print(f"  [WARNING] Signal too short for one window. Skipping.")
        return [], [], []

    t1 = int(N * TRAIN_SPLIT)
    t2 = int(N * (TRAIN_SPLIT + VAL_SPLIT))

    if run_diagnostics:
        t = np.arange(len(signal_detrended)) / fs
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 5))
        ax1.plot(t, signal_detrended, color="steelblue")
        ax1.set_title("TCN Diagnostic 1/2 — Raw (Detrended)")
        ax1.set_ylabel("Amplitude")
        ax2.plot(t, signal_filtered, color="darkorange")
        ax2.set_title(f"TCN Diagnostic 2/2 — Filtered (Cutoff {LOWPASS_CUTOFF_HZ}Hz)")
        ax2.set_xlabel("Time (s)");
        ax2.set_ylabel("Amplitude")
        plt.tight_layout()
        print("\n[TCN Diagnostics] Plots shown. Close window to continue...")
        plt.show()

    def get_windows_in_region(start_idx, end_idx):
        segments = []
        for start in range(start_idx, end_idx - window_size + 1, stride):
            end = start + window_size
            seg = signal_filtered[start:end].copy()

            # Disregard any window that has a signal above the value of 10000
            if np.any(seg > 10000):
                continue

            std = np.std(seg)
            if std > 0:
                seg = (seg - np.mean(seg)) / std
            else:
                seg = seg - np.mean(seg)
            segments.append(seg.astype(np.float32))
        return segments

    train_segs = get_windows_in_region(0, t1)
    val_segs = get_windows_in_region(t1, t2)
    test_segs = get_windows_in_region(t2, N)

    return train_segs, val_segs, test_segs

# SECTION 5 - DATASET & DATA LOADING
# ==================================

class BearingFaultDatasetTCN(Dataset):
    def __init__(self, features: list, labels: np.ndarray,
                 class_names: list, label_encoder):
        self.X = torch.tensor(np.stack(features), dtype=torch.float32)[:, np.newaxis, :]
        self.y = torch.tensor(labels, dtype=torch.long)
        self.class_names = class_names
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _process_source(files, loader_fn, fs, shaft_freq_arg,
                    diagnostics_shown, run_diagnostics, source_tag):
    train, val, test = {"feats": [], "labels": []}, {"feats": [], "labels": []}, {"feats": [], "labels": []}
    skipped = []

    for fpath in files:
        label = parse_compound_label(fpath.stem)
        if label is None:
            skipped.append(fpath.name)
            continue
        try:
            result = loader_fn(str(fpath))

            if isinstance(result, tuple):
                signal, shaft_freq, read_fs = result
                if read_fs: fs = read_fs
            else:
                signal = result
                shaft_freq = shaft_freq_arg

        except Exception as e:
            print(f"  [ERROR] Failed to load {fpath.name}: {e}")
            continue

        print(f"  {source_tag} {fpath.name}  →  '{label}'  "
              f"(shaft: {shaft_freq:.2f} Hz)")

        run_diag = run_diagnostics and not diagnostics_shown
        tr, va, te = process_and_split_file_tcn(signal, fs, run_diag)

        if run_diag:
            diagnostics_shown = True

        print(f"               Segments -> Train: {len(tr)} | Val: {len(va)} | Test: {len(te)}")
        train["feats"].extend(tr);
        train["labels"].extend([label] * len(tr))
        val["feats"].extend(va);
        val["labels"].extend([label] * len(va))
        test["feats"].extend(te);
        test["labels"].extend([label] * len(te))

    return train, val, test, diagnostics_shown, skipped

def load_and_split_data_tcn(adams_dir=None, cwru_dir=None, run_diagnostics=True):
    all_train, all_val, all_test = {"feats": [], "labels": []}, {"feats": [], "labels": []}, {"feats": [], "labels": []}
    all_raw_labels = []
    diagnostics_shown = False

    adams_path = Path(adams_dir) if adams_dir else None
    if adams_path and adams_path.exists():
        tab_files = sorted(adams_path.glob("*.tab"))
        print(f"[Adams] Found {len(tab_files)} .tab files")
        tr, va, te, diagnostics_shown, skipped = _process_source(
            tab_files, load_tab_file, ADAMS_FS, ADAMS_SHAFT_FREQ_HZ,
            diagnostics_shown, run_diagnostics, "[Adams]"
        )
        for d, s in [(all_train, tr), (all_val, va), (all_test, te)]:
            d["feats"].extend(s["feats"]);
            d["labels"].extend(s["labels"])
        all_raw_labels.extend(tr["labels"] + va["labels"] + te["labels"])
        if skipped: print(f"  [Adams] Skipped {skipped}")

    cwru_path = Path(cwru_dir) if cwru_dir else None
    if cwru_path and cwru_path.exists():
        mat_files = sorted(cwru_path.glob("*.mat"))
        print(f"[CWRU] Found {len(mat_files)} .mat files")
        tr, va, te, diagnostics_shown, skipped = _process_source(
            mat_files, load_mat_file, CWRU_FS, None,
            diagnostics_shown, run_diagnostics, "[CWRU] "
        )
        for d, s in [(all_train, tr), (all_val, va), (all_test, te)]:
            d["feats"].extend(s["feats"]);
            d["labels"].extend(s["labels"])
        all_raw_labels.extend(tr["labels"] + va["labels"] + te["labels"])
        if skipped: print(f"  [CWRU] Skipped {skipped}")

    if not all_raw_labels:
        raise ValueError("No data loaded. Check directories and label rules.")

    label_encoder = LabelEncoder()
    label_encoder.fit(all_raw_labels)
    class_names = list(label_encoder.classes_)

    def make_bucket(bucket, name):
        if not bucket["feats"]: raise ValueError(f"{name} split is empty.")
        labels = label_encoder.transform(bucket["labels"])
        return BearingFaultDatasetTCN(bucket["feats"], labels, class_names, label_encoder)

    train_set = make_bucket(all_train, "Train")
    val_set = make_bucket(all_val, "Val")
    test_set = make_bucket(all_test, "Test")

    n_tr, n_va, n_te = len(train_set), len(val_set), len(test_set)
    print(f"\n[Dataset Summary]")
    print(f"  Train: {n_tr} | Val: {n_va} | Test: {n_te}")
    print(f"  Classes ({len(class_names)}): {class_names}\n")

    return train_set, val_set, test_set, class_names, label_encoder

# SECTION 6 - TCN MODEL
# =====================

class CausalConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((self.padding, 0), 0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                              dilation=dilation, padding=0)
        nn.utils.parametrizations.weight_norm(self.conv)

    def forward(self, x):
        x = self.pad(x)
        return self.conv(x)

class TCNResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, kernel_size=1)
                              if in_channels != out_channels else None)
        self.relu_out = nn.ReLU()

    def forward(self, x):
        residual = x if self.residual_conv is None else self.residual_conv(x)
        out = self.drop1(self.relu1(self.conv1(x)))
        out = self.drop2(self.relu2(self.conv2(out)))
        return self.relu_out(out + residual)

class TCN(nn.Module):
    def __init__(self, num_classes: int, tcn_channels: list = TCN_CHANNELS,
                 kernel_size: int = KERNEL_SIZE, dropout: float = DROPOUT_RATE):
        super().__init__()
        blocks = []
        in_ch = 1
        for i, out_ch in enumerate(tcn_channels):
            dilation = 2 ** i
            blocks.append(TCNResidualBlock(in_ch, out_ch, kernel_size, dilation, dropout))
            in_ch = out_ch

        self.tcn = nn.Sequential(*blocks)
        self.global_avg = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Linear(tcn_channels[-1], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        out = self.tcn(x)
        out = self.global_avg(out).squeeze(-1)
        return self.classifier(out)

    def receptive_field(self):
        rf = 1
        for block in self.tcn:
            dilation = block.conv1.conv.dilation[0]
            rf += 2 * (KERNEL_SIZE - 1) * dilation
        return rf

# SECTION 7 - TRAINING & EVALUATION
# =================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out = model(X_batch)
        loss = criterion(out, y_batch)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct += (out.argmax(dim=1) == y_batch).sum().item()
        total += len(y_batch)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out = model(X_batch)
        loss = criterion(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds = out.argmax(dim=1)
        correct += (preds == y_batch).sum().item()
        total += len(y_batch)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(y_batch.cpu().numpy())
    return total_loss / total, correct / total, all_preds, all_targets

def get_aggregated_label(label_str: str) -> str:
    """
    Converts specific fault labels into their parent fault types.
    """
    label_lower = label_str.lower()

    # Handle Compound Faults
    if "ir_" in label_lower and "or_" in label_lower:
        return "Compound Fault"

    # Handle Spalling
    if "spalling" in label_lower:
        return "Spalling"

    # Handle Healthy/Roller
    if label_str == "healthy": return "Healthy"
    if label_str == "roller_fault": return "Roller Fault"

    # Handle Standard (Type_Size_Orientation)
    parts = label_str.split('_')
    if not parts: return label_str

    # Extract known sizes
    known_sizes = [s for _, s in SIZE_RULES]
    size_index = -1
    for i, part in enumerate(parts):
        if part in known_sizes:
            size_index = i
            break

    # If a size was found, join the parts before the size.
    if size_index != -1:
        return "_".join(parts[:size_index])

    # Fallback: return original label if no rules matched
    return label_str

def show_training_results(train_losses, val_losses, train_accs, val_accs,
                          targets, preds, class_names, label_encoder, save_dir):
    """
    Generates and saves plots:
    1. Training & Validation Curves.
    2. Full CM.
    3. Aggregated CM (By Fault Type).
    4. Zoomed-in CM (Fault Sizes within each Type).
    """

    true_labels_str = label_encoder.inverse_transform(targets)
    pred_labels_str = label_encoder.inverse_transform(preds)

    # Plot 1: Training Curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(train_losses, label="Train", color="steelblue")
    ax1.plot(val_losses, label="Validation", color="darkorange")
    ax1.set_title("Loss per Epoch")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(train_accs, label="Train", color="steelblue")
    ax2.plot(val_accs, label="Validation", color="darkorange")
    ax2.set_title("Accuracy per Epoch")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim([0, 1.05])
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle("TCN Training & Validation History", fontsize=13, y=1.01)
    plt.tight_layout()

    save_path1 = os.path.join(save_dir, "tcn_training_curves.png")
    plt.savefig(save_path1, dpi=150)
    print(f"[Plot] Training curves saved to '{save_path1}'")
    plt.close()

    # Plot 2: Full CM
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    ax.set_title("TCN Confusion Matrix - Test Set (Full Detail)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    save_path2 = os.path.join(save_dir, "tcn_confusion_matrix.png")
    plt.savefig(save_path2, dpi=150)
    print(f"[Plot] Confusion matrix saved to '{save_path2}'")
    plt.close()

    # Plot 3: Aggregated CM (By Fault Type)
    true_types = [get_aggregated_label(l) for l in true_labels_str]
    pred_types = [get_aggregated_label(l) for l in pred_labels_str]

    # Get unique sorted types for the axis
    unique_types = sorted(list(set(true_types)))

    cm_type = confusion_matrix(true_types, pred_types, labels=unique_types)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm_type, annot=True, fmt='d', cmap="Greens",
                xticklabels=unique_types, yticklabels=unique_types,
                ax=ax)
    ax.set_title("TCN Confusion Matrix - Aggregated by Fault Type")
    ax.set_xlabel("Predicted Type")
    ax.set_ylabel("True Type")
    plt.tight_layout()

    save_path3 = os.path.join(save_dir, "tcn_confusion_matrix_by_type.png")
    plt.savefig(save_path3, dpi=150)
    print(f"[Plot] Aggregated confusion matrix saved to '{save_path3}'")
    plt.close()

    # Plot 4: Zoomed-In Matrices (Sizes within Types)
    print("[Plot] Generating zoomed-in confusion matrices for each fault type.")

    # Group specific class names by their aggregated parent type
    type_to_classes = {}
    for cls in class_names:
        parent_type = get_aggregated_label(cls)
        if parent_type not in type_to_classes:
            type_to_classes[parent_type] = []
        type_to_classes[parent_type].append(cls)

    # Loop through each fault type and create a sub-matrix
    for parent_type, sub_classes in type_to_classes.items():
        if len(sub_classes) < 2:
            continue

        # Filter true and predicted labels to only include samples belonging to this parent type
        indices = [i for i, l in enumerate(true_labels_str) if l in sub_classes]

        if not indices:
            continue

        subset_true = [true_labels_str[i] for i in indices]
        subset_pred = [pred_labels_str[i] for i in indices]

        # Create CM for this subset
        cm_zoom = confusion_matrix(subset_true, subset_pred, labels=sub_classes)

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm_zoom, annot=True, fmt='d', cmap="Purples",
                    xticklabels=sub_classes, yticklabels=sub_classes, ax=ax)

        safe_type_name = parent_type.replace(" ", "_").replace("/", "_")
        ax.set_title(f"TCN Confusion Matrix - Zoomed: {parent_type}")
        ax.set_xlabel("Predicted Size/Variant")
        ax.set_ylabel("True Size/Variant")
        plt.tight_layout()

        save_path_zoom = os.path.join(save_dir, f"tcn_confusion_matrix_zoomed_{safe_type_name}.png")
        plt.savefig(save_path_zoom, dpi=150)
        print(f"[Plot] Zoomed matrix for '{parent_type}' saved to '{save_path_zoom}'")
        plt.close()

    # Show all plots
    plt.show(block=True)

# SECTION 8 - MAIN
# ================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    # Load data (This will take a bit longer now with smaller stride)
    train_set, val_set, test_set, class_names, label_encoder = load_and_split_data_tcn(
        adams_dir=ADAMS_DATA_DIR, cwru_dir=CWRU_DATA_DIR, run_diagnostics=True)

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, pin_memory=True)

    print("[Calculating Class Weights.]")
    y_train_indices = label_encoder.transform(train_set.dataset.y) if isinstance(train_set,
                                                                                 torch.utils.data.Subset) else train_set.y.numpy()
    class_weights_arr = compute_class_weight(class_weight='balanced', classes=np.unique(y_train_indices),
        y=y_train_indices)
    class_weights_tensor = torch.tensor(class_weights_arr, dtype=torch.float32)

    class_weights_tensor = class_weights_tensor.to(device)
    print(f"  Weights: Min={class_weights_tensor.min():.2f}, Max={class_weights_tensor.max():.2f}")

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    model = TCN(num_classes=len(class_names)).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Params: {n_params:,} | RF: {model.receptive_field()} samples")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")

    print("[Training]\n")
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        vl_loss, vl_acc, _, _ = evaluate(model, val_loader, criterion, device)
        scheduler.step(vl_loss)

        train_losses.append(tr_loss); val_losses.append(vl_loss)
        train_accs.append(tr_acc);   val_accs.append(vl_acc)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>3}/{EPOCHS}  "
                  f"Train  loss={tr_loss:.4f}  acc={tr_acc:.3f}  |  "
                  f"Val  loss={vl_loss:.4f}  acc={vl_acc:.3f}")

        if vl_loss < best_val_loss:
            best_val_loss = vl_loss
            torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, "best_tcn_model.pt"))

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_PATH, "best_tcn_model.pt"),
                                     map_location=device, weights_only=True))
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)

    print(f"\n[Test] Loss: {test_loss:.4f} | Accuracy: {test_acc:.3f}\n")
    print("[Classification Report]")
    print(classification_report(targets, preds, target_names=class_names, zero_division=0))

    show_training_results(train_losses, val_losses, train_accs, val_accs,
                          targets, preds, class_names, label_encoder, MODEL_SAVE_PATH)
    joblib.dump(label_encoder, os.path.join(MODEL_SAVE_PATH, "tcn_label_encoder.pkl"))
    joblib.dump({
        "tcn_channels": TCN_CHANNELS, "kernel_size": KERNEL_SIZE,
        "dropout_rate": DROPOUT_RATE, "num_classes": len(class_names),
        "class_names": class_names, "window_seconds": WINDOW_SECONDS,
    }, os.path.join(MODEL_SAVE_PATH, "tcn_config.pkl"))

    print(f"\n[Done] Saved to '{MODEL_SAVE_PATH}/'")

if __name__ == "__main__":
    main()