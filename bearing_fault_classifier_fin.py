"""
Bearing Fault Classification - Feedforward Neural Network
=========================================================
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, hilbert
from scipy.signal.windows import hamming
from numpy.fft import fft
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

# SECTION 1 - CONFIGURATION
# =========================

# Directories
ADAMS_DATA_DIR  = "./data_adams"   # folder of Adams .tab files
CWRU_DATA_DIR   = None    # folder of CWRU .mat files "./data_cwru"
MODEL_SAVE_PATH = "./model"

# Signal Settings
ADAMS_FS = 10_000   # Hz, Adams sampling rate
CWRU_FS  = 12_000   # Hz, CWRU sampling rate

# Shaft rotational frequency for Adams order tracking.
# CWRU shaft frequency is read automatically from the .mat RPM field.
ADAMS_SHAFT_FREQ_HZ = 9.8 # Hz

# Low-pass filter settings.
LOWPASS_CUTOFF_HZ = 4999  # Hz
LOWPASS_ORDER     = 8     # Butterworth filter order

# Windowing
WINDOW_SECONDS = 1.0   # seconds
STRIDE_SECONDS = 0.25  # seconds

# Positional Split Ratios
TRAIN_SPLIT = 0.45
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.45

# Order Spectrum Settings
ORDER_LOW   = 0.5
ORDER_HIGH  = 100.0
ORDER_STEP  = 0.5

# Label Rules
TYPE_RULES = [
    ("bpfi",    "inner_race_fault"),
    ("bpfo",    "outer_race_fault"),
    ("IR",      "inner_race_fault"),
    ("OR",      "outer_race_fault"),
    ("inner",   "inner_race_fault"),
    ("outer",   "outer_race_fault"),
    ("healthy", "healthy"),
    ("normal",  "healthy"),
    ("baseline","healthy"),
    ("nominal", "healthy"),
    ("roller",  "roller_fault"),
    ("spalling","spalling"),
]
SIZE_RULES = [
    ("005", "005"),
    ("01",  "01"),
    ("03",  "03"),
    ("05",  "05"),
    ("007",  "007"),
    ("014",  "014"),
    ("021",  "021"),
]

# Manual override: Use for unusual filenames.
MANUAL_LABEL_MAP = {
    # "unusual_filename": "inner_race_fault_007",
}

# Network Architecture
HIDDEN_LAYERS = [128, 128, 128]
DROPOUT_RATE  = 0.3

# Training Settings
BATCH_SIZE    = 32
LEARNING_RATE = 1e-3
EPOCHS        = 100
RANDOM_SEED   = 42

# SECTION 2 - LABEL PARSING
# =========================

def parse_compound_label(filename_stem: str):
    """
    Infer a label from a filename stem based on updated logic.
    """
    if filename_stem in MANUAL_LABEL_MAP:
        return MANUAL_LABEL_MAP[filename_stem]

    stem_lower = filename_stem.lower()

    # Check if the filename contains both an IR size AND an OR size
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

    # Check for spalling and determine position
    if "spalling" in stem_lower:
        if "ir" in stem_lower or "inner" in stem_lower:
            return "spalling_inner_race"
        elif "or" in stem_lower or "outer" in stem_lower:
            return "spalling_outer_race"
        else:
            # Fallback if position is missing
            return "spalling"

    # Determine fault type
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

    # The model will ignore the orientation, treating 'IR_05_45deg' exactly the same as 'IR_05'.
    if "225deg" in stem_lower or "45deg" in stem_lower:
        return f"{fault_type}_{fault_size}"

    # Construct final label
    base_label = f"{fault_type}_{fault_size}"

    return base_label

# SECTION 3 - OTHERS
# ==================

def detrend_signal(x: np.ndarray) -> np.ndarray:
    return x - np.mean(x)

def get_feature_size() -> int:
    return len(np.arange(ORDER_LOW, ORDER_HIGH + ORDER_STEP / 2, ORDER_STEP))

# SECTION 4 - DATA LOADING
# ========================

def load_tab_file(filepath: str) -> np.ndarray:
    """
    Load an MSC Adams .tab export file.
    Returns the Q (accelerometer) column as a float32 array.
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    raw_headers = lines[6].strip().split("\t")
    col_names   = [h.strip().strip('"').split(".")[-1] for h in raw_headers]

    rows = []
    for line in lines[7:]:
        line = line.strip()
        if not line:
            continue
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
        # Found explicit sample rate in file
        fs = float(data[fs_key].flatten()[0])
        print(f"  [INFO] Read Sample Rate from file metadata: {fs} Hz")
    else:
        # Fallback for standard CWRU if data missing
        fs = CWRU_FS
        print(f"  [WARNING] No 'Sample_rate' key found. Defaulting to CWRU_FS ({fs} Hz).")

    de_key = None

    # Look for Housing Acceleration (AI_4)
    if de_key is None:
        for k in keys:
            if "AI_4" in k and "Housing" in k:
                de_key = k
                print(f"  [INFO] Using correct channel: Housing Accel ('{de_key}')")
                break

    # Fallback to Standard CWRU
    if de_key is None:
        de_key = next((k for k in keys if k.endswith("_DE_time")), None)
        if de_key:
            print(f"  [INFO] Using standard CWRU channel: '{de_key}'")

    # Last Resort: Shaft Displacement (AI_1)
    if de_key is None:
        print(f"  [WARNING] Housing Accel not found. Using fallback: Shaft Vertical Disp.")
        # Try to find specific Chalmers key
        if "AI_1_AI_1_minus_Probe_Y" in str(keys):
            de_key = next((k for k in keys if "AI_1_AI_1_minus_Probe_Y" in k), None)
        elif "AI_4_AI_4_minus_Probe_Y" in str(keys):
            # Handle edge case where key exists but didn't match above logic
            de_key = next((k for k in keys if "AI_4_AI_4_minus_Probe_Y" in k), None)
        else:
            # Desperate attempt for any time channel
            de_key = next((k for k in keys if "time" in k), None)

    if de_key is None:
        de_key = next((k for k in keys if "AI_4" in k and "time" not in k), None)

    if de_key is None:
        raise ValueError(
            f"No suitable signal key found in '{filepath}'.\n"
            f"Available keys: {[k for k in keys if not k.startswith('__')]}"
        )

    # Extract Signal
    signal = data[de_key].flatten().astype(np.float32)

    # Unit Conversion (g to mm/s²)
    signal = signal * 9.81 * 1000

    # Detect Shaft Frequency
    shaft_freq_hz = 0.0
    rpm_key = next((k for k in keys if k.endswith("RPM")), None)

    if rpm_key:
        rpm = float(data[rpm_key].flatten()[0])
        shaft_freq_hz = rpm / 60.0
    else:
        # Parse from filename
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

# SECTION 5 - SIGNAL PROCESSING
# =============================

def apply_lowpass_filter(signal: np.ndarray, fs: float) -> np.ndarray:
    """
    Apply zero-phase Butterworth low-pass filter using Second-Order Sections.
    """
    nyquist = fs / 2.0
    # Normalize cutoff frequency
    wn = min(LOWPASS_CUTOFF_HZ, nyquist * 0.9999) / nyquist

    # Use output='sos'
    sos = butter(LOWPASS_ORDER, wn, btype="lowpass", output='sos')

    # Apply zero-phase filtering using SOS
    return sosfiltfilt(sos, signal)

def compute_order_spectrum(segment: np.ndarray, fs: float, shaft_freq_hz: float) -> np.ndarray:
    """
    Compute one envelope order spectrum feature vector from a signal window.
    """
    seg_len = len(segment)

    analytic_signal = hilbert(segment)
    envelope = np.abs(analytic_signal)
    envelope = detrend_signal(envelope)

    win = hamming(seg_len)
    windowed = envelope * win
    data_freq = (seg_len / np.sum(win)) * np.abs(fft(windowed)) / (seg_len / 2)

    f_axis = np.linspace(0, 1, num=seg_len // 2 + 1, endpoint=True) * fs / 2
    f_axis = f_axis[1:]
    data_freq = data_freq[1: seg_len // 2 + 1]

    order_axis = f_axis / shaft_freq_hz

    order_vector = np.arange(ORDER_LOW, ORDER_HIGH + ORDER_STEP / 2, ORDER_STEP)
    order_amp = np.zeros(len(order_vector))
    for k, order in enumerate(order_vector):
        band = (order_axis >= order - ORDER_STEP / 2) & \
               (order_axis <= order + ORDER_STEP / 2)
        if np.any(band):
            order_amp[k] = np.sqrt(np.mean(data_freq[band] ** 2))

    max_amp = np.max(order_amp)
    if max_amp > 0:
        order_amp = order_amp / max_amp

    return order_amp.astype(np.float32)

def _show_diagnostic_plots(raw_seg, filtered_seg, seg, feat, fs, shaft_freq_hz):
    """
    Display four sequential blocking diagnostic plots, one per stage.
    Close each window to advance. Stop if graph is inaccurate.
    """
    t            = np.arange(len(raw_seg)) / fs
    order_vector = np.arange(ORDER_LOW, ORDER_HIGH + ORDER_STEP / 2, ORDER_STEP)

    # Plot 1: Raw signal
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, raw_seg, linewidth=0.5, color="steelblue")
    ax.set_title("Diagnostic 1/4 — Raw signal (after import & detrend)", fontsize=11)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    plt.tight_layout()
    print("\n[Diagnostic 1/4] Raw signal")
    plt.show()

    # Plot 2: Filtered signal
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(t, filtered_seg, linewidth=0.5, color="darkorange")
    ax.set_title(f"Diagnostic 2/4 — After low-pass filter "
                 f"(cutoff ≈ {LOWPASS_CUTOFF_HZ} Hz, order {LOWPASS_ORDER})", fontsize=11)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    plt.tight_layout()
    print("[Diagnostic 2/4] Filtered signal")
    plt.show()

    # Plot 3: Hilbert envelope
    envelope = np.abs(hilbert(seg))
    fig, ax  = plt.subplots(figsize=(13, 3))
    ax.plot(t, seg,      linewidth=0.3, alpha=0.5, color="darkorange",
            label="Filtered window")
    ax.plot(t, envelope, linewidth=0.8, color="crimson", label="Envelope")
    ax.set_title("Diagnostic 3/4 — Hilbert transform: envelope signal", fontsize=11)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("Amplitude")
    ax.legend(loc="upper right")
    plt.tight_layout()
    print("[Diagnostic 3/4] Hilbert envelope")
    plt.show()

    # Plot 4: Order spectrum (feature vector)
    fig, ax = plt.subplots(figsize=(13, 3))
    ax.plot(order_vector, feat, linewidth=0.8, color="seagreen")
    ax.set_title(f"Diagnostic 4/4 - Envelope order spectrum "
        f"(shaft freq = {shaft_freq_hz:.2f} Hz, {len(feat)} bins)", fontsize=11)
    ax.set_xlabel("Order (× shaft frequency)")
    ax.set_ylabel("Normalised RMS amplitude")
    plt.tight_layout()
    print("[Diagnostic 4/4] Order spectrum")
    plt.show()

def process_and_split_file(signal: np.ndarray, fs: float, shaft_freq_hz: float,
                           run_diagnostics: bool = False):
    """
    Process one file and return windows split into train / val / test.
    Includes signal cleaning: discards windows with signal > 10000.
    """
    signal_detrended = detrend_signal(signal)
    signal_filtered = apply_lowpass_filter(signal_detrended, fs)

    window_size = int(WINDOW_SECONDS * fs)
    stride = int(STRIDE_SECONDS * fs)
    N = len(signal_filtered)

    if N < window_size:
        print(f"  [WARNING] Signal too short for even one window "
              f"(length={N}, need={window_size}). Skipping file.")
        return [], [], []

    # Region boundaries (sample indices)
    t1 = int(N * TRAIN_SPLIT)  # end of train region
    t2 = int(N * (TRAIN_SPLIT + VAL_SPLIT))  # end of val region

    # Show diagnostics from the first window before full processing
    if run_diagnostics:
        first_window = signal_filtered[:window_size]

        # If the first window is > 10000, plot it anyway for visualization but discard.
        if np.any(first_window > 10000):
            print(f"  [Note] First window contains data > 10000. Showing for diagnostics.")

        first_feat = compute_order_spectrum(first_window, fs, shaft_freq_hz)
        _show_diagnostic_plots(raw_seg=signal_detrended[:window_size],
            filtered_seg=signal_filtered[:window_size], seg=first_window,
            feat=first_feat, fs=fs, shaft_freq_hz=shaft_freq_hz)

    def windows_in_region(start_pos: int, end_pos: int) -> list:
        """
        Generate overlapping windows that fit entirely within [start_pos, end_pos).
        Windows starting at start_pos, start_pos+stride, ...
        Skips windows containing any sample > 10000.
        """
        feats = []
        s = start_pos
        while s + window_size <= end_pos:
            seg = signal_filtered[s: s + window_size]

            # Disregard any sample that has a signal above the value of 10000
            if np.any(seg > 10000):
                s += stride
                continue

            feat = compute_order_spectrum(seg, fs, shaft_freq_hz)
            feats.append(feat)
            s += stride
        return feats

    train_feats = windows_in_region(0, t1)
    val_feats = windows_in_region(t1, t2)
    test_feats = windows_in_region(t2, N)

    return train_feats, val_feats, test_feats

def process_file(signal: np.ndarray, fs: float, shaft_freq_hz: float,
                 run_diagnostics: bool = False) -> list:
    """
    Process one file and return all feature windows (no split).
    """
    signal_detrended = detrend_signal(signal)
    signal_filtered  = apply_lowpass_filter(signal_detrended, fs)

    window_size = int(WINDOW_SECONDS * fs)
    N           = len(signal_filtered)

    if N < window_size:
        print(f"  [WARNING] Signal too short for one window. Skipping.")
        return []

    if run_diagnostics:
        first_window = signal_filtered[:window_size]
        first_feat   = compute_order_spectrum(first_window, fs, shaft_freq_hz)
        _show_diagnostic_plots(raw_seg = signal_detrended[:window_size],
            filtered_seg = signal_filtered[:window_size], seg = first_window,
            feat = first_feat, fs = fs, shaft_freq_hz= shaft_freq_hz)

    feats = []
    s     = 0
    while s + window_size <= N:
        seg  = signal_filtered[s: s + window_size]
        feat = compute_order_spectrum(seg, fs, shaft_freq_hz)
        feats.append(feat)
        s   += window_size   # non-overlapping for inference

    return feats

# SECTION 6 - DATASET
# ===================

class BearingFaultDataset(Dataset):
    def __init__(self, features: list, labels: np.ndarray,
                 class_names: list, label_encoder):
        self.X             = torch.tensor(np.stack(features), dtype=torch.float32)
        self.y             = torch.tensor(labels, dtype=torch.long)
        self.class_names   = class_names
        self.label_encoder = label_encoder

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def _process_source(files, loader_fn, fs, shaft_freq_arg,
                    diagnostics_shown: bool, run_diagnostics: bool,
                    source_tag: str):
    """
    Iterate over files from one data source, collect train/val/test feature lists and raw labels.
    """
    train = {"feats": [], "labels": []}
    val   = {"feats": [], "labels": []}
    test  = {"feats": [], "labels": []}
    skipped = []

    for fpath in files:
        label = parse_compound_label(fpath.stem)
        if label is None:
            skipped.append(fpath.name)
            continue

        try:
            result = loader_fn(str(fpath))
        except Exception as e:
            print(f"  [ERROR] Failed to load '{fpath.name}': {e}")
            continue

        if isinstance(result, tuple):
            signal, shaft_freq = result
        else:
            signal     = result
            shaft_freq = shaft_freq_arg

        print(f"  {source_tag} {fpath.name}  →  '{label}'  "
              f"(shaft: {shaft_freq:.2f} Hz)")

        run_diag = run_diagnostics and not diagnostics_shown
        tr, va, te = process_and_split_file(signal, fs, shaft_freq, run_diag)

        if run_diag and (tr or va or te):
            diagnostics_shown = True

        n_tr, n_va, n_te = len(tr), len(va), len(te)
        print(f"           Windows → train: {n_tr}  val: {n_va}  test: {n_te}")

        train["feats"].extend(tr); train["labels"].extend([label] * n_tr)
        val["feats"].extend(va);   val["labels"].extend([label] * n_va)
        test["feats"].extend(te);  test["labels"].extend([label] * n_te)

    return train, val, test, diagnostics_shown, skipped

def load_and_split_data(adams_dir: str = None, cwru_dir:  str = None,
                        run_diagnostics: bool = True):
    """
    Load all data files, apply the full processing pipeline, and return
    three ready-to-use PyTorch Datasets (train, val, test).
    Each file's windows are split positionally (45/10/45) so that no
    overlapping window ever appears in more than one split.
    """
    all_train = {"feats": [], "labels": []}
    all_val   = {"feats": [], "labels": []}
    all_test  = {"feats": [], "labels": []}
    all_raw_labels = []
    diagnostics_shown = False

    # Adams .tab files
    adams_path = Path(adams_dir) if adams_dir else None
    if adams_path and adams_path.exists():
        tab_files = sorted(adams_path.glob("*.tab"))
        print(f"[Adams] Found {len(tab_files)} .tab file(s)\n")
        tr, va, te, diagnostics_shown, skipped = _process_source(
            files            = tab_files,
            loader_fn        = load_tab_file,
            fs               = ADAMS_FS,
            shaft_freq_arg   = ADAMS_SHAFT_FREQ_HZ,
            diagnostics_shown= diagnostics_shown,
            run_diagnostics  = run_diagnostics,
            source_tag       = "[Adams]",
        )
        for bucket, src in [(all_train, tr), (all_val, va), (all_test, te)]:
            bucket["feats"].extend(src["feats"])
            bucket["labels"].extend(src["labels"])
        all_raw_labels.extend(tr["labels"] + va["labels"] + te["labels"])

        if skipped:
            print(f"\n  [WARNING] {len(skipped)} Adams file(s) skipped "
                  f"(no label match): {skipped}")
            print("  → Add entries to TYPE_RULES/SIZE_RULES or MANUAL_LABEL_MAP.\n")
    elif adams_dir:
        print(f"[Adams] Directory '{adams_dir}' not found - skipping.")

    # CWRU .mat files
    cwru_path = Path(cwru_dir) if cwru_dir else None
    if cwru_path and cwru_path.exists():
        mat_files = sorted(cwru_path.glob("*.mat"))
        print(f"\n[CWRU]  Found {len(mat_files)} .mat file(s)\n")
        tr, va, te, diagnostics_shown, skipped = _process_source(
            files            = mat_files,
            loader_fn        = load_mat_file,
            fs               = CWRU_FS,
            shaft_freq_arg   = None,   # read from file
            diagnostics_shown= diagnostics_shown,
            run_diagnostics  = run_diagnostics,
            source_tag       = "[CWRU] ",
        )
        for bucket, src in [(all_train, tr), (all_val, va), (all_test, te)]:
            bucket["feats"].extend(src["feats"])
            bucket["labels"].extend(src["labels"])
        all_raw_labels.extend(tr["labels"] + va["labels"] + te["labels"])

        if skipped:
            print(f"\n  [WARNING] {len(skipped)} CWRU file(s) skipped "
                  f"(no label match): {skipped}")
            print("  → Add entries to TYPE_RULES/SIZE_RULES or MANUAL_LABEL_MAP.\n")
    elif cwru_dir:
        print(f"[CWRU]  Directory '{cwru_dir}' not found - skipping.")

    if not all_raw_labels:
        raise ValueError(
            "No labeled samples were loaded.\n"
            "Check that:\n"
            "  1. DATA_DIR paths exist and contain files.\n"
            "  2. Filenames match TYPE_RULES + SIZE_RULES or MANUAL_LABEL_MAP."
        )

    # Fit label encoder on all labels across all splits
    label_encoder = LabelEncoder()
    label_encoder.fit(all_raw_labels)
    class_names = list(label_encoder.classes_)

    def make_dataset(bucket: dict, split_name: str) -> BearingFaultDataset:
        if not bucket["feats"]:
            raise ValueError(
                f"The {split_name} split is empty. "
                f"Check that files are long enough to produce windows in "
                f"all three time regions (train/val/test)."
            )
        encoded = label_encoder.transform(bucket["labels"])
        return BearingFaultDataset(bucket["feats"], encoded,
                                   class_names, label_encoder)

    train_dataset = make_dataset(all_train, "training")
    val_dataset   = make_dataset(all_val,   "validation")
    test_dataset  = make_dataset(all_test,  "test")

    # Summary
    n_train = len(train_dataset)
    n_val   = len(val_dataset)
    n_test  = len(test_dataset)
    n_total = n_train + n_val + n_test
    print(f"\n{'='*60}")
    print(f"[Dataset] Total windows : {n_total}")
    print(f"          Train         : {n_train}  ({n_train/n_total*100:.1f}%)")
    print(f"          Validation    : {n_val}  ({n_val/n_total*100:.1f}%)")
    print(f"          Test          : {n_test}  ({n_test/n_total*100:.1f}%)")
    print(f"          Classes ({len(class_names)}): {class_names}")
    print(f"{'='*60}\n")

    return train_dataset, val_dataset, test_dataset, class_names, label_encoder

# SECTION 7 - MODEL
# =================

class FeedForwardNet(nn.Module):
    """
    Feedforward neural network for bearing fault classification.
    CrossEntropyLoss is used (no explicit Softmax in forward pass -
    it is included in PyTorch's CrossEntropyLoss).
    """

    def __init__(self, input_size: int, hidden_layers: list,
                 num_classes: int, dropout_rate: float = 0.3):
        super().__init__()
        layers  = []
        in_size = input_size
        for h in hidden_layers:
            layers += [nn.Linear(in_size, h), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_size = h
        layers.append(nn.Linear(in_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# SECTION 8 - TRAINING & EVALUATION
# =================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
        correct    += (out.argmax(dim=1) == y_batch).sum().item()
        total      += len(y_batch)
    return total_loss / total, correct / total

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_targets     = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        out  = model(X_batch)
        loss = criterion(out, y_batch)
        total_loss += loss.item() * len(y_batch)
        preds       = out.argmax(dim=1)
        correct    += (preds == y_batch).sum().item()
        total      += len(y_batch)
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

    # Handle Standard (Type_Size)
    parts = label_str.split('_')
    if not parts: return label_str

    # Extract known sizes
    known_sizes = [s for _, s in SIZE_RULES]

    if parts[-1] in known_sizes:
        # Remove the last part and rejoin
        return "_".join(parts[:-1])

    # Fallback: return original label if no rules matched
    return label_str


def show_training_results(train_losses, val_losses, train_accs, val_accs,
                          targets, preds, class_names, label_encoder, save_dir):
    """
    Generates and saves plots:
    1. Training & Validation Curves.
    2. Full Confusion Matrix.
    3. Aggregated Confusion Matrix (By Fault Type).
    4. Zoomed-in Confusion Matrices.
    """

    # Decode integer targets/preds back to string labels
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

    plt.suptitle("Training & Validation History", fontsize=13, y=1.01)
    plt.tight_layout()

    save_path1 = os.path.join(save_dir, "training_curves.png")
    plt.savefig(save_path1, dpi=150)
    print(f"[Plot] Training curves saved to '{save_path1}'")
    plt.close()

    # Plot 2: Full CM
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_title("Confusion Matrix - Test Set (Full Detail)")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    plt.tight_layout()

    save_path2 = os.path.join(save_dir, "confusion_matrix.png")
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
                xticklabels=unique_types, yticklabels=unique_types, ax=ax)
    ax.set_title("Confusion Matrix - Aggregated by Fault Type")
    ax.set_xlabel("Predicted Type")
    ax.set_ylabel("True Type")
    plt.tight_layout()

    save_path3 = os.path.join(save_dir, "confusion_matrix_by_type.png")
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
        # Skip if there is only 1 class
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
        ax.set_title(f"Confusion Matrix - Zoomed: {parent_type}")
        ax.set_xlabel("Predicted Size/Variant")
        ax.set_ylabel("True Size/Variant")
        plt.tight_layout()

        save_path_zoom = os.path.join(save_dir, f"confusion_matrix_zoomed_{safe_type_name}.png")
        plt.savefig(save_path_zoom, dpi=150)
        print(f"[Plot] Zoomed matrix for '{parent_type}' saved to '{save_path_zoom}'")
        plt.close()

    # Show all plots
    plt.show(block=True)

# SECTION 9 - MAIN
# ================

def main():
    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}\n")

    # Load and process data
    print("[Loading]\n")
    train_dataset, val_dataset, test_dataset, class_names, label_encoder = \
        load_and_split_data(adams_dir = ADAMS_DATA_DIR, cwru_dir = CWRU_DATA_DIR, run_diagnostics = True)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE)

    # Build model
    input_size  = get_feature_size()
    num_classes = len(class_names)
    model       = FeedForwardNet(input_size, HIDDEN_LAYERS,
                                 num_classes, DROPOUT_RATE).to(device)
    n_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Model] Input: {input_size}  |  Classes: {num_classes}  |  "
          f"Parameters: {n_params:,}")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

    # Training loop
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    best_val_loss = float("inf")

    print("\n[Training]\n")
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
            torch.save(model.state_dict(),
                       os.path.join(MODEL_SAVE_PATH, "best_model.pt"))

    # Test evaluation
    model.load_state_dict(
        torch.load(os.path.join(MODEL_SAVE_PATH, "best_model.pt"), map_location=device, weights_only=True))
    test_loss, test_acc, preds, targets = evaluate(model, test_loader, criterion, device)

    print(f"\n[Test]  Loss: {test_loss:.4f}  |  Accuracy: {test_acc:.3f}\n")
    print("[Classification Report]")
    print(classification_report(targets, preds, target_names=class_names, zero_division=0))

    # Results plots
    show_training_results(train_losses, val_losses, train_accs, val_accs,
                          targets, preds, class_names, label_encoder, MODEL_SAVE_PATH)

    # Save artifacts
    joblib.dump(label_encoder, os.path.join(MODEL_SAVE_PATH, "label_encoder.pkl"))
    joblib.dump({
        "input_size":          input_size,
        "hidden_layers":       HIDDEN_LAYERS,
        "dropout_rate":        DROPOUT_RATE,
        "num_classes":         num_classes,
        "class_names":         class_names,
        "adams_shaft_freq_hz": ADAMS_SHAFT_FREQ_HZ,
        "lowpass_cutoff_hz":   LOWPASS_CUTOFF_HZ,
        "lowpass_order":       LOWPASS_ORDER,
        "window_seconds":      WINDOW_SECONDS,
        "order_low":           ORDER_LOW,
        "order_high":          ORDER_HIGH,
        "order_step":          ORDER_STEP,
    }, os.path.join(MODEL_SAVE_PATH, "config.pkl"))

    print(f"\n[Done] Model and config saved to '{MODEL_SAVE_PATH}/'")

if __name__ == "__main__":
    main()