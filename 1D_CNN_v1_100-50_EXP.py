if __name__ == "__main__": # packets
    import time
    t0 = time.perf_counter()
    # =========================
    # Standard Library
    # =========================
    import os
    # import random
    # import pickle
    import copy
    # =========================
    # Numerical & Data Handling
    # =========================
    import numpy as np
    import pandas as pd
    import scipy.io as scio

    # =========================
    # Signal Processing
    # =========================
    from scipy import signal, ndimage
    # from scipy.signal import hilbert
    # from numpy.fft import fft
    # =========================
    # Machine Learning (Sklearn)
    # =========================
    # from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import classification_report, confusion_matrix
    # from sklearn.preprocessing import MinMaxScaler
    # from sklearn.svm import SVC, OneClassSVM
    # =========================
    # Deep Learning
    # =========================
    import torch
    import torch.nn as nn
    # import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    # =========================
    # Visualization
    # =========================
    import matplotlib.pyplot as plt
    import seaborn as sns
    # =========================
    # Visualization t-SNE
    # =========================
    from sklearn.manifold import TSNE
    import matplotlib.colors as mcolors
    from matplotlib.lines import Line2D
    import colorsys
    import matplotlib.cm as mpl_cm
    import glasbey





# =========================================================
# Loading + preprocessing
# Mostly Pais code
# =========================================================
def preprocess_signal(x: np.ndarray) -> np.ndarray:
        x = x.astype(np.float32).squeeze()
        x = signal.detrend(x) * 9.81 * 1000
    
        b, a = signal.butter(8, 0.8, btype='lowpass')
        x = signal.filtfilt(b, a, x)
    
        return x.astype(np.float32)

def load_tab(tab_path: str) -> np.ndarray | None:
    # Given by Pai but changed fron wv to scipy.signal
    df = pd.read_csv(tab_path, names=["Data"])
    df = df.drop(index=[0, 1, 2, 3])
    df = pd.concat([df, df["Data"].str.split("\t", expand=True)], axis=1)
    df = df.drop(columns="Data")
    
    data = df.iloc[2:, 1].values.astype(np.float32)
    
    return preprocess_signal(data)

def load_tabfile_class_signals(data_folder: str):
    # find and sort files that end with .tab
    filelist = sorted(
        f for f in os.listdir(data_folder)
        if f.lower().endswith(".tab")
    )

    signals = []
    class_names = []
    # load each file, preprocess, and store signal + class name
    for filename in filelist:
        filepath = os.path.join(data_folder, filename)
        sig = load_tab(filepath)

        signals.append(sig)
        class_names.append(os.path.splitext(filename)[0])

    return signals, class_names


def load_mat(mat_path: str) -> np.ndarray | None:
    data = scio.loadmat(mat_path)

    key = "Data1_AI_4_AI_4_minus_Housing_Y"
    if key not in data:
        print(f"Skipping {os.path.basename(mat_path)}: key '{key}' not found")
        return None

    return preprocess_signal(data[key])

def load_matfile_class_signals(data_folder: str) -> dict[str, np.ndarray]:
    accdata_dict = {}

    class_folders = sorted(
        f for f in os.listdir(data_folder)
        if os.path.isdir(os.path.join(data_folder, f))
    )

    for class_name in class_folders:
        class_path = os.path.join(data_folder, class_name)

        filelist = sorted(
            f for f in os.listdir(class_path)
            if f.lower().endswith(".mat")
        )

        class_signals = []
        for filename in filelist:
            filepath = os.path.join(class_path, filename)
            sig = load_mat(filepath)
            if sig is not None:
                class_signals.append(sig)

        if class_signals:
            accdata_dict[class_name] = np.concatenate(class_signals).astype(np.float32)

    return accdata_dict, list(accdata_dict.keys())
# =========================================================
# Split + window helpers
# =========================================================
def center_crop_to_length(x: np.ndarray, target_len: int) -> np.ndarray:
    start = (len(x) - target_len) // 2
    return x[start:start + target_len]

def split_signal_sections(
    x: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
):
    
    
    total = train_ratio + val_ratio + test_ratio

    if not np.isclose(total, 1.0):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.0")
    
    # Split signal into train/val/test sections
    n = len(x)
    
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    train = x[:n_train]
    val = x[n_train:n_train + n_val]
    test = x[n_train + n_val:]

    # Normalize using training set stats
    mean = train.mean()
    std = train.std()

    train = (train - mean) / std
    val = (val - mean) / std
    test = (test - mean) / std

    return train, val, test

def filter_windows(X: np.ndarray, max_threshold = 1e3) -> np.ndarray:
    # Remove windows where max(abs(window)) exceeds 1e3

    max_vals = np.max(np.abs(X), axis=1)
    keep_idx = max_vals <= max_threshold

    return X[keep_idx]

def make_windows(x: np.ndarray, window_size: int, stride: int) -> np.ndarray:
    if len(x) < window_size:
        return np.empty((0, window_size), dtype=np.float32)

    starts = range(0, len(x) - window_size + 1, stride)

    windows = [
        x[start:start + window_size]
        for start in starts
    ]

    return np.stack(windows).astype(np.float32)

def shuffle_xy(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


# =========================================================
# Dataset builder
# =========================================================
def build_datasets_fileclasses(
    signals: list[np.ndarray],
    class_names: list[str],
    sim_classes: int,
    window_size: int,
    stride: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    random_seed=None,
) -> dict[str, np.ndarray | list[str] | dict[str, int]]:
    
    rng = np.random.default_rng(random_seed)

    n_classes = len(signals)
    print(n_classes)
    class_to_label = {name: i for i, name in enumerate(class_names)}
    
    # Use same usable length for every class to avoid bias
    shared_len = min(len(sig) for sig in signals[sim_classes:])
    
    print("Signals lengths:", [len(sig) for sig in signals])
    print("Number of classes:", n_classes)
    print("Class names:", class_names)
    print("Shared usable length:", shared_len)

    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    X_test_list, y_test_list = [], []

    print("sim classes")
    label = 0
    # crop to shared length (center crop)
    for class_name in class_names[:sim_classes]:
        print(class_name, label)

        signal = center_crop_to_length(signals[label], shared_len)
        # split into train/val/test sections and normalize
        train_sig, val_sig, test_sig = split_signal_sections(
            signal,
            train_ratio,
            val_ratio,
            test_ratio,
        )

        # create windows for each section
        X_train_c = make_windows(train_sig, window_size, stride)
        X_val_c = make_windows(val_sig, window_size, stride)
        X_test_c = make_windows(test_sig, window_size, stride)

        # filter out windows with extreme max values
        X_train_c = filter_windows(X_train_c)
        X_val_c = filter_windows(X_val_c)
        X_test_c = filter_windows(X_test_c)



        print(
            f"{class_name}: "
            f"windows train/val/test = "
            f"{len(X_train_c)}/{len(X_val_c)}/{len(X_test_c)}"
        )

        # For each signal, append windows (xdata) and corresponding labels (ydata) to lists
        X_train_list.append(X_train_c)
        y_train_list.append(np.full(len(X_train_c), label, dtype=np.int64))

        X_val_list.append(X_val_c)
        y_val_list.append(np.full(len(X_val_c), label, dtype=np.int64))

        X_test_list.append(X_test_c)
        y_test_list.append(np.full(len(X_test_c), label, dtype=np.int64))
        label += 1

    # Put all EXP classes at the in test set and not train/val
    print("exp classes")
    for class_name in class_names[sim_classes:]:
        print(class_name,label)

        signal = signals[label]
        # Normalize entire signal using its own stats (since we won't train on it)
        mean = signal.mean()
        std = signal.std()

        signal = (signal - mean) / std
        # create windows
        X_test_c = make_windows(signal, window_size, stride)

        print(
            f"{class_name}: "
            f"windows test = "
            f"{len(X_test_c)}"
        )

        X_test_list.append(X_test_c)
        y_test_list.append(np.full(len(X_test_c), label, dtype=np.int64))
        label += 1


    # After processing all classes, concatenate lists into final arrays
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)

    X_val = np.concatenate(X_val_list, axis=0)
    y_val = np.concatenate(y_val_list, axis=0)

    X_test = np.concatenate(X_test_list, axis=0)
    y_test = np.concatenate(y_test_list, axis=0)

    # Add channel dimension: (samples, channels, window_size)
    X_train = X_train[:, np.newaxis, :]
    X_val = X_val[:, np.newaxis, :]
    X_test = X_test[:, np.newaxis, :]

    # Shuffle datasets to ensure random distribution of classes
    # 
    X_train, y_train = shuffle_xy(X_train, y_train, rng)
    X_val, y_val = shuffle_xy(X_val, y_val, rng)
    X_test, y_test = shuffle_xy(X_test, y_test, rng)

    return {
        "X_train": X_train.astype(np.float32),
        "y_train": y_train,
        "X_val": X_val.astype(np.float32),
        "y_val": y_val,
        "X_test": X_test.astype(np.float32),
        "y_test": y_test,
        "class_names": class_names,
        "class_to_label": class_to_label,
    }


# =========================================================
# Main
# =========================================================
# Global parameters
# Windowsize is how the signal is split into smaller segments for the CNN to process.
# Stride is how much the window moves for the next segment. 
# So overlapp is 50 % right now.

WINDOW_SIZE = 5000 # half second
STRIDE = 2500 # quarter second

TRAIN_RATIO = 0.45
VAL_RATIO = 0.10
TEST_RATIO = 0.45

RANDOM_SEED = None

# loop for average results across multiple runs
n_runs = 1

cms = []
cms_percent = []
accs = []
epochs = []

for runs in range(n_runs):
    DATA_FOLDER_SIM = r'Proj\Sim_data\Adams Simulation Data'
    DATA_FOLDER_EXP = r'Proj\Sim_data\EXP dataset'

    # Load signals and class names
    SIM_signals, SIM_class_names = load_tabfile_class_signals(DATA_FOLDER_SIM)
    EXP_signals, EXP_class_names = load_matfile_class_signals(DATA_FOLDER_EXP)
    for signal_list in EXP_signals.values():
        SIM_signals.append(signal_list)
    signals  = SIM_signals
    class_names = SIM_class_names + EXP_class_names
    sim_classes = len(SIM_class_names)
    # print("SIM class names:", SIM_class_names)
    # print("EXP class names:", EXP_class_names)
    # print("Number of SIM classes:", len(SIM_class_names))
    # print("Number of EXP classes:", len(EXP_class_names))
    # # print(SIM_signals)
    # class_names = np.concatenate([SIM_class_names, EXP_class_names])
    # print("Data type of SIM signals:", len(SIM_signals))
    # print("Data type of EXP signals:", type(EXP_signals))
    # make datasets
    num_classes = len(class_names)
    datasets = build_datasets_fileclasses(
        signals=signals,
        class_names=class_names,
        sim_classes=sim_classes,
        window_size=WINDOW_SIZE,
        stride=STRIDE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        random_seed=RANDOM_SEED,
    )

    # -------------------------------------------------
    # extract datasets
    # -------------------------------------------------

    X_train = datasets["X_train"]
    y_train = datasets["y_train"]
    X_val = datasets["X_val"]
    y_val = datasets["y_val"]
    X_test = datasets["X_test"]
    y_test = datasets["y_test"]

    print("\nFinal dataset shapes:")
    print("Train:", X_train.shape, y_train.shape)
    print("Val:  ", X_val.shape, y_val.shape)
    print("Test: ", X_test.shape, y_test.shape)
    

    if __name__ == "__main__": # CNN

        # -------------------------------------------------
        # Build dataset as per
        # -------------------------------------------------
        class BearingDataset(Dataset):
            def __init__(self, X, y):
                X = np.asarray(X, dtype=np.float32)
                y = np.asarray(y, dtype=np.int64)
                if X.ndim == 2:
                    X = X[:, np.newaxis, :]
                self.X = torch.from_numpy(X)
                self.y = torch.from_numpy(y)
            def __len__(self):
                return len(self.y)
            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]


        # -------------------------------------------------
        # Create datasets
        # -------------------------------------------------

        train_dataset = BearingDataset(X_train, y_train)
        val_dataset   = BearingDataset(X_val, y_val)
        test_dataset  = BearingDataset(X_test, y_test)

        # -------------------------------------------------
        # Create dataloaders
        # -------------------------------------------------

        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader   = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)

        class Bearing1DCNN(nn.Module):
            def __init__(self, num_classes=num_classes):
                super().__init__()
                # Kernel layers with batch norm, ReLU, and max pooling
                # Kernel sizes 100 and 50 chosen based on literature
                # maxpooling layers because thats what you use?
                self.features = nn.Sequential(
                    nn.Conv1d(1, 64, kernel_size=100, stride=2, padding=50),
                    # nn.BatchNorm1d(64),
                    nn.ReLU(),
                    nn.MaxPool1d(4),

                    nn.Conv1d(64, 32, kernel_size=50, stride=1, padding=25),
                    # nn.BatchNorm1d(32),
                    nn.ReLU(),
                    nn.MaxPool1d(4),
                )
                
                self.classifier = nn.Sequential(
                    nn.Flatten(),
                    # nn.Dropout(0.5),
                    nn.LazyLinear(100),
                    nn.ReLU(),
                    nn.Linear(100, num_classes)
                )

            def forward(self, x):
                x = self.features(x)
                x = self.classifier(x)
                return x
        # use "cuda" (GPU) if available, otherwise CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Using device:", device)
        print(f"Run {runs}")
        t1 = time.perf_counter()
        print(f"Total runtime: {t1 - t0:.4f} seconds")

        model = Bearing1DCNN(num_classes=num_classes).to(device)
        criterion = nn.CrossEntropyLoss()
        # change lr (learning rate) and weight decay as needed
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-5)

        # training function that returns loss and accuracy for the epoch
        def train_one_epoch(model, loader, criterion, optimizer, device):
            model.train()

            running_loss = 0.0
            correct = 0
            total = 0

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                optimizer.zero_grad()
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)

                loss.backward()
                optimizer.step()

                running_loss += loss.item() * X_batch.size(0)
                preds = outputs.argmax(dim=1)

                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
            epoch_loss = running_loss / total
            epoch_acc = correct / total

            return epoch_loss, epoch_acc

        @torch.no_grad()
        def extract_embeddings(model, loader, device):
            model.eval()
            embeddings = []
            labels = []

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)

                x = model.features(X_batch)
                x = model.classifier[:-1](x)   # output before final Linear layer

                embeddings.append(x.cpu().numpy())
                labels.append(y_batch.numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)

            return embeddings, labels

        @torch.no_grad()
        # evaluation function that returns loss, accuracy, and predictions for confusion matrix
        def evaluate(model, loader, criterion, device):
            model.eval()
            running_loss = 0.0
            correct = 0
            total = 0

            all_preds = []
            all_labels = []

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                running_loss += loss.item() * X_batch.size(0)

                preds = outputs.argmax(dim=1)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())

            epoch_loss = running_loss / total
            epoch_acc = correct / total
            return epoch_loss, epoch_acc, all_labels, all_preds

        epoc_hist = []
        train_loss_hist = []
        val_loss_hist = []
        train_acc_hist = []
        val_acc_hist = []

        num_epochs = 1000
        best_val_acc = 0.0
        best_val_loss = float('inf')
        patience = 20
        # when val loss does not improve for "patience" epochs, stop training to prevent overfitting
        counter = 0
        best_model_state = None

        for epoch in range(num_epochs):

            train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            epoc_hist.append(epoch)

            train_loss_hist.append(train_loss)
            val_loss_hist.append(val_loss)
            train_acc_hist.append(train_acc)
            val_acc_hist.append(val_acc)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "Proj\\Models\\best_bearing_1dcnn_k100_k50_Adams_Simulation_Data.pt")
            else:
                counter += 1 # patience counter

            if counter >= patience:
                print("Early stopping triggered")
                break

    if __name__ == "__main__": # Final evaluation and plotting
        # Restore best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)


        test_loss, test_acc, y_true, y_pred = evaluate(model, test_loader, criterion, device)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Acc:  {test_acc:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=class_names))

        cm = confusion_matrix(y_true, y_pred)
        cms.append(cm)

        # Row-wise percentages
        cm_percent = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100
        cms_percent.append(cm_percent)

        # Create annotation labels: percent + count
        annot_labels = np.empty_like(cm).astype(str)
        acc = (np.array(y_true) == np.array(y_pred)).mean()
        accs.append(acc)
        epochs.append(epoc_hist[-1])
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot_labels[i, j] = f"{cm_percent[i, j]:.1f}%\n({cm[i, j]})"

        if n_runs == 1:
            # # Plot
            # plt.figure(figsize=(8, 6))

            # sns.heatmap(
            #     cm_percent,
            #     annot=annot_labels,
            #     fmt="",
            #     cmap="Blues",
            #     xticklabels=class_names,
            #     yticklabels=class_names,
            #     cbar_kws={"label": "Percentage (%)"}
            # )

            # plt.xlabel("Predicted Labels")
            # plt.ylabel("True Labels")
            # plt.title("Confusion Matrix (%) with Counts")
            # plt.tight_layout()
            # plt.savefig("Proj\\Figures\\CM_CWRU_3x.eps")
            # plt.show()

            fig, axes = plt.subplots(1, 2, figsize=(12, 5)) 

            axes[0].plot(epoc_hist, train_loss_hist, label="Training loss")
            axes[0].plot(epoc_hist, val_loss_hist, label="Validation loss")
            axes[0].set_xlabel("Epochs")
            axes[0].set_ylabel("Loss")
            axes[0].set_title("Training and Validation Loss")
            axes[0].legend()

            axes[1].plot(epoc_hist, train_acc_hist, label="Training acc")
            axes[1].plot(epoc_hist, val_acc_hist, label="Validation acc")
            axes[1].set_xlabel("Epochs")
            axes[1].set_ylabel("Accuracy")
            axes[1].set_title("Training and Validation Accuracy")
            axes[1].legend()

            plt.tight_layout()
            plt.savefig("Proj\\Figures\\Adams Simulation Data\\ACC-LOSS_Sim_&_EXP_Data.png")

            # -------------------------------------------------
            # t-SNE plot 
            # -------------------------------------------------

            family_base_colors = {
                "Healthy": (1.0, 0.85, 0.0),      # yellow
                "IR": (1.0, 0.10, 0.0),           # red/orange
                "IR_03_OR": (0.05, 0.05, 0.05),   # grayscale / black-gray
                "IR_05_OR": (1.0, 0.0, 0.85),    # bright magenta
                "OR": (0.0, 0.20, 1.0),           # blue
                "RR": (0.0, 0.65, 0.1),           # green
                "EXP": (0.7, 0.7, 0.7),            # silver-gray
            }

            def get_fault_family(name):
                if "EXP" in name:
                    return "EXP"
                if name == "Healthy":
                    return "Healthy"
                if "IR" in name and "OR" in name:
                    if "IR_03" in name:
                        return "IR_03_OR"
                    if "IR_05" in name:
                        return "IR_05_OR"
                if name.startswith("IR"):
                    return "IR"
                if name.startswith("OR"):
                    return "OR"
                if name.startswith("RR"):
                    return "RR"
                return "Other"

            family_to_classes = {}
            for idx, name in enumerate(class_names):
                fam = get_fault_family(name)
                family_to_classes.setdefault(fam, []).append(idx)


            class_colors = {}

            for fam, indices in family_to_classes.items():

                n = len(indices)

                # Healthy: always yellow
                if fam == "Healthy":
                    for cls_idx in indices:
                        class_colors[cls_idx] = (1.0, 0.85, 0.0)
                    continue

                # IR_03_OR: grayscale, light to dark
                if fam == "IR_03_OR":
                    gray_values = np.linspace(0.0, 0.50, n)
                    for cls_idx, gray in zip(indices, gray_values):
                        class_colors[cls_idx] = (gray, gray, gray)
                    continue

                base_rgb = family_base_colors.get(fam, (0.5, 0.5, 0.5))
                h, s, v = colorsys.rgb_to_hsv(*base_rgb)

                for i, cls_idx in enumerate(indices):
                    hue_shift = (i - (n - 1) / 2) * 0.05
                    sat = 0.9
                    val = 0.4 + 0.6 * (i / max(1, n - 1))

                    r, g, b = colorsys.hsv_to_rgb((h + hue_shift) % 1.0, sat, val)
                    class_colors[cls_idx] = (r, g, b)

            embeddings, tsne_labels = extract_embeddings(model, train_loader, device)

            perplexity = min(30, len(embeddings) - 1)

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                learning_rate="auto",
                init="pca",
                random_state=RANDOM_SEED
            )

            embeddings_2d = tsne.fit_transform(embeddings)

            plt.figure(figsize=(16, 9))

            for class_idx, class_name in enumerate(class_names):
                idx = tsne_labels == class_idx
                plt.scatter(
                    embeddings_2d[idx, 0],
                    embeddings_2d[idx, 1],
                    color=class_colors[class_idx],
                    label=class_name,
                    edgecolors='k',
                    linewidths=0.2,
                    alpha=0.8
                )

            plt.title(f"t-SNE of 1D-CNN")
            plt.xlabel("Component 1")
            plt.ylabel("Component 2")
            # -------------------------------------------------
            # Grouped legend by fault family
            # -------------------------------------------------
            legend_handles = []

            family_order = ["Healthy", "IR", "IR_03_OR", "IR_05_OR", "OR", "RR", "EXP"]
            family_titles = {
                "Healthy": "Healthy",
                "IR": "IR faults",
                "IR_03_OR": "IR 03 + OR faults",
                "IR_05_OR": "IR 05 + OR faults",
                "OR": "OR faults",
                "RR": "RR faults",
                "EXP": "Exp data"
            }

            for fam in family_order:
                if fam not in family_to_classes:
                    continue
                
                # fake bold-ish group header
                legend_handles.append(
                    Line2D([], [], color="none", label=f"\n{family_titles[fam]}")
                )

                for class_idx in family_to_classes[fam]:
                    class_name = class_names[class_idx]

                    legend_handles.append(
                        Line2D(
                            [0], [0],
                            marker="o",
                            color="w",
                            label=f"  {class_name}",
                            markerfacecolor=class_colors[class_idx],
                            markersize=8
                        )
                    )

            plt.legend(
                handles=legend_handles,
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=True,
                fontsize=9
            )
            plt.tight_layout()
            plt.savefig(f"Proj\\Figures\\Adams Simulation Data\\tSNE_Sim_&_EXP_Data.png", bbox_inches='tight')
            

# -----------------------
# averages
# -----------------------
avg_cm = np.mean(cms, axis=0)
tot_cm = np.sum(cms, axis=0)   
avg_cm_pct = np.mean(cms_percent, axis=0)
avg_epochs = np.mean(epochs,axis=0)
print("Mean Accuracy:", np.mean(accs))
print("Std Accuracy :", np.std(accs))
print("Average epochs :", avg_epochs)
print(f"Shortest epoch run : {min(epochs)}")
print(f"Longest epoch run : {max(epochs)}")




annot = np.empty(avg_cm.shape, dtype=object)

for i in range(avg_cm.shape[0]):
    for j in range(avg_cm.shape[1]):
        annot[i, j] = (
            # f"{avg_cm_pct[i,j]:.1f}\n"
            # f"avg:{avg_cm[i,j]:.1f}\n"
            f"{tot_cm[i,j]}"
        )
plt.figure(figsize=(16,12))

sns.heatmap(
    avg_cm_pct,
    annot=annot,
    fmt="",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names,
    cbar_kws={"label":"Average Percentage %"}
)

plt.title(f"Mean Confusion Matrix Across {n_runs} Runs\n(% / avg count / total count)")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
plt.savefig(f"Proj\\Figures\\Adams Simulation Data\\CM_Adams_Sim_&_EXP_Data_{n_runs}_runs_nob.png", bbox_inches='tight')
t1 = time.perf_counter()
print(f"Total runtime: {t1 - t0:.4f} seconds with {n_runs} runs")
plt.show()