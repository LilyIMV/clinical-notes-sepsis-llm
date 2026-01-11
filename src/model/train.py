"""
train.py

Train and evaluate a BiLSTM sepsis prediction model.
Post-hoc interpretability via KernelSHAP (6h horizon only).

"""

# ======================================================
# IMPORTS
# ======================================================
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import shap
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score

from config import OUTPUT_DIR, NOTES_ONLY
from model import MyLSTM


# ======================================================
# SEEDING
# ======================================================
SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ======================================================
# DATASET
# ======================================================
class VisitSequenceWithLabelDataset(Dataset):
    def __init__(self, seqs, labels):
        self.seqs = seqs
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.seqs[idx], self.labels[idx]


# ======================================================
# COLLATE FUNCTION (VARIABLE LENGTH)
# ======================================================
def time_collate_fn(batch, max_len=200):
    seqs, labels = zip(*batch)
    lengths = [s.shape[0] for s in seqs]

    num_features = seqs[0].shape[1]
    max_length = min(max(lengths), max_len)

    order = np.argsort(lengths)[::-1]

    padded, sorted_labels, sorted_lengths = [], [], []

    for i in order:
        s = seqs[i][-max_length:]
        pad = max_length - s.shape[0]
        if pad > 0:
            s = np.vstack([s, np.zeros((pad, num_features), dtype=np.float32)])
        padded.append(s)
        sorted_labels.append(labels[i])
        sorted_lengths.append(min(lengths[i], max_length))

    return (
        torch.FloatTensor(np.stack(padded)),
        torch.LongTensor(sorted_lengths)
    ), torch.LongTensor(sorted_labels)


# ======================================================
# TRAIN / EVAL
# ======================================================
def run_epoch(model, loader, optimizer, criterion, device, train=True):
    model.train() if train else model.eval()

    total_loss, n = 0.0, 0
    y_true, y_prob = [], []

    with torch.enable_grad() if train else torch.no_grad():
        for (x, lengths), y in loader:
            x, lengths, y = x.to(device), lengths.to(device), y.to(device)

            logits = model((x, lengths))
            loss = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            probs = torch.softmax(logits, dim=1)[:, 1]

            total_loss += loss.item() * y.size(0)
            n += y.size(0)

            y_true.extend(y.cpu().numpy())
            y_prob.extend(probs.detach().cpu().numpy())

    return {
        "loss": total_loss / n,
        "auprc": average_precision_score(y_true, y_prob),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "accuracy": accuracy_score(y_true, (np.array(y_prob) >= 0.5).astype(int))
    }


# ======================================================
# KERNEL SHAP (FLATTENED TIME SERIES)
# ======================================================
def compute_kernel_shap(
    model, model_name, horizon, 
    val_seqs, test_seqs, 
    n_bg, n_ex, n_spl,
    variables, device, out_dir
):
    """
    KernelSHAP explanation with flattened (time × feature) inputs.
    """

    # ------------------------------
    # Prepare data
    # ------------------------------
    background = np.stack(val_seqs[:n_bg])   
    explain    = np.stack(test_seqs[:n_ex])   

    T, F = background.shape[1], background.shape[2]

    background_flat = background.reshape(background.shape[0], -1)
    explain_flat    = explain.reshape(explain.shape[0], -1)

    # ------------------------------
    # Model wrapper (numpy -> torch)
    # ------------------------------
    def model_forward_np(x_flat):
        batch_size = x_flat.shape[0]
        x = x_flat.reshape(batch_size, T, F)

        x = torch.tensor(x, dtype=torch.float32, device=device)
        lengths = torch.full(
            (batch_size,), T, dtype=torch.long, device=device
        )

        with torch.no_grad():
            logits = model((x, lengths))
            probs = torch.softmax(logits, dim=1)[:, 1]

        return probs.cpu().numpy()

    # ------------------------------
    # KernelSHAP
    # ------------------------------
    explainer = shap.KernelExplainer(model_forward_np, background_flat)

    shap_vals = explainer.shap_values(
        explain_flat, nsamples=n_spl
    )

    shap_vals = np.array(shap_vals)
    shap_vals = shap_vals.reshape(explain.shape[0], T, F)

    # ------------------------------
    # Plot top-10 features
    # ------------------------------
    feat_importance = np.mean(np.abs(shap_vals), axis=(0, 1))
    idx = np.argsort(feat_importance)[-10:][::-1]

    plt.figure(figsize=(6, 4))
    plt.barh(
        [variables[i] for i in idx][::-1],
        feat_importance[idx][::-1]
    )
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 10 Most Influential Features (6h Horizon)")
    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"top10_shap_features_{horizon}h_{model_name}.png"),
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()


# ======================================================
# MAIN ENTRY POINT
# ======================================================
def train_and_evaluate_lstm(data, model_name, source, horizon):

    variables = data["variables"]
    y_train, y_val, y_test = map(np.asarray, data["labels"])
    train_dfs, val_dfs, test_dfs = data["data"]

    train_seqs = [df.values.astype(np.float32) for df in train_dfs]
    val_seqs   = [df.values.astype(np.float32) for df in val_dfs]
    test_seqs  = [df.values.astype(np.float32) for df in test_dfs]

    # ------------------------------
    # NOTES ONLY OPTION
    # ------------------------------
    if NOTES_ONLY:
        keep = [
            i for i, v in enumerate(variables)
            if v.lower().startswith(("m1", "m2", "m3"))
        ]
        train_seqs = [x[:, keep] for x in train_seqs]
        val_seqs   = [x[:, keep] for x in val_seqs]
        test_seqs  = [x[:, keep] for x in test_seqs]
        variables  = [variables[i] for i in keep]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------
    # DATALOADERS
    # ------------------------------
    train_loader = DataLoader(
        VisitSequenceWithLabelDataset(train_seqs, y_train),
        batch_size=32, shuffle=True, collate_fn=time_collate_fn
    )

    val_loader = DataLoader(
        VisitSequenceWithLabelDataset(val_seqs, y_val),
        batch_size=64, shuffle=False, collate_fn=time_collate_fn
    )

    test_loader = DataLoader(
        VisitSequenceWithLabelDataset(test_seqs, y_test),
        batch_size=64, shuffle=False, collate_fn=time_collate_fn
    )

    # ------------------------------
    # MODEL
    # ------------------------------
    model = MyLSTM(
        dim_input=train_seqs[0].shape[1],
        bilstm_input=64,
        hidden_dim=64,
        num_layers=1,
        dropout=0.3
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # ------------------------------
    # TRAINING (EARLY STOPPING ON AUPRC)
    # ------------------------------
    best_val_pr, wait = -np.inf, 0
    best_state = None

    for epoch in range(1, 101):
        train_metrics = run_epoch(
            model, train_loader, optimizer, criterion, device, True
        )
        val_metrics = run_epoch(
            model, val_loader, None, criterion, device, False
        )

        if val_metrics["auprc"] > best_val_pr + 1e-4:
            best_val_pr = val_metrics["auprc"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= 10:
                break

    model.load_state_dict(best_state)

    # ------------------------------
    # TEST EVALUATION
    # ------------------------------
    test_metrics = run_epoch(
        model, test_loader, None, criterion, device, False
    )

    print(
        f"[{model_name}] "
        f"AUPRC={test_metrics['auprc']:.3f} "
        f"ROC-AUC={test_metrics['roc_auc']:.3f}"
    )

    # ------------------------------
    # SHAP 
    # ------------------------------
    compute_kernel_shap(
        model=model,
        model_name=model_name,
        horizon=horizon,
        val_seqs=val_seqs,
        test_seqs=test_seqs,
        n_bg=500, 
        n_ex=100, 
        n_spl=500,
        variables=variables,
        device=device,
        out_dir=os.path.join(OUTPUT_DIR, "shap_6h")
    )

    return test_metrics