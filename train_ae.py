import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn.functional as F

from config import (
    SEED, DEVICE,
    BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, DATA_DIR,
    LATENT_DIM, AE_EPOCHS, AE_LR, AE_WEIGHT_DECAY,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR,
)
from data import get_cifar10_loaders
from models import UNetAutoencoder

# Matplotlib LaTeX-like fonts
plt.rcParams.update({
    "font.size": 11,
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "mathtext.rm": "serif",
})


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def train_ae():
    set_seed(SEED)
    device = torch.device(DEVICE)

    trainloader, testloader, _, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_dir=DATA_DIR,
    )

    ae = UNetAutoencoder(latent_dim=LATENT_DIM).to(device)
    opt_ae = torch.optim.AdamW(ae.parameters(), lr=AE_LR, weight_decay=AE_WEIGHT_DECAY)

    history_ae = {"epoch": [], "train_mse": [], "val_mse": []}

    print("=== Autoencoder warm-start training ===")
    for epoch in range(AE_EPOCHS):
        ae.train()
        sum_loss, n = 0.0, 0
        loop = tqdm(trainloader, desc=f"AE Train E{epoch}", leave=False)
        for x, _ in loop:
            x = x.to(device, non_blocking=True)
            _, xr = ae(x)
            loss = F.mse_loss(xr, x)

            opt_ae.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae.parameters(), 1.0)
            opt_ae.step()

            sum_loss += loss.item() * x.size(0)
            n += x.size(0)
            loop.set_postfix(AE_loss=sum_loss / n)

        train_mse = sum_loss / n

        # Validation
        ae.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, _ in testloader:
                x = x.to(device, non_blocking=True)
                _, xr = ae(x)
                val_sum += F.mse_loss(xr, x).item() * x.size(0)
                val_n += x.size(0)
        val_mse = val_sum / val_n

        history_ae["epoch"].append(epoch)
        history_ae["train_mse"].append(train_mse)
        history_ae["val_mse"].append(val_mse)
        print(f"Epoch {epoch:02d} | train_mse={train_mse:.3e} | val_mse={val_mse:.3e}")

    # Save checkpoint and history (tables as CSV)
    torch.save(ae.state_dict(), os.path.join(CHECKPOINT_DIR, "ae_warm.pth"))
    pd.DataFrame(history_ae).to_csv(os.path.join(RESULTS_DIR, "ae_history.csv"), index=False)
    print("Saved AE warm-start checkpoint and history.")

    # Plot training curves (show + save PDF)
    fig = plt.figure(figsize=(6, 4))
    plt.plot(history_ae["epoch"], history_ae["train_mse"], "o-", label="Train MSE")
    plt.plot(history_ae["epoch"], history_ae["val_mse"], "s-", label="Val MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Autoencoder Reconstruction Loss")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "ae_training_curves.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved AE training curves to:", out_path)

    return history_ae


if __name__ == "__main__":
    train_ae()
