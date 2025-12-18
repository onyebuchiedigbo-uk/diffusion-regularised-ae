import os, math, random, warnings
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
    LATENT_DIM, DIFFUSION_TIMESTEPS, SCHEDULE_NAME,
    JOINT_EPOCHS, JOINT_LR, JOINT_WEIGHT_DECAY,
    GRAD_NORM_MAX, LR_SCHEDULER_T_MAX,
    EMA_DECAY,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR,
)
from data import get_cifar10_loaders
from models import UNetAutoencoder, SmallTransformerDenoiser, EMA
from diffusion import get_linear_betas, get_cosine_betas, prepare_alphas, q_sample_batch

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

def grad_norm_of_params(params):
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach()
            total += g.pow(2).sum().item()
    return math.sqrt(total) if total > 0 else 0.0

def train_joint():
    set_seed(SEED)
    device = torch.device(DEVICE)

    trainloader, testloader, _, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_dir=DATA_DIR,
    )

    # Load AE warm-start
    ae = UNetAutoencoder(latent_dim=LATENT_DIM).to(device)
    warm_path = os.path.join(CHECKPOINT_DIR, "ae_warm.pth")
    ae.load_state_dict(torch.load(warm_path, map_location=device))

    # Diffusion schedule
    if SCHEDULE_NAME == "cosine":
        betas = get_cosine_betas(DIFFUSION_TIMESTEPS, device=device)
    else:
        betas = get_linear_betas(DIFFUSION_TIMESTEPS, device=device)
    alphas_dict = prepare_alphas(betas)

    # Freeze encoder modules (weights + BN stats)
    for module_name, module in ae.named_modules():
        if module_name.startswith("enc"):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
    print("Encoder frozen (weights + BN stats).")

    # Denoiser
    denoiser = SmallTransformerDenoiser(
        latent_dim=LATENT_DIM,
        n_tokens=8,
        d_model=128,
        n_heads=4,
        n_layers=4,
    ).to(device)

    # Trainable params = unfrozen AE parts + denoiser
    trainable_params = [p for p in ae.parameters() if p.requires_grad]
    trainable_params += list(denoiser.parameters())

    opt = torch.optim.AdamW(trainable_params, lr=JOINT_LR, weight_decay=JOINT_WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=LR_SCHEDULER_T_MAX)

    # EMA for AE
    ema_ae = EMA(ae, decay=EMA_DECAY)
    print("Joint training setup complete.")

    # History
    history_joint = {
        "epoch": [],
        "train_mse_ae": [],
        "train_mse_diff": [],
        "train_mse_total": [],
        "scale_diff": [],
        "grad_norm_encoder": [],
        "grad_norm_denoiser": [],
        "val_mse": [],
    }

    running_ae = None
    running_diff = None

    num_pixels = 3 * 32 * 32
    scale_latent_to_pixel = float(LATENT_DIM) / float(num_pixels)

    print("=== Joint optimisation (decoder + denoiser) ===")
    for epoch in range(JOINT_EPOCHS):
        ae.train()
        denoiser.train()
        s_ae = s_diff = s_tot = 0.0
        n = 0

        loop = tqdm(trainloader, desc=f"Joint E{epoch}", leave=False)
        for x, _ in loop:
            x = x.to(device, non_blocking=True)
            z0, xr = ae(x)   # encoder frozen

            t = torch.randint(0, DIFFUSION_TIMESTEPS, (x.size(0),), device=device, dtype=torch.long)
            zt, eps = q_sample_batch(z0, t, alphas_dict)
            eps_pred = denoiser(zt, t)

            L_diff = F.mse_loss(eps_pred, eps)          # per-latent
            L_ae   = F.mse_loss(xr, x)                  # per-pixel

            # EMA of losses
            if running_ae is None:
                running_ae   = L_ae.item()
                running_diff = L_diff.item()
            else:
                running_ae   = 0.99 * running_ae   + 0.01 * L_ae.item()
                running_diff = 0.99 * running_diff + 0.01 * L_diff.item()

            # Scale diffusion loss to pixel-equivalent
            L_diff_scaled       = L_diff * scale_latent_to_pixel
            running_diff_scaled = running_diff * scale_latent_to_pixel

            eps_small  = 1e-8
            scale_diff = running_ae / (running_diff_scaled + eps_small)
            scale_diff = float(max(min(scale_diff, 1e3), 1e-4))  # clamp

            L_total = L_ae + scale_diff * L_diff_scaled

            opt.zero_grad()
            L_total.backward()

            # Gradient norms for logging
            gn_ae  = grad_norm_of_params([p for p in ae.parameters() if p.requires_grad])
            gn_den = grad_norm_of_params(list(denoiser.parameters()))

            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=GRAD_NORM_MAX)
            opt.step()
            ema_ae.update(ae)

            s_ae   += L_ae.item()   * x.size(0)
            s_diff += L_diff.item() * x.size(0)
            s_tot  += L_total.item()* x.size(0)
            n += x.size(0)
            loop.set_postfix(L_ae=s_ae/n, L_diff=s_diff/n, lam=scale_diff)

        train_mse_ae    = s_ae   / n
        train_mse_diff  = s_diff / n
        train_mse_total = s_tot  / n

        # Validation MSE using EMA weights
        ema_ae.apply_to(ae)
        ae.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, _ in testloader:
                x = x.to(device, non_blocking=True)
                _, xr = ae(x)
                val_sum += F.mse_loss(xr, x).item() * x.size(0)
                val_n   += x.size(0)
        val_mse = val_sum / val_n
        ema_ae.restore(ae)

        history_joint["epoch"].append(epoch)
        history_joint["train_mse_ae"].append(train_mse_ae)
        history_joint["train_mse_diff"].append(train_mse_diff)
        history_joint["train_mse_total"].append(train_mse_total)
        history_joint["scale_diff"].append(scale_diff)
        history_joint["grad_norm_encoder"].append(gn_ae)
        history_joint["grad_norm_denoiser"].append(gn_den)
        history_joint["val_mse"].append(val_mse)

        print(
            f"Epoch {epoch:02d} | AE={train_mse_ae:.3e} | DIFF={train_mse_diff:.3e} | "
            f"TOT={train_mse_total:.3e} | VAL={val_mse:.3e} | λ={scale_diff:.3f} | "
            f"gn_ae={gn_ae:.3f} | gn_den={gn_den:.3f}"
        )

        scheduler.step()

    # Save joint history
    pd.DataFrame(history_joint).to_csv(os.path.join(RESULTS_DIR, "joint_history.csv"), index=False)
    torch.save(ae.state_dict(), os.path.join(CHECKPOINT_DIR, "ae_joint.pth"))
    torch.save(denoiser.state_dict(), os.path.join(CHECKPOINT_DIR, "denoiser_joint.pth"))
    print("Joint training finished and checkpoints saved.")

    # Plot joint curves
    df_joint = pd.DataFrame(history_joint)
    plt.figure(figsize=(14, 8))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    axes[0,0].plot(df_joint.epoch, df_joint.train_mse_ae, "o-", lw=2, label="AE loss")
    axes[0,0].set_yscale("log")
    axes[0,0].set_xlabel("Epoch"); axes[0,0].set_ylabel("MSE")
    axes[0,0].set_title("AE Reconstruction Loss")
    axes[0,0].grid(True, alpha=0.3); axes[0,0].legend()

    axes[0,1].plot(df_joint.epoch, df_joint.train_mse_diff, "o-", lw=2, label="Diffusion MSE")
    axes[0,1].set_yscale("log")
    axes[0,1].set_xlabel("Epoch"); axes[0,1].set_ylabel("MSE")
    axes[0,1].set_title("Diffusion Denoising Loss")
    axes[0,1].grid(True, alpha=0.3); axes[0,1].legend()

    axes[0,2].plot(df_joint.epoch, df_joint.train_mse_total, "o-", lw=2, label="Total loss")
    axes[0,2].set_yscale("log")
    axes[0,2].set_xlabel("Epoch"); axes[0,2].set_ylabel("Loss")
    axes[0,2].set_title("Total Balanced Loss")
    axes[0,2].grid(True, alpha=0.3); axes[0,2].legend()

    axes[1,0].plot(df_joint.epoch, df_joint.val_mse, "o-", lw=2, label="Val MSE")
    axes[1,0].set_yscale("log")
    axes[1,0].set_xlabel("Epoch"); axes[1,0].set_ylabel("MSE")
    axes[1,0].set_title("Validation Reconstruction Loss")
    axes[1,0].grid(True, alpha=0.3); axes[1,0].legend()

    axes[1,1].plot(df_joint.epoch, df_joint.grad_norm_encoder, "o-", lw=2, label="AE grad norm")
    axes[1,1].plot(df_joint.epoch, df_joint.grad_norm_denoiser, "s-", lw=2, label="Denoiser grad norm")
    axes[1,1].set_yscale("log")
    axes[1,1].set_xlabel("Epoch"); axes[1,1].set_ylabel(r"$\|\nabla\|_2$")
    axes[1,1].set_title("Gradient Norms")
    axes[1,1].grid(True, alpha=0.3); axes[1,1].legend()

    axes[1,2].plot(df_joint.epoch, df_joint.scale_diff, "o-", lw=2, label=r"$\lambda(t)$")
    axes[1,2].set_xlabel("Epoch"); axes[1,2].set_ylabel("Scale factor")
    axes[1,2].set_title(r"Dynamic Scale Factor $\lambda(t)$")
    axes[1,2].grid(True, alpha=0.3); axes[1,2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "joint_training_curves.pdf"), bbox_inches="tight")
    plt.close()

    return history_joint

if __name__ == "__main__":
    train_joint()
