import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
from torchvision import utils as tv_utils
from torchvision.transforms import Resize

import lpips
from pytorch_msssim import ssim as pssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torch_fidelity import calculate_metrics
from PIL import Image

from config import (
    SEED, DEVICE,
    BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, DATA_DIR,
    LATENT_DIM,
    CHECKPOINT_DIR, RESULTS_DIR, FIGURES_DIR,
    N_FID_SAMPLES, FID_RESIZE_SIZE,
)
from data import get_cifar10_loaders
from models import UNetAutoencoder, EMA

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

def evaluate_recon(ae_model, dataloader, device):
    ae_model.eval()
    psnr_all, ssim_all, lpips_all = [], [], []

    lpips_fn = lpips.LPIPS(net="vgg").to(device)

    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Eval recon", leave=False):
            x = x.to(device, non_blocking=True)
            _, xr = ae_model(x)

            x01  = (x  * 0.5 + 0.5).clamp(0, 1)
            xr01 = (xr * 0.5 + 0.5).clamp(0, 1)

            for a, b in zip(
                x01.cpu().numpy().transpose(0,2,3,1),
                xr01.cpu().numpy().transpose(0,2,3,1),
            ):
                psnr_all.append(sk_psnr(a, b, data_range=1.0))

            ssim_batch = pssim(xr01, x01, data_range=1.0, size_average=False)
            ssim_all.extend(ssim_batch.cpu().numpy())

            lp = lpips_fn(xr, x)
            lpips_all.extend(lp.squeeze().cpu().numpy())

    return {
        "PSNR": float(np.mean(psnr_all)),
        "SSIM": float(np.mean(ssim_all)),
        "LPIPS": float(np.mean(lpips_all)),
    }

def compute_fid(ae_model, dataloader, device, n_samples, out_real_dir, out_fake_dir):
    resize = Resize(FID_RESIZE_SIZE, antialias=True)
    count = 0

    ae_model.eval()
    with torch.no_grad():
        for x, _ in tqdm(dataloader, desc="Saving FID images", leave=False):
            x = x.to(device)
            _, xr = ae_model(x)

            for i in range(x.size(0)):
                if count >= n_samples:
                    break

                real = (x[i]  * 0.5 + 0.5).clamp(0, 1)
                fake = (xr[i] * 0.5 + 0.5).clamp(0, 1)

                tv_utils.save_image(resize(real.cpu()),
                                    os.path.join(out_real_dir, f"{count:05d}_real.png"))
                tv_utils.save_image(resize(fake.cpu()),
                                    os.path.join(out_fake_dir, f"{count:05d}_fake.png"))
                count += 1

            if count >= n_samples:
                break

    print(f"Saved {count} FID image pairs.")

    fid_out = calculate_metrics(
        input1=out_real_dir,
        input2=out_fake_dir,
        fid=True, isc=False, kid=False,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    return float(fid_out["frechet_inception_distance"])

def show_recon_grid(ae_model, dataloader, device, n=8):
    ae_model.eval()
    x_batch, _ = next(iter(dataloader))
    x = x_batch[:n].to(device)
    with torch.no_grad():
        _, xr = ae_model(x)

    x01  = (x  * 0.5 + 0.5).clamp(0, 1).cpu()
    xr01 = (xr * 0.5 + 0.5).clamp(0, 1).cpu()
    diff = torch.abs(xr01 - x01)

    grid = tv_utils.make_grid(torch.cat([x01, xr01, diff], dim=0),
                              nrow=n, padding=2, normalize=False)
    plt.figure(figsize=(1.6*n, 4.5))
    plt.imshow(grid.permute(1,2,0).numpy())
    plt.axis("off")
    plt.title(r"Original / Reconstruction / $|x - \hat{x}|$")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "recon_grid.pdf"), bbox_inches="tight")
    plt.close()

def per_image_mse_plots(ae_model, dataloader, device):
    ae_model.eval()
    mse_list, label_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, xr = ae_model(x)
            per_img_mse = ((xr - x)**2).view(x.size(0), -1).mean(dim=1).cpu().numpy()
            mse_list.append(per_img_mse)
            label_list.append(y.numpy())
    mse_all    = np.concatenate(mse_list)
    labels_all = np.concatenate(label_list)

    plt.figure(figsize=(6,3))
    plt.hist(mse_all, bins=100)
    plt.xlabel("Per-image MSE")
    plt.ylabel("Count")
    plt.title("Per-image Reconstruction Error")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "per_image_mse_hist.pdf"), bbox_inches="tight")
    plt.close()

    df_mse = pd.DataFrame({"mse": mse_all, "label": labels_all})
    means  = df_mse.groupby("label")["mse"].mean()
    plt.figure(figsize=(7,3))
    means.plot(kind="bar")
    plt.ylabel("Mean MSE")
    plt.title("Mean Reconstruction Error per CIFAR-10 Class")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "per_class_mse.pdf"), bbox_inches="tight")
    plt.close()
    print("Per-class MSE:\n", means)

def latent_umap(ae_model, dataloader, device, max_points=2000):
    import umap
    ae_model.eval()
    Z_list, y_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            z, _ = ae_model.encode(x)
            Z_list.append(z.cpu().numpy())
            y_list.append(y.numpy())
    Z = np.concatenate(Z_list, axis=0)
    Y = np.concatenate(y_list, axis=0)

    idx = np.random.choice(len(Z), min(max_points, len(Z)), replace=False)
    Z_sub = Z[idx]
    Y_sub = Y[idx]

    reducer = umap.UMAP(n_components=2, random_state=SEED)
    Z_umap  = reducer.fit_transform(Z_sub)

    plt.figure(figsize=(7,6))
    sc = plt.scatter(Z_umap[:,0], Z_umap[:,1], c=Y_sub, cmap="tab10", s=6)
    plt.colorbar(sc, ticks=range(10))
    plt.title("UMAP of Latent Space")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "latent_umap.pdf"), bbox_inches="tight")
    plt.close()

def run_eval():
    set_seed(SEED)
    device = torch.device(DEVICE)

    _, testloader, _, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_dir=DATA_DIR,
    )

    ae = UNetAutoencoder(latent_dim=LATENT_DIM).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "ae_joint.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(CHECKPOINT_DIR, "ae_warm.pth")
    ae.load_state_dict(torch.load(ckpt_path, map_location=device))

    ema_ae = EMA(ae, decay=1.0)  # use current weights as EMA
    ema_ae.apply_to(ae)

    # Metrics
    metrics = evaluate_recon(ae, testloader, device)

    import tempfile
    real_dir = tempfile.mkdtemp()
    fake_dir = tempfile.mkdtemp()
    fid_value = compute_fid(ae, testloader, device, N_FID_SAMPLES, real_dir, fake_dir)

    # Save metrics summary
    summary = {
        **metrics,
        "FID": fid_value,
        "FID_samples": N_FID_SAMPLES,
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    print("\nReconstruction + FID Results (CIFAR-10 test set)")
    print(df_summary.round(4).to_string(index=False))

    # Visualizations
    show_recon_grid(ae, testloader, device, n=8)
    per_image_mse_plots(ae, testloader, device)
    latent_umap(ae, testloader, device)

    ema_ae.restore(ae)

if __name__ == "__main__":
    run_eval()
