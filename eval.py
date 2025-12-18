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
                x01.cpu().numpy().transpose(0, 2, 3, 1),
                xr01.cpu().numpy().transpose(0, 2, 3, 1),
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

                tv_utils.save_image(
                    resize(real.cpu()),
                    os.path.join(out_real_dir, f"{count:05d}_real.png"),
                )
                tv_utils.save_image(
                    resize(fake.cpu()),
                    os.path.join(out_fake_dir, f"{count:05d}_fake.png"),
                )
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

    grid = tv_utils.make_grid(
        torch.cat([x01, xr01, diff], dim=0),
        nrow=n, padding=2, normalize=False,
    )
    fig = plt.figure(figsize=(1.6 * n, 4.5))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.axis("off")
    plt.title(r"Original / Reconstruction / $|x - \hat{x}|$")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "recon_grid.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved reconstruction grid to:", out_path)


def per_image_mse_plots(ae_model, dataloader, device):
    ae_model.eval()
    mse_list, label_list = [], []
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            _, xr = ae_model(x)
            per_img_mse = ((xr - x) ** 2).view(x.size(0), -1).mean(dim=1).cpu().numpy()
            mse_list.append(per_img_mse)
            label_list.append(y.numpy())
    mse_all    = np.concatenate(mse_list)
    labels_all = np.concatenate(label_list)

    # Per-image MSE histogram
    fig = plt.figure(figsize=(6, 3))
    plt.hist(mse_all, bins=100)
    plt.xlabel("Per-image MSE")
    plt.ylabel("Count")
    plt.title("Per-image Reconstruction Error")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "per_image_mse_hist.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved per-image MSE histogram to:", out_path)

    # Per-class mean MSE bar plot
    df_mse = pd.DataFrame({"mse": mse_all, "label": labels_all})
    means  = df_mse.groupby("label")["mse"].mean()

    fig = plt.figure(figsize=(7, 3))
    means.plot(kind="bar")
    plt.ylabel("Mean MSE")
    plt.title("Mean Reconstruction Error per CIFAR-10 Class")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "per_class_mse.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved per-class MSE bar plot to:", out_path)
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

    fig = plt.figure(figsize=(7, 6))
    sc = plt.scatter(Z_umap[:, 0], Z_umap[:, 1], c=Y_sub, cmap="tab10", s=6)
    plt.colorbar(sc, ticks=range(10))
    plt.title("UMAP of Latent Space")
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "latent_umap.pdf")
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved latent UMAP to:", out_path)


def cross_class_interpolation(ae_model, trainset, device, out_path, steps=12):
    """Generate cross-class interpolation figure and save to out_path."""
    import numpy as np

    CLASSNAMES = [
        "Plane", "Car", "Bird", "Cat", "Deer",
        "Dog", "Frog", "Horse", "Ship", "Truck",
    ]

    def get_random_by_class(class_id, dataset):
        labels = np.array(dataset.targets)
        idxs = np.where(labels == class_id)[0]
        idx = np.random.choice(idxs)
        img, _ = dataset[idx]
        return img.unsqueeze(0).to(device)

    @torch.no_grad()
    def encode_img(x):
        z, skips = ae_model.encode(x)
        skips = tuple(s.clone() for s in skips)
        return z, skips

    def lerp(a, b, t):
        return (1.0 - t) * a + t * b

    @torch.no_grad()
    def interpolate_pair(class_a, class_b, steps):
        img_a = get_random_by_class(class_a, trainset)
        img_b = get_random_by_class(class_b, trainset)
        z_a, skips_a = encode_img(img_a)
        z_b, skips_b = encode_img(img_b)

        frames = []
        for i in range(steps):
            t = i / (steps - 1)
            z_t = lerp(z_a, z_b, t)
            skips_t = [lerp(sa, sb, t) for sa, sb in zip(skips_a, skips_b)]
            rec = ae_model.decode(z_t, skips_t)
            frames.append((rec * 0.5 + 0.5).clamp(0, 1).cpu())
        return torch.cat(frames, dim=0)

    PAIRS = [(2, 8), (3, 0), (6, 9), (1, 7)]
    num_steps = steps

    fig, axes = plt.subplots(
        len(PAIRS), num_steps,
        figsize=(num_steps * 1.2, len(PAIRS) * 1.5),
    )

    for r, (a_cls, b_cls) in enumerate(PAIRS):
        frames = interpolate_pair(a_cls, b_cls, num_steps)
        for c in range(num_steps):
            axes[r, c].imshow(frames[c].permute(1, 2, 0).numpy())
            axes[r, c].axis("off")
            if c == 0:
                axes[r, c].set_title(CLASSNAMES[a_cls], fontsize=8)
            if c == num_steps - 1:
                axes[r, c].set_title(CLASSNAMES[b_cls], fontsize=8)

    plt.suptitle("Cross-Class Interpolation (Latent + Skip Blending)", fontsize=14)
    plt.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.show()
    plt.close(fig)
    print("Saved cross-class interpolation to:", out_path)


def run_eval():
    set_seed(SEED)
    device = torch.device(DEVICE)

    trainloader, testloader, trainset, _ = get_cifar10_loaders(
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

    # Save metrics summary (table as CSV)
    summary = {
        **metrics,
        "FID": fid_value,
        "FID_samples": N_FID_SAMPLES,
    }
    df_summary = pd.DataFrame([summary])
    df_summary.to_csv(os.path.join(RESULTS_DIR, "metrics_summary.csv"), index=False)
    print("\nReconstruction + FID Results (CIFAR-10 test set)")
    print(df_summary.round(4).to_string(index=False))

    # Visualizations (show + save PDFs)
    show_recon_grid(ae, testloader, device, n=8)
    per_image_mse_plots(ae, testloader, device)
    latent_umap(ae, testloader, device)

    # Cross-class interpolation figure
    cross_class_interpolation(
        ae_model=ae,
        trainset=trainset,
        device=device,
        out_path=os.path.join(FIGURES_DIR, "cross_class_interpolation.pdf"),
        steps=12,
    )

    ema_ae.restore(ae)


if __name__ == "__main__":
    run_eval()
