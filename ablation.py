import os, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F

import lpips
from pytorch_msssim import ssim as pssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from torchvision.transforms import Resize
from torch_fidelity import calculate_metrics

from config import (
    SEED, DEVICE,
    BATCH_SIZE, VAL_BATCH_SIZE, NUM_WORKERS, DATA_DIR,
    AB_LATENT_DIMS, AB_SCHEDULES,
    AB_AE_EPOCHS, AB_JOINT_EPOCHS, AB_FID_SAMPLES,
    CHECKPOINT_DIR, RESULTS_DIR,
)
from data import get_cifar10_loaders
from models import UNetAutoencoder, SmallTransformerDenoiser
from diffusion import get_linear_betas, get_cosine_betas, prepare_alphas, q_sample_batch


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def eval_recon_local(ae_model, testloader, device, cfg_name=""):
    ae_model.eval()
    psnr_all, ssim_all, lpips_all = [], [], []
    lpips_fn = lpips.LPIPS(net="vgg").to(device)
    with torch.no_grad():
        for x, _ in tqdm(testloader,
                         desc=f"Eval recon ({cfg_name})",
                         leave=False):
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


def eval_fid_local(ae_model, testloader, device, n_samples, cfg_name=""):
    resize_299 = Resize((299, 299), antialias=True)
    import tempfile
    real_dir = tempfile.mkdtemp()
    fake_dir = tempfile.mkdtemp()
    count = 0

    ae_model.eval()
    with torch.no_grad():
        for x, _ in tqdm(testloader,
                         desc=f"Sav FID ({cfg_name})",
                         leave=False):
            x = x.to(device)
            _, xr = ae_model(x)
            for i in range(x.size(0)):
                if count >= n_samples:
                    break
                real = (x[i]  * 0.5 + 0.5).clamp(0, 1)
                fake = (xr[i] * 0.5 + 0.5).clamp(0, 1)
                from torchvision import utils as tv_utils
                tv_utils.save_image(resize_299(real.cpu()),
                                    os.path.join(real_dir, f"{count:05d}_real.png"))
                tv_utils.save_image(resize_299(fake.cpu()),
                                    os.path.join(fake_dir,  f"{count:05d}_fake.png"))
                count += 1
            if count >= n_samples:
                break

    fid_out = calculate_metrics(
        input1=real_dir,
        input2=fake_dir,
        fid=True, isc=False, kid=False,
        verbose=False,
        cuda=torch.cuda.is_available(),
    )
    return float(fid_out["frechet_inception_distance"])


def run_single_ablation(latent_dim, schedule_name, use_diffusion=True):
    cfg_name = f"ld={latent_dim},sched={schedule_name},diff={use_diffusion}"
    print(f"\n========== Ablation: {cfg_name} ==========")
    set_seed(SEED)
    device = torch.device(DEVICE)

    trainloader, testloader, _, _ = get_cifar10_loaders(
        batch_size=BATCH_SIZE,
        val_batch_size=VAL_BATCH_SIZE,
        num_workers=NUM_WORKERS,
        data_dir=DATA_DIR,
    )

    # AE warm-start
    ae_ab = UNetAutoencoder(latent_dim=latent_dim).to(device)
    denoiser_ab = SmallTransformerDenoiser(latent_dim=latent_dim).to(device)

    opt_ae_ab = torch.optim.AdamW(ae_ab.parameters(), lr=2e-4, weight_decay=1e-6)

    history_ae_ab = {"epoch": [], "train_mse": [], "val_mse": []}
    print(f"\n[AE warm-up] {cfg_name}")
    for epoch in range(AB_AE_EPOCHS):
        ae_ab.train()
        s_loss, n = 0.0, 0
        for x, _ in tqdm(trainloader,
                         desc=f"AE E{epoch} ({cfg_name})",
                         leave=False):
            x = x.to(device, non_blocking=True)
            _, xr = ae_ab(x)
            loss = F.mse_loss(xr, x)
            opt_ae_ab.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ae_ab.parameters(), 1.0)
            opt_ae_ab.step()

            s_loss += loss.item() * x.size(0)
            n      += x.size(0)
        train_mse = s_loss / n

        # validation
        ae_ab.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, _ in testloader:
                x = x.to(device, non_blocking=True)
                _, xr = ae_ab(x)
                val_sum += F.mse_loss(xr, x).item() * x.size(0)
                val_n   += x.size(0)
        val_mse = val_sum / val_n

        history_ae_ab["epoch"].append(epoch)
        history_ae_ab["train_mse"].append(train_mse)
        history_ae_ab["val_mse"].append(val_mse)
        print(f"AE E{epoch:02d}: train={train_mse:.3e}, val={val_mse:.3e}")

    # Diffusion schedule
    diffusion_timesteps = 100
    if schedule_name == "cosine":
        betas_ab = get_cosine_betas(diffusion_timesteps, device=device)
    else:
        betas_ab = get_linear_betas(diffusion_timesteps, device=device)
    alphas_ab = prepare_alphas(betas_ab)

    # Freeze encoder
    for module_name, module in ae_ab.named_modules():
        if module_name.startswith("enc"):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

    # Joint training
    trainable_params_ab = [p for p in ae_ab.parameters() if p.requires_grad]
    if use_diffusion:
        trainable_params_ab += list(denoiser_ab.parameters())
    opt_ab = torch.optim.AdamW(trainable_params_ab, lr=2e-4, weight_decay=1e-6)

    running_ae_ab   = None
    running_diff_ab = None
    num_pixels = 3 * 32 * 32
    scale_latent_to_pixel_ab = float(latent_dim) / float(num_pixels)

    history_joint_ab = {
        "epoch": [],
        "train_mse_ae": [],
        "train_mse_diff": [],
        "train_mse_total": [],
        "scale_diff": [],
        "val_mse": [],
    }

    print(f"\n[Joint phase] {cfg_name}")
    for epoch in range(AB_JOINT_EPOCHS):
        ae_ab.train()
        if use_diffusion:
            denoiser_ab.train()

        s_ae = s_diff = s_tot = 0.0
        n = 0

        for x, _ in tqdm(trainloader,
                         desc=f"Joint E{epoch} ({cfg_name})",
                         leave=False):
            x = x.to(device, non_blocking=True)
            z0, xr = ae_ab(x)
            L_ae = F.mse_loss(xr, x)

            if use_diffusion:
                t = torch.randint(0, diffusion_timesteps, (x.size(0),),
                                  device=device, dtype=torch.long)
                zt, eps = q_sample_batch(z0, t, alphas_ab)
                eps_pred = denoiser_ab(zt, t)
                L_diff = F.mse_loss(eps_pred, eps)

                if running_ae_ab is None:
                    running_ae_ab   = L_ae.item()
                    running_diff_ab = L_diff.item()
                else:
                    running_ae_ab   = 0.99 * running_ae_ab   + 0.01 * L_ae.item()
                    running_diff_ab = 0.99 * running_diff_ab + 0.01 * L_diff.item()

                L_diff_scaled       = L_diff * scale_latent_to_pixel_ab
                running_diff_scaled = running_diff_ab * scale_latent_to_pixel_ab

                eps_small = 1e-8
                scale_diff_ab = running_ae_ab / (running_diff_scaled + eps_small)
                scale_diff_ab = float(max(min(scale_diff_ab, 1e3), 1e-4))

                L_total = L_ae + scale_diff_ab * L_diff_scaled
            else:
                L_diff = torch.tensor(0.0, device=x.device)
                scale_diff_ab = 0.0
                L_total = L_ae

            opt_ab.zero_grad()
            L_total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params_ab, max_norm=5.0)
            opt_ab.step()

            s_ae   += L_ae.item()   * x.size(0)
            s_diff += L_diff.item() * x.size(0)
            s_tot  += L_total.item()* x.size(0)
            n      += x.size(0)

        train_mse_ae    = s_ae   / n
        train_mse_diff  = s_diff / n
        train_mse_total = s_tot  / n

        # validation
        ae_ab.eval()
        val_sum, val_n = 0.0, 0
        with torch.no_grad():
            for x, _ in testloader:
                x = x.to(device, non_blocking=True)
                _, xr = ae_ab(x)
                val_sum += F.mse_loss(xr, x).item() * x.size(0)
                val_n   += x.size(0)
        val_mse = val_sum / val_n

        history_joint_ab["epoch"].append(epoch)
        history_joint_ab["train_mse_ae"].append(train_mse_ae)
        history_joint_ab["train_mse_diff"].append(train_mse_diff)
        history_joint_ab["train_mse_total"].append(train_mse_total)
        history_joint_ab["scale_diff"].append(scale_diff_ab)
        history_joint_ab["val_mse"].append(val_mse)

        print(
            f"Joint E{epoch:02d}: AE={train_mse_ae:.3e}, DIFF={train_mse_diff:.3e}, "
            f"TOT={train_mse_total:.3e}, VAL={val_mse:.3e}, λ={scale_diff_ab:.3f}"
        )

    # Reconstruction metrics
    metrics_ab = eval_recon_local(ae_ab, testloader, device, cfg_name=cfg_name)
    fid_ab = eval_fid_local(ae_ab, testloader, device, AB_FID_SAMPLES, cfg_name=cfg_name)

    return metrics_ab, fid_ab, history_ae_ab, history_joint_ab


def run_ablation():
    all_summaries = []
    all_histories = []

    for ld in AB_LATENT_DIMS:
        for sched in AB_SCHEDULES:
            for use_diff in [True, False]:
                metrics_ab, fid_ab, hist_ae_ab, hist_joint_ab = run_single_ablation(
                    latent_dim=ld, schedule_name=sched, use_diffusion=use_diff
                )
                summary_row = {
                    "latent_dim": ld,
                    "schedule": sched,
                    "use_diffusion": use_diff,
                    "PSNR": metrics_ab["PSNR"],
                    "SSIM": metrics_ab["SSIM"],
                    "LPIPS": metrics_ab["LPIPS"],
                    "FID": fid_ab,
                    "FID_samples": AB_FID_SAMPLES,
                }
                all_summaries.append(summary_row)

                all_histories.append({
                    "config": f"ld={ld},sched={sched},diff={use_diff}",
                    "ae": hist_ae_ab,
                    "joint": hist_joint_ab,
                })

                # Also save per-run histories as CSV (tables)
                cfg_safe = f"ld{ld}_sched{sched}_diff{use_diff}".replace(",", "").replace(" ", "")
                pd.DataFrame(hist_ae_ab).to_csv(
                    os.path.join(RESULTS_DIR, f"ablation_{cfg_safe}_ae_history.csv"),
                    index=False,
                )
                pd.DataFrame(hist_joint_ab).to_csv(
                    os.path.join(RESULTS_DIR, f"ablation_{cfg_safe}_joint_history.csv"),
                    index=False,
                )

    df_summary = pd.DataFrame(all_summaries)
    df_summary.to_csv(os.path.join(RESULTS_DIR, "ablation_summary.csv"), index=False)
    print("\nFull ablation summary:")
    print(df_summary.round(4).to_string(index=False))

    # Save full histories as a pickle for later analysis
    import pickle
    with open(os.path.join(RESULTS_DIR, "ablation_histories.pkl"), "wb") as f:
        pickle.dump(all_histories, f)


if __name__ == "__main__":
    run_ablation()
