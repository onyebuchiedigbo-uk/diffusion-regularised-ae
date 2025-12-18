# Diffusion‑Regularised UNet Autoencoder

This repository contains the code and experiments for a diffusion‑regularised UNet autoencoder for high‑fidelity image reconstruction on CIFAR‑10. The code implements autoencoder warm‑start, joint optimisation with a latent‑space diffusion denoiser, evaluation metrics (PSNR, SSIM, LPIPS, FID), and ablation studies over latent dimension and diffusion schedule.

## 1. Project structure

- `config.py` – Global configuration and hyperparameters (paths, seeds, model and training settings).  
- `models.py` – UNet autoencoder, transformer denoiser, and EMA implementation.  
- `diffusion.py` – Beta schedules and forward diffusion helpers.  
- `data.py` – CIFAR‑10 dataloaders and preprocessing.  
- `train_ae.py` – Autoencoder warm‑start training.  
- `train_joint.py` – Joint training of AE decoder and diffusion denoiser (encoder frozen).  
- `eval.py` – Evaluation and visualisations (PSNR, SSIM, LPIPS, FID, grids, UMAP, histograms, interpolations).  
- `ablation.py` – Ablation experiments over latent dimensions and diffusion schedules.  
- `configs/main_run.txt` – Canonical configuration for the main experiment reported in the paper.  
- `notebooks/main_experiment_log.ipynb` – Notebook log of the full main experiment (warm‑start, joint training, evaluation, ablations).  
- `checkpoints/` – Saved model checkpoints (`ae_warm.pth`, `ae_joint.pth`, `denoiser_joint.pth`).  
- `results/` – CSV tables with training histories and metrics.  
- `figures/` – PDF figures used in the paper.

## 2. Requirements

- Python 3.10+  
- PyTorch and torchvision  
- Additional Python packages:
  - `numpy`, `pandas`, `matplotlib`, `seaborn`, `tqdm`  
  - `scikit-image`, `scikit-learn`, `umap-learn`  
  - `lpips`, `pytorch-msssim`, `torch-fidelity`, `einops`, `accelerate`, `transformers`, `diffusers`

On Google Colab, the notebook `notebooks/main_experiment_log.ipynb` (or the example in the root) installs all dependencies automatically in Cell 0.

## 3. Quick start (main pipeline)

From the repo root:

1. **Autoencoder warm‑start**

   ```bash
   python train_ae.py
   ```

   Outputs:
   - `checkpoints/ae_warm.pth`  
   - `results/ae_history.csv`  
   - `figures/ae_training_curves.pdf` (also displayed during execution)

2. **Joint training (decoder + denoiser)**

   ```bash
   python train_joint.py
   ```

   Outputs:
   - `checkpoints/ae_joint.pth`, `checkpoints/denoiser_joint.pth`  
   - `results/joint_history.csv`  
   - `figures/joint_training_curves.pdf` (2×3 multi‑panel figure with subplots (a)–(f))

3. **Evaluation and visualisations**

 ```bash
   python eval.py
   ```

   Uses the joint checkpoint (`ae_joint.pth`, falling back to `ae_warm.pth` if needed) and computes reconstruction and distribution metrics on the CIFAR‑10 test set.

   Outputs:
   - `results/metrics_summary.csv` (PSNR, SSIM, LPIPS, FID, number of FID samples)  
   - `figures/recon_grid.pdf` – Original / reconstruction / \|x − x̂\|  
   - `figures/per_image_mse_hist.pdf` – Per‑image MSE histogram  
   - `figures/per_class_mse.pdf` – Mean reconstruction error per CIFAR‑10 class  
   - `figures/latent_umap.pdf` – UMAP of latent codes  
   - `figures/cross_class_interpolation.pdf` – Cross‑class interpolations (latent + skip blending)

   All figures are displayed inline when run in a notebook and saved as PDFs.

5. **Ablation experiments**

   ```bash
   python ablation.py
   ```

   Runs ablations over:
   - `latent_dim ∈ {64, 128, 256}`  
   - `schedule ∈ {linear, cosine}`  
   - `use_diffusion ∈ {True, False}`

   Outputs:
   - `results/ablation_summary.csv` – Aggregate metrics table (PSNR, SSIM, LPIPS, FID) for each configuration.  
   - `results/ablation_histories.pkl` – Full histories for all runs.  
   - `results/ablation_ld*_sched*_diff*_ae_history.csv` – Per‑run AE warm‑up histories.  
   - `results/ablation_ld*_sched*_diff*_joint_history.csv` – Per‑run joint training histories.

## 4. Reproducing the main experiment (notebook)

To reproduce the exact sequence used for the paper:

1. Open `notebooks/main_experiment_log.ipynb` in Jupyter or Colab.  
2. Run all cells in order:
   - Cell 0: environment setup and repo clone (for Colab).  
   - Cell 1: global imports and plotting defaults.  
   - Cell 2: `history_ae = train_ae()`  
   - Cell 3: `history_joint = train_joint()`  
   - Cell 4: `run_eval()`  
   - Cell 5: `run_ablation()`

The notebook logs console outputs and generates all CSVs and figures under `results/` and `figures/` as described above.

## 5. Configuration and main reported results

The canonical configuration for the main model is documented in:

- `configs/main_run.txt`

This file lists all key hyperparameters and settings (latent dimension, schedules, training epochs, optimiser settings, diffusion steps, FID configuration, etc.) and records the main CIFAR‑10 test‑set results (PSNR, SSIM, LPIPS, FID) used in the paper.

## 6. Citation

