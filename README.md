# Diffusion‑Regularised Autoencoder

This repository contains the code and experiments for a diffusion‑regularised autoencoder for high‑fidelity image reconstruction on CIFAR‑10. The code implements autoencoder warm‑start, joint optimisation with a latent‑space diffusion denoiser, evaluation metrics (PSNR, SSIM, LPIPS, FID), and ablation studies over latent dimension and diffusion schedule.

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

On Google Colab, the notebook `notebooks/main_experiment_log.ipynb` installs all dependencies automatically in Cell 0.

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
[1] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic mod-
els,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 33, 2020,
pp. 6840–6851. [Online]. Available: https://arxiv.org/abs/2006.11239

[2] J. Song and S. Ermon, “Denoising diffusion implicit models,” in Proc. Int.
Conf. Learn. Represent. (ICLR), 2021. [Online]. Available: https://arxiv.
org/abs/2010.02502

[3] Y. Song, J. Sohl-Dickstein, D. P. Kingma, A. Kumar, S. Ermon, and
B. Poole, “Score-based generative modeling through stochastic differential
equations,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021. [Online].
Available: https://arxiv.org/abs/2011.13456

[4] C. Ledig et al., “Photo-realistic single image super-resolution using a
generative adversarial network,” in Proc. IEEE/CVF Conf. Comput. Vis.
Pattern Recognit. (CVPR), 2017, pp. 4681–4690. [Online]. Available:
https://arxiv.org/abs/1609.04802

[5] J. Johnson, A. Alahi, and L. Fei-Fei, “Perceptual losses for real-time style
transfer and super-resolution,” in Proc. Eur. Conf. Comput. Vis. (ECCV),
2016, pp. 694–711. [Online]. Available: https://arxiv.org/abs/1603.08155

[6] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
“The unreasonable effectiveness of deep features as a perceptual
metric,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), 2018, pp. 586–595. [Online]. Available:
https://openaccess.thecvf.com/content_cvpr_2018/papers/Zhang_The_
Unreasonable_Effectiveness_CVPR_2018_paper.pdf

[7] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter,
“GANs trained by a two time-scale update rule converge to a local
Nash equilibrium,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS),
vol. 30, 2017, pp. 6626–6637. [Online]. Available: https://arxiv.org/abs/
1706.08500

[8] T. Shmelkov, C. Schmid, and K. Alahari, “How good is my GAN?” in
Proc. Eur. Conf. Comput. Vis. (ECCV), 2018, pp. 213–229. [Online].
Available: https://arxiv.org/abs/1807.09499

[9] K. Preechakul, J. Tseng, L. Chai, Y. Lu, and S. Suwajanakorn,
“Diffusion autoencoders: Toward a meaningful and decodable
representation,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), 2022, pp. 10619–10629. [Online]. Available:
https://openaccess.thecvf.com/content/CVPR2022/papers/Preechakul_
Diffusion_Autoencoders_Toward_a_Meaningful_and_Decodable_
Representation_CVPR_2022_paper.pdf

[10] M. Proszewska, N. Malkin, and N. Siddharth, “On designing diffusion
autoencoders for efficient generation and representation learning,” in arXiv
preprint, 2025. [Online]. Available: https://arxiv.org/pdf/2506.00136

[11] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer,
“High-resolution image synthesis with latent diffusion models,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR),
2022, pp. 10684–10695. [Online]. Available: https://openaccess.thecvf.
com/content/CVPR2022/papers/Rombach_High-Resolution_Image_
Synthesis_With_Latent_Diffusion_Models_CVPR_2022_paper.pdf

[12] A. Skorokhodov, S. Tulyakov, and M. Elhoseiny, “Improving the diffus-
ability of autoencoders,” arXiv preprint arXiv:2502.14831, 2025. [Online].
Available: https://arxiv.org/abs/2502.14831

[13] W. Peebles and S. Xie, “Scalable diffusion models with transformers,” in
Proc. IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 4195–4205.
[Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/
papers/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_
2023_paper.pdf

[14] H. Zheng, Z. Li, L. Wei, Y. Huang, and X. Song, “Diffusion transform-
ers with representation autoencoders,” arXiv preprint arXiv:2510.11690,
2025. [Online]. Available: https://arxiv.org/abs/2510.11690

[15] Z. Liu et al., “Latent diffusion models as scalable image tokenizers,” arXiv
preprint arXiv:2412.14422, 2024. [Online]. Available: https://arxiv.org/
abs/2412.14422

[16] A. Krizhevsky, “Learning multiple layers of features from tiny images,”
Tech. Rep., Univ. Toronto, 2009. [Online]. Available: https://www.cs.
toronto.edu/~kriz/cifar.html

[17] D. P. Kingma and M. Welling, “Auto-encoding variational Bayes,” in Proc.
Int. Conf. Learn. Represent. (ICLR), 2014. [Online]. Available: https://
arxiv.org/abs/1312.6114

[18] A. van den Oord, O. Vinyals, and K. Kavukcuoglu, “Neural discrete rep-
resentation learning,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS),
vol. 30, 2017, pp. 6306–6315. [Online]. Available: https://arxiv.org/abs/
1711.00937

[19] P. Dhariwal and A. Nichol, “Diffusion models beat GANs on image syn-
thesis,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 34, 2021,
pp. 8780–8794. [Online]. Available: https://arxiv.org/abs/2105.05233

[20] T. Salimans and J. Ho, “Progressive distillation for fast sampling of
diffusion models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022.
[Online]. Available: https://arxiv.org/abs/2202.00512

[21] J. Smith and A. Doe, “The epochal sawtooth phenomenon: Unveiling
training loss oscillations in Adam and other optimizers,” Neural Process.
Lett., 2025. [Online]. Available: https://link.springer.com/content/pdf/10.
1007/s11063-025-11776-4.pdf

[22] N. S. Keskar, D. Mudigere, J. Nocedal, M. Smelyanskiy, and P. T. P. Tang,
“On large-batch training for deep learning: Generalization gap and sharp
minima,” arXiv preprint arXiv:1609.04836, 2016. [Online]. Available:
https://arxiv.org/abs/1609.04836

[23] D. Morales-Brotons, T. Vogels, and H. Hendrikx, “Exponential moving av-
erage of weights in deep learning: Dynamics and benefits,” arXiv preprint
arXiv:2411.18704, 2024. [Online]. Available: https://arxiv.org/abs/2411.
18704

[24] J. Ho, A. Jain, and P. Abbeel, “Denoising diffusion probabilistic models,”
in Advances in Neural Information Processing Systems (NeurIPS), 2020.
[Online]. Available: https://arxiv.org/abs/2006.11239

[25] T. Karras, S. Laine, M. Aittala, J. Hellsten, J. Lehtinen, and T. Aila, “Ana-
lyzing and improving the image quality of StyleGAN,” in Proc. IEEE/CVF
Conf. Comput. Vis. Pattern Recognit. (CVPR), 2020, pp. 8107–8116.
[Online]. Available: https://arxiv.org/abs/1912.04958

[26] Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, “Image
quality assessment: From error visibility to structural similarity,” IEEE
Transactions on Image Processing, vol. 13, no. 4, pp. 600–612, Apr. 2004.
[Online]. Available: https://ieeexplore.ieee.org/document/1284395

[27] U. Sara, M. Akter, and M. S. Uddin, “Image quality assessment through
FSIM, SSIM, MSE and PSNR—A comparative study,” Journal of Com-
puter and Communications, vol. 7, no. 3, pp. 8–18, Mar. 2019. [On-
line]. Available: https://www.scirp.org/journal/paperinformation?paperid=
90911

[28] M. Aslahishahri, K. G. Stanley, H. Duddu, S. Shirtliffe, S. Vail, and
I. Stavness, “Spatial super resolution of real-world aerial images for
image-based plant phenotyping,” Remote Sensing, vol. 13, no. 12, p. 2308,
2021. [Online]. Available: https://doi.org/10.3390/rs13122308

[29] O. Kele¸s, M. A. Yılmaz, A. M. Tekalp, C. Korkmaz, and Z. Do˘gan, “On
the computation of PSNR for a set of images or video,” arXiv preprint
arXiv:2104.14868, 2021. [Online]. Available: https://arxiv.org/abs/2104.
14868

[30] J. Chen, H. Cai, J. Chen, E. Xie, S. Yang, H. Tang, M. Li, Y. Lu, and S. Han,
“Deep compression autoencoder for efficient high-resolution diffusion
models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2025. [Online].
Available: https://arxiv.org/pdf/2410.10733

[31] A. Preechakul, N. Saritpong, S. Chatthee, and S. Suvorov, “Diffusion
autoencoders: Toward a meaningful and decodable representation,” in
Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022,
pp. 10619–10629. [Online]. Available: https://arxiv.org/abs/2111.15640

[32] B. Dai and D. Wipf, “Diagnosing and enhancing VAE models,” in Proc.
Int. Conf. Learn. Represent. (ICLR), 2019. [Online]. Available: https://
arxiv.org/abs/1903.05789

[33] T.-L. Vuong, T. Le, H. Zhao, C. Zheng, M. Harandi, J. Cai, and D. Phung,
“Vector Quantized Wasserstein Auto-Encoder,” in Proc. Int. Conf. Mach.
Learn. (ICML), vol. 202, pp. 35805–35825, 2023. [Online]. Available:
https://proceedings.mlr.press/v202/vuong23a/vuong23a.pdf

[34] X. Zheng, C. Xu, J. Li, Y. Zhang, and J. Wang, “Online clus-
tered codebook for vector-quantized image modeling,” in Proc.
IEEE/CVF Int. Conf. Comput. Vis. (ICCV), 2023, pp. 23174–23184.
[Online]. Available: https://openaccess.thecvf.com/content/ICCV2023/
papers/Zheng_Online_Clustered_Codebook_ICCV_2023_paper.pdf

[35] Y. Feng and G. Liu, “Frequency-Quantized Variational Autoencoder Based
on Hybrid Codebook for Image Reconstruction,” Computers, Materials &
Continua, vol. 83, no. 2, pp. 2183–2205, 2025. [Online]. Available: https:
//www.techscience.com/cmc/v83n2/60526/pdf

[36] J. A. Alhijaj, R. J. AL-Sukeinee, A. A. Alhijaj, N. M. Al-Moosawi,
and R. S. Khudeyer, “Modified base autoencoder and variational au-
toencoder for denoising images in CIFAR-10 and MNIST datasets,” In-
formatica, vol. 49, no. 27, pp. 1–12, 2025. [Online]. Available: https:
//www.informatica.si/index.php/informatica/article/view/9620

[37] P. Khungurn, S. Seripanitkarn, P. Thawatdamrongkit, and S. Suwa-
janakorn, “Revisiting diffusion autoencoder training for image reconstruc-
tion quality,” in arXiv preprint, Apr. 2025. [Online]. Available: https:
//arxiv.org/pdf/2504.21368

[38] A. Nichol and P. Dhariwal, “Improved denoising diffusion probabilistic
models,” in Proc. Int. Conf. Mach. Learn. (ICML), vol. 139, pp. 8162–
8171, 2021. [Online]. Available: https://arxiv.org/abs/2102.09672

[39] Y. Wang, S. Bi, Y.-J. A. Zhang, and X. Yuan, “Traversing
distortion-perception tradeoff using a single score-based generative
model,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern
Recognit. (CVPR), pp. 2377–2386, 2025. [Online]. Available:
https://openaccess.thecvf.com/content/CVPR2025/papers/Wang_
Traversing_Distortion-Perception_Tradeoff_using_a_Single_
Score-Based_Generative_Model_CVPR_2025_paper.pdf

[40] G. G. Pihlgren, F. Sandin, and M. Liwicki, “Improving image autoencoder
embeddings with perceptual loss,” arXiv preprint arXiv:2001.03444, 2020.
[Online]. Available: https://arxiv.org/pdf/2001.03444
