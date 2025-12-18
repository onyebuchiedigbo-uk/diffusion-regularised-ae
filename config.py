import os
import torch 
# Reproducibility
SEED = 42

# Paths
CHECKPOINT_DIR = "checkpoints"
RESULTS_DIR = "results"
FIGURES_DIR = "figures"
DATA_DIR = "./data"

# Create directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Data
DATASET = "CIFAR-10"
IMAGE_SIZE = 32
NUM_CHANNELS = 3
BATCH_SIZE = 128
VAL_BATCH_SIZE = 256
NUM_WORKERS = 2

# Autoencoder
LATENT_DIM = 256
AE_EPOCHS = 20
AE_LR = 2e-4
AE_WEIGHT_DECAY = 1e-6

# Diffusion
DIFFUSION_TIMESTEPS = 100
SCHEDULE_NAME = "cosine"  # "linear" or "cosine"
BETA_START = 1e-4
BETA_END = 2e-2
COSINE_S = 0.008

# Denoiser
DENOISER_LATENT_DIM = LATENT_DIM
DENOISER_N_TOKENS = 8
DENOISER_D_MODEL = 128
DENOISER_N_HEADS = 4
DENOISER_N_LAYERS = 4
DENOISER_LR = 2e-4
DENOISER_WEIGHT_DECAY = 1e-6

# Joint training
JOINT_EPOCHS = 30
JOINT_LR = 2e-4
JOINT_WEIGHT_DECAY = 1e-6
GRAD_NORM_MAX = 5.0
LR_SCHEDULER_T_MAX = 30

# EMA
EMA_DECAY = 0.999

# Metrics
N_FID_SAMPLES = 5000
FID_RESIZE_SIZE = (299, 299)

# Ablation
AB_LATENT_DIMS = [64, 128, 256]
AB_SCHEDULES = ["linear", "cosine"]
AB_AE_EPOCHS = 10
AB_JOINT_EPOCHS = 10
AB_FID_SAMPLES = 2000

# High-res reconstruction
TILE_SIZE = 32  # matches CIFAR-10 resolution
