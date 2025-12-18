import torch
import math

def get_linear_betas(T, beta_start=1e-4, beta_end=2e-2, device="cpu"):
    return torch.linspace(beta_start, beta_end, T, device=device)

def get_cosine_betas(T, s=0.008, device="cpu"):
    """
    Improved DDPM cosine schedule (Nichol & Dhariwal, 2021).
    """
    steps = T + 1
    x = torch.linspace(0, T, steps, device=device) / T
    alphas_cumprod = torch.cos(((x + s) / (1 + s)) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-4, 0.999)

def prepare_alphas(betas):
    alphas = 1 - betas
    alphas_cp = torch.cumprod(alphas, 0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cp,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cp),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1 - alphas_cp),
    }

def q_sample_batch(z0, t, d, noise=None):
    """
    Forward diffusion: z_t = sqrt(alpha_bar_t) * z0 + sqrt(1 - alpha_bar_t) * eps.
    t: (B,) indices in [0, T-1].
    """
    if noise is None:
        noise = torch.randn_like(z0)
    sa = d["sqrt_alphas_cumprod"].index_select(0, t).unsqueeze(1)
    so = d["sqrt_one_minus_alphas_cumprod"].index_select(0, t).unsqueeze(1)
    return sa * z0 + so * noise, noise
