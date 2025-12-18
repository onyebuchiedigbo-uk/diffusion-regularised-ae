import torch
import torch.nn as nn
import torch.nn.functional as F

class DownBlock(nn.Module):
    """Conv-BN-ReLU x2 + strided conv downsample."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.Conv2d(out_c, out_c, 4, 2, 1)  # /2

    def forward(self, x):
        y = self.conv(x)
        return y, self.pool(y)

class UpBlock(nn.Module):
    """ConvTranspose2d upsample + concat skip + Conv-BN-ReLU x2."""
    def __init__(self, in_channels_prev_decoder, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels_prev_decoder, skip_channels, kernel_size=4, stride=2, padding=1
        )
        self.conv = nn.Sequential(
            nn.Conv2d(skip_channels*2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)             # spatial size doubles
        x = torch.cat([x, skip], dim=1)  # shapes match by design
        return self.conv(x)

class UNetAutoencoder(nn.Module):
    """
    Resolution path:
      Enc: 32x32 -> 16x16 -> 8x8 -> 4x4
      Latent: 4x4 conv to 512, then FC to latent_dim
      Dec: latent_dim -> 4x4 -> 8x8 -> 16x16 -> 32x32
    """
    def __init__(self, latent_dim=256):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder
        self.enc1 = DownBlock(3,   64)   # 32 -> 16
        self.enc2 = DownBlock(64,  128)  # 16 -> 8
        self.enc3 = DownBlock(128, 256)  # 8  -> 4

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),  # 4x4 -> 4x4
            nn.ReLU(inplace=True),
        )

        self.fc_enc = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 512 * 4 * 4)

        # Decoder
        self.up3 = UpBlock(512, 256, 256)  # 4 -> 8, skip enc3
        self.up2 = UpBlock(256, 128, 128)  # 8 -> 16, skip enc2
        self.up1 = UpBlock(128, 64,  64)   # 16 -> 32, skip enc1

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 3, 1),
            nn.Tanh(),  # outputs in [-1,1] to match normalized CIFAR-10
        )

    def encode(self, x):
        s1, p1 = self.enc1(x)        # s1:64x32x32, p1:64x16x16
        s2, p2 = self.enc2(p1)       # s2:128x16x16, p2:128x8x8
        s3, p3 = self.enc3(p2)       # s3:256x8x8,  p3:256x4x4
        bt = self.bottleneck(p3).view(p3.size(0), -1)  # (B,512*4*4)
        z  = self.fc_enc(bt)
        return z, (s1, s2, s3)

    def decode(self, z, skips):
        s1, s2, s3 = skips
        bt = self.fc_dec(z).view(z.size(0), 512, 4, 4)
        x  = self.up3(bt, s3)
        x  = self.up2(x,  s2)
        x  = self.up1(x,  s1)
        x  = self.final_conv(x)
        return x

    def forward(self, x):
        z, skips = self.encode(x)
        xr = self.decode(z, skips)
        return z, xr

class SmallTransformerDenoiser(nn.Module):
    """Lightweight Transformer denoiser on tokenised latent vectors."""
    def __init__(self, latent_dim=256, n_tokens=8, d_model=128, n_heads=4, n_layers=4):
        super().__init__()
        assert latent_dim % n_tokens == 0
        self.n_tokens  = n_tokens
        self.token_dim = latent_dim // n_tokens

        self.proj_in = nn.Linear(self.token_dim, d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, batch_first=True, dim_feedforward=4*d_model
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.proj_out = nn.Linear(d_model, self.token_dim)

        self.t_mlp = nn.Sequential(
            nn.Linear(128, d_model), nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
        )

    def forward(self, z, t):
        B = z.size(0)
        tokens = z.view(B, self.n_tokens, self.token_dim)
        h = self.proj_in(tokens)
        t_emb = positional_time_embedding(t, 128).to(z.device)
        h = h + self.t_mlp(t_emb).unsqueeze(1)
        h = self.transformer(h)
        out_tokens = self.proj_out(h)
        return out_tokens.view(B, -1)

def positional_time_embedding(timesteps, dim=128):
    device = timesteps.device
    half   = dim // 2
    freqs  = torch.exp(-math.log(1e4) * torch.arange(0, half, device=device) / half)
    args   = timesteps[:, None].float() * freqs[None, :]
    emb    = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        # Store a copy of all parameters/buffers on CPU
        self.shadow = {k: v.detach().cpu().clone()
                       for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            v_cpu = v.detach().cpu()
            if v_cpu.dtype.is_floating_point:
                # EMA only for floating-point tensors
                self.shadow[k].mul_(self.decay).add_(v_cpu, alpha=1 - self.decay)
            else:
                # For integer/bool tensors (e.g. counters), just keep latest value
                self.shadow[k] = v_cpu.clone()

    @torch.no_grad()
    def apply_to(self, model):
        self.backup = {}
        for k, v in model.state_dict().items():
            self.backup[k] = v.clone()
            if k in self.shadow:
                v.copy_(self.shadow[k].to(v.device))

    @torch.no_grad()
    def restore(self, model):
        for k, v in model.state_dict().items():
            if k in self.backup:
                v.copy_(self.backup[k])
        self.backup = {}
