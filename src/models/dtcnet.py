"""DTCNet: Dilated-Transposed CNN for ECoG → finger flexion.

Reproduces DTCNet (Wang et al., 2025, Frontiers in Computational Neuroscience).
Architecture: encoder-decoder with dilated 1D convolutions + MaxPool downsampling,
sawtooth dilation pattern (1,2,3,1,2), ConvTranspose1d upsampling in decoder,
concatenative skip connections.

Key design (confirmed from paper text):
- Encoder: Conv1d + LayerNorm + GELU + Dropout + MaxPool (one block per stage)
- Decoder: ConvTranspose1d (restore resolution) + cat(skip) + Conv1d + LN + GELU
- Skip connections: concatenation of encoder output with decoder upsample
- The name "Dilated-Transposed" = dilated encoder + transposed-conv decoder

Input:  (batch, N_ch*40, T)  — flattened Morlet spectrogram from pipeline,
        OR (batch, N_ch, 40, T) — 4D pipeline format (auto-reshaped)
Output: (batch, n_channels_out, T)

Encoder stages (dims = [n_in, 64, 64, 128, 256, 512, 512]):
  Stage 0  FR:  Conv(n_in→64,  k=3, d=1) + LN + GELU + Drop   [no MaxPool]
  Stage 1  Enc: Conv(64→64,   k=7, d=1) + LN + GELU + Drop + MaxPool(2)
  Stage 2  Enc: Conv(64→128,  k=7, d=2) + LN + GELU + Drop + MaxPool(2)
  Stage 3  Enc: Conv(128→256, k=5, d=3) + LN + GELU + Drop + MaxPool(2)
  Stage 4  Enc: Conv(256→512, k=5, d=1) + LN + GELU + Drop + MaxPool(2)
  Stage 5  Enc: Conv(512→512, k=5, d=2) + LN + GELU + Drop + MaxPool(2) [bottleneck]

Decoder (symmetric, skip from enc stages 4→0):
  ConvTranspose(s→s, k=2, stride=2) + cat(skip) + Conv(2s→s//2, k, d) + LN + GELU

Reference:
    Wang et al. "DTCNet: Dilated-Transposed Convolutional Neural Network for
    Motor Decoding from Electrocorticography"
    Frontiers in Computational Neuroscience, 2025.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _LN1d(nn.Module):
    """Channel-wise LayerNorm for (B, C, T) tensors."""
    def __init__(self, n_ch: int):
        super().__init__()
        self.ln = nn.LayerNorm(n_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x.transpose(-2, -1)).transpose(-2, -1)


class _EncBlock(nn.Module):
    """Encoder block: Conv1d (same-pad) → LN → GELU → Dropout → MaxPool."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int,
                 dilation: int = 1, dropout: float = 0.1, pool: int = 2):
        super().__init__()
        pad = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, dilation=dilation,
                              padding=pad, bias=False)
        self.ln   = _LN1d(out_ch)
        self.drop = nn.Dropout(p=dropout)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.drop(F.gelu(self.ln(self.conv(x)))))


class _DecBlock(nn.Module):
    """Decoder block: ConvTranspose1d (upsample) → cat(skip) → Conv → LN → GELU."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int,
                 kernel: int, dilation: int = 1, dropout: float = 0.1):
        super().__init__()
        # Upsample: doubles temporal resolution, halves channels
        self.up   = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=2, stride=2)
        # After cat with skip: (in_ch + skip_ch) → out_ch
        cat_ch = in_ch + skip_ch
        pad    = dilation * (kernel - 1) // 2
        self.conv = nn.Conv1d(cat_ch, out_ch, kernel, dilation=dilation,
                              padding=pad, bias=False)
        self.ln   = _LN1d(out_ch)
        self.drop = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        # Trim/pad to match skip length (ConvTranspose may be ±1 off)
        if x.shape[-1] != skip.shape[-1]:
            x = x[..., :skip.shape[-1]]
        x = torch.cat([x, skip], dim=1)
        return self.drop(F.gelu(self.ln(self.conv(x))))


class DTCNet(nn.Module):
    """Dilated-Transposed CNN encoder-decoder for continuous ECoG decoding."""

    _DIMS    = [64,  64,  128, 256, 512, 512]   # output ch per stage
    _KERNELS = [3,   7,   7,   5,   5,   5  ]   # conv kernel per stage
    _DILATS  = [1,   1,   2,   3,   1,   2  ]   # dilation per stage

    def __init__(self, n_channels_in: int, n_channels_out: int = 5,
                 dropout: float = 0.1):
        super().__init__()

        dims    = self._DIMS
        kernels = self._KERNELS
        dilats  = self._DILATS

        # ---- Encoder ----
        in_chs = [n_channels_in] + dims[:-1]
        # Stage 0 (Feature Reduction): no MaxPool
        self.fr  = _EncBlock(in_chs[0], dims[0], kernels[0], dilats[0], dropout, pool=1)
        # Stages 1-5: each with MaxPool(2)
        self.enc = nn.ModuleList([
            _EncBlock(in_chs[i], dims[i], kernels[i], dilats[i], dropout, pool=2)
            for i in range(1, 6)
        ])

        # ---- Decoder ----
        # dec[j].up upsamples from prev-dec output; then cat with skip; then conv → out
        # dec[0]: in=enc5(512), skip=enc4(512), out=256
        # dec[1]: in=dec0(256), skip=enc3(256), out=128
        # dec[2]: in=dec1(128), skip=enc2(128), out=64
        # dec[3]: in=dec2(64),  skip=enc1(64),  out=64
        # dec[4]: in=dec3(64),  skip=fr(64),    out=64
        dec_in   = [512, 256, 128,  64,  64]   # input from prev dec (or bottleneck)
        dec_skip = [512, 256, 128,  64,  64]   # skip from enc stages enc[3..0] then fr
        dec_out  = [256, 128,  64,  64,  64]   # output channels
        dec_k    = [  5,   5,   7,   7,   3]   # mirror of enc kernels 4→0
        dec_d    = [  1,   3,   2,   1,   1]   # mirror of enc dilations 4→0

        self.dec = nn.ModuleList([
            _DecBlock(dec_in[j], dec_skip[j], dec_out[j], dec_k[j], dec_d[j], dropout)
            for j in range(5)
        ])

        # ---- Head ----
        self.head = nn.Conv1d(64, n_channels_out, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # Encoder
        x0 = self.fr(x)           # (B, 64, T)
        x1 = self.enc[0](x0)      # (B, 64,  T//2)
        x2 = self.enc[1](x1)      # (B, 128, T//4)
        x3 = self.enc[2](x2)      # (B, 256, T//8)
        x4 = self.enc[3](x3)      # (B, 512, T//16)
        x5 = self.enc[4](x4)      # (B, 512, T//32)  — bottleneck

        # Decoder (skip connections from enc outputs)
        d = self.dec[0](x5, x4)   # 512 up + skip(512) → 256
        d = self.dec[1](d,  x3)   # 256 up + skip(256) → 128
        d = self.dec[2](d,  x2)   # 128 up + skip(128) → 64
        d = self.dec[3](d,  x1)   # 64  up + skip(64)  → 64
        d = self.dec[4](d,  x0)   # 64  up + skip(64)  → 64

        return self.head(d)        # (B, n_out, T)
