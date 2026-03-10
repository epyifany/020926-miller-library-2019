"""NestedUNet: 3-level UNet++ for ECoG → finger flexion.

Reproduces DeepFingerNet (Tao et al., 2025, IEEE TIM).
Architecture: 3 nested U-Nets with dense skip connections (UNet++ style).

Encoder: single Conv→LN→GELU per level (k=3), channels [32, 64, 128, 256]
Decoder: DoubleConv (two Conv→LN→GELU blocks) per node, k=1 (pointwise projection)
Upsampling: ConvTranspose1d (first/direct decoder path); Upsample (nested paths)

This design gives ~320K parameters with N=62 input channels, matching the
paper's reported 1.30 MB / ~325K params.

Input:  (batch, N, T)           — raw ECoG, N electrodes at 100 Hz
        OR (batch, N, 1, T)     — 4-D pipeline format (n_wavelets=0 mode)
Output: (batch, n_channels_out, T)

Reference:
    Tao et al. "DeepFingerNet: A Nested U-Net for ECoG-based Finger Decoding"
    IEEE Transactions on Instrumentation and Measurement, 2025.
    DOI: 10.1109/TIM.2025.3644562

Architecture notes:
  Decoder channel inputs (from paper Table I, matched as close as possible):
    Dec4 (X21): ct(256→128) + enc2(128) = 256  → 128   [direct, k=1 DoubleConv]
    Dec2 (X11): ct(128→64)  + enc1(64)  = 128  → 64    [direct, k=1 DoubleConv]
    Dec1 (X01): ct(64→32)   + enc0(32)  = 64   → 32    [direct, k=1 DoubleConv]
    Dec5 (X12): up(X21=128) + enc1(64) + X11(64) = 256 → 64   [nested, k=1]
    Dec3 (X02): up(X11=64)  + enc0(32) + X01(32) = 128 → 32   [nested, k=1]
    Dec6 (X03): up(X12=64)  + enc0(32) + X01(32) + X02(32) = 160 → 32 [nested]
                (paper Table I lists 192 for Dec6; our formula gives 160 — 32ch gap
                likely from an unrecoverable implementation detail in the figure)
"""

import torch
import torch.nn as nn


class _LN(nn.Module):
    """LayerNorm on channel dim for (B, C, T) tensors."""

    def __init__(self, n_ch):
        super().__init__()
        self.ln = nn.LayerNorm(n_ch)

    def forward(self, x):
        return self.ln(x.transpose(-2, -1)).transpose(-2, -1)


def _conv_ln_gelu(in_ch, out_ch, k=3):
    """Conv1d(padding=same, bias=True) → LayerNorm → GELU."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, k, padding="same", bias=True),
        _LN(out_ch),
        nn.GELU(),
    )


class _DoubleConv(nn.Module):
    """Two pointwise (k=1) conv blocks: (Conv1d → LN → GELU) × 2.

    Used in decoder for feature projection / channel combination.
    k=1 keeps decoder param count low, matching the paper's ~325K total.
    """

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.c1 = _conv_ln_gelu(in_ch, out_ch, k=1)
        self.c2 = _conv_ln_gelu(out_ch, out_ch, k=1)

    def forward(self, x):
        return self.c2(self.c1(x))


class NestedUNet(nn.Module):
    """3-level UNet++ for ECoG decoding (DeepFingerNet, Tao et al. 2025).

    Parameters
    ----------
    n_channels_in : int
        Number of input channels. For faithful DeepFingerNet reproduction
        this is N_electrodes (62 for BCI-IV), NOT N_electrodes × N_freqs.
    n_channels_out : int
        Number of output fingers (5).
    base_ch : int
        Base channel width. Encoder channels = [base_ch, 2x, 4x, 8x].
    kernel_size : int
        Kernel size for encoder convolutions (k=3 in paper).
    """

    def __init__(
        self,
        n_channels_in: int,
        n_channels_out: int = 5,
        base_ch: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()
        C = [base_ch * (2**i) for i in range(4)]  # [32, 64, 128, 256]
        k = kernel_size  # encoder kernel size (k=3)

        # ── Encoder: single Conv→LN→GELU per level, k=3 ──────────────────────
        # (paper Table I: Encoder 1/2/3 each have a single input→output mapping)
        self.sr   = _conv_ln_gelu(n_channels_in, C[0], k)   # N → 32
        self.enc1 = _conv_ln_gelu(C[0], C[1], k)            # 32 → 64
        self.enc2 = _conv_ln_gelu(C[1], C[2], k)            # 64 → 128
        self.enc3 = _conv_ln_gelu(C[2], C[3], k)            # 128 → 256 (bottleneck)
        self.pool = nn.MaxPool1d(2)

        # ── First (direct) decoder path: ConvTranspose + DoubleConv(k=1) ──────
        # ConvTranspose halves channels while doubling temporal resolution.
        # Concatenate with same-level encoder output → DoubleConv(k=1).

        self.ct21  = nn.ConvTranspose1d(C[3], C[2], kernel_size=2, stride=2)
        self.dec21 = _DoubleConv(C[2] + C[2], C[2])   # 256 → 128  (Dec4)

        self.ct11  = nn.ConvTranspose1d(C[2], C[1], kernel_size=2, stride=2)
        self.dec11 = _DoubleConv(C[1] + C[1], C[1])   # 128 → 64   (Dec2)

        self.ct01  = nn.ConvTranspose1d(C[1], C[0], kernel_size=2, stride=2)
        self.dec01 = _DoubleConv(C[0] + C[0], C[0])   # 64  → 32   (Dec1)

        # ── Nested decoder paths: Upsample (no params) + DoubleConv(k=1) ─────
        self.up = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)

        # up(X21=128) + enc1(64) + X11(64) = 256  (Dec5)
        self.dec12 = _DoubleConv(C[2] + C[1] + C[1], C[1])   # 256 → 64

        # up(X11=64) + enc0(32) + X01(32) = 128  (Dec3)
        self.dec02 = _DoubleConv(C[1] + C[0] + C[0], C[0])   # 128 → 32

        # up(X12=64) + enc0(32) + X01(32) + X02(32) = 160  (Dec6; paper: 192)
        self.dec03 = _DoubleConv(C[1] + C[0] * 3, C[0])      # 160 → 32

        # ── Output head ───────────────────────────────────────────────────────
        self.head = nn.Conv1d(C[0], n_channels_out, kernel_size=1)

    @staticmethod
    def _match(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Trim or zero-pad x along time dim to match ref's length (off-by-1 fix)."""
        diff = ref.shape[-1] - x.shape[-1]
        if diff == 0:
            return x
        if diff > 0:
            return torch.nn.functional.pad(x, (0, diff))
        return x[..., :ref.shape[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept 4-D input (B, N, 1, T) from n_wavelets=0 pipeline → (B, N, T)
        if x.dim() == 4:
            B, C, F, T = x.shape
            x = x.reshape(B, C * F, T)

        # ── Encode ────────────────────────────────────────────────────────────
        e0 = self.sr(x)                         # (B, 32,  T)
        e1 = self.enc1(self.pool(e0))            # (B, 64,  T//2)
        e2 = self.enc2(self.pool(e1))            # (B, 128, T//4)
        e3 = self.enc3(self.pool(e2))            # (B, 256, T//8)

        # ── Direct decoder path (ConvTranspose) ───────────────────────────────
        d21 = self.dec21(torch.cat([self._match(self.ct21(e3), e2), e2], dim=1))
        d11 = self.dec11(torch.cat([self._match(self.ct11(e2), e1), e1], dim=1))
        d01 = self.dec01(torch.cat([self._match(self.ct01(e1), e0), e0], dim=1))

        # ── Nested decoder paths (Upsample, preserves exact length) ───────────
        d12 = self.dec12(torch.cat([self.up(d21), e1,  d11       ], dim=1))
        d02 = self.dec02(torch.cat([self.up(d11), e0,  d01       ], dim=1))
        d03 = self.dec03(torch.cat([self.up(d12), e0,  d01,  d02 ], dim=1))

        return self.head(d03)                    # (B, n_targets, T)
