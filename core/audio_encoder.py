"""Audio encoder for TranscendPlexity multi-modal AGI.

Mel spectrogram → transformer encoder → embeddings.
Supports speech, music, environmental sounds.
Output shape: [batch, num_frames, hidden_dim] for fusion with other modalities.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class MelSpectrogramFrontend(nn.Module):
    """Convert raw audio waveform to mel spectrogram features.

    Uses learnable filterbank for end-to-end training.
    Falls back to simple STFT + linear mel approximation (no torchaudio dependency).
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 512,
        hop_length: int = 160,
        n_mels: int = 80,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Learnable mel filterbank (avoids torchaudio dependency)
        self.mel_proj = nn.Linear(n_fft // 2 + 1, n_mels, bias=False)

        # Initialize with approximate mel spacing
        with torch.no_grad():
            freqs = torch.linspace(0, sample_rate / 2, n_fft // 2 + 1)
            mel_freqs = 2595 * torch.log10(1 + freqs / 700)
            mel_points = torch.linspace(mel_freqs[0], mel_freqs[-1], n_mels + 2)
            for i in range(n_mels):
                low, center, high = mel_points[i], mel_points[i + 1], mel_points[i + 2]
                weights = torch.zeros(n_fft // 2 + 1)
                for j in range(n_fft // 2 + 1):
                    if low <= mel_freqs[j] <= center:
                        weights[j] = (mel_freqs[j] - low) / max(center - low, 1e-8)
                    elif center < mel_freqs[j] <= high:
                        weights[j] = (high - mel_freqs[j]) / max(high - center, 1e-8)
                self.mel_proj.weight.data[i] = weights

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        """Convert waveform to mel spectrogram.

        Args:
            waveform: [B, T] raw audio samples

        Returns:
            [B, num_frames, n_mels] mel spectrogram
        """
        # Pad to n_fft
        if waveform.shape[-1] < self.n_fft:
            waveform = F.pad(waveform, (0, self.n_fft - waveform.shape[-1]))

        # STFT via unfold + FFT
        window = torch.hann_window(self.n_fft, device=waveform.device)
        # Unfold into frames
        frames = waveform.unfold(-1, self.n_fft, self.hop_length)  # [B, num_frames, n_fft]
        frames = frames * window

        # FFT
        spec = torch.fft.rfft(frames, dim=-1)  # [B, num_frames, n_fft//2+1]
        mag = spec.abs()  # Magnitude spectrogram

        # Apply mel filterbank
        mel = self.mel_proj(mag)  # [B, num_frames, n_mels]

        # Log mel (with floor for numerical stability)
        mel = torch.log(mel.clamp(min=1e-8))

        return mel


class AudioTransformerBlock(nn.Module):
    """Transformer block for audio processing."""

    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class AudioEncoder(nn.Module):
    """Audio encoder for TranscendPlexity.

    Converts raw waveforms or pre-computed spectrograms into
    embeddings compatible with the compound braid fusion layer.

    Args:
        hidden_dim: Output embedding dimension (must match model hidden_dim)
        n_mels: Number of mel frequency bins
        num_layers: Number of transformer layers
        num_heads: Number of attention heads
        max_frames: Maximum number of audio frames (for positional encoding)
        sample_rate: Audio sample rate in Hz
        dropout: Dropout rate
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        n_mels: int = 80,
        num_layers: int = 4,
        num_heads: int = 8,
        max_frames: int = 3000,
        sample_rate: int = 16000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_mels = n_mels

        # Audio frontend
        self.frontend = MelSpectrogramFrontend(
            sample_rate=sample_rate,
            n_mels=n_mels,
        )

        # Project mel features to hidden dim
        self.input_proj = nn.Linear(n_mels, hidden_dim)

        # Special tokens
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.modality_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Sinusoidal positional encoding (better for variable-length audio)
        self.register_buffer('pos_embed', self._sinusoidal_pos(max_frames + 1, hidden_dim))

        # Transformer layers
        self.layers = nn.ModuleList([
            AudioTransformerBlock(hidden_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(hidden_dim)
        self.adaptive_proj = nn.Linear(hidden_dim, hidden_dim)

        # Subsampling conv (reduce frame rate by 4× for efficiency)
        self.subsample = nn.Sequential(
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )

    @staticmethod
    def _sinusoidal_pos(max_len: int, dim: int) -> torch.Tensor:
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        return pe

    def forward(
        self,
        audio: torch.Tensor,
        is_spectrogram: bool = False,
        target_seq_len: Optional[int] = None,
    ) -> dict:
        """Encode audio into embeddings.

        Args:
            audio: [B, T] raw waveform OR [B, num_frames, n_mels] spectrogram
            is_spectrogram: If True, skip mel extraction
            target_seq_len: Pool output to this length for alignment

        Returns:
            dict with:
                'embeddings': [B, seq_len, hidden_dim]
                'cls_token': [B, hidden_dim]
                'num_frames': int
        """
        B = audio.shape[0]

        # Extract mel spectrogram if raw waveform
        if not is_spectrogram:
            mel = self.frontend(audio)  # [B, num_frames, n_mels]
        else:
            mel = audio

        # Project to hidden dim
        x = self.input_proj(mel)  # [B, num_frames, D]

        # Subsample (reduce temporal resolution by 4×)
        x = x.transpose(1, 2)  # [B, D, T]
        x = self.subsample(x)  # [B, D, T/4]
        x = x.transpose(1, 2)  # [B, T/4, D]

        num_frames = x.shape[1]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T/4+1, D]

        # Add positional encoding
        x = x + self.pos_embed[:, :x.shape[1]]

        # Add modality token
        x = x + self.modality_token

        # Transformer blocks
        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # Split CLS and frame tokens
        cls_out = x[:, 0]  # [B, D]
        frame_out = x[:, 1:]  # [B, T/4, D]

        # Adaptive pooling
        if target_seq_len is not None and frame_out.shape[1] != target_seq_len:
            frame_out = frame_out.transpose(1, 2)
            frame_out = F.adaptive_avg_pool1d(frame_out, target_seq_len)
            frame_out = frame_out.transpose(1, 2)

        frame_out = self.adaptive_proj(frame_out)

        return {
            'embeddings': frame_out,
            'cls_token': cls_out,
            'num_frames': num_frames,
        }

    def encode_tokens_as_audio(self, tokens: torch.Tensor, sample_rate: int = 16000) -> dict:
        """Encode a token sequence as if it were an audio signal.

        Useful for cross-modal pre-training: treat any sequence as
        a 'sound' and let the audio encoder find structure.

        Args:
            tokens: [B, T] integer tokens

        Returns:
            Same as forward()
        """
        # Convert tokens to pseudo-waveform via embedding lookup + interpolation
        freqs = (tokens.float() + 1) * 100  # Map tokens to frequencies
        t = torch.arange(sample_rate // 10, device=tokens.device).float() / sample_rate
        # Generate micro-tones for each token
        waves = []
        for i in range(tokens.shape[1]):
            wave = torch.sin(2 * math.pi * freqs[:, i:i+1] * t.unsqueeze(0))
            waves.append(wave)
        waveform = torch.cat(waves, dim=1)  # [B, T * sr/10]
        return self.forward(waveform)
