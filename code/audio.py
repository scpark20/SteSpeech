"""Audio contracts: wave [C,N], batched wave [B,C,N], spectrum [B,C,F,T]."""
import math
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.signal import resample_poly
import torch
from torch import nn


def read_audio(path, sample_rate, mono=False):
    path = Path(path)
    if path.suffix.lower() == ".pcm":
        if sample_rate != 16000:
            raise ValueError("KsponSpeech PCM is 16 kHz little-endian int16")
        array = np.fromfile(path, dtype="<i2").astype(np.float32)[:, None] / 32768
        original_sr = 16000
    else:
        array, original_sr = sf.read(path, dtype="float32", always_2d=True)
    if len(array) == 0 or not np.isfinite(array).all():
        raise ValueError("Empty/non-finite audio")
    if original_sr != sample_rate:
        gcd = math.gcd(original_sr, sample_rate)
        array = resample_poly(array, sample_rate // gcd, original_sr // gcd, axis=0)
    wave = torch.from_numpy(np.ascontiguousarray(array.T)).float()
    return wave.mean(0, keepdim=True) if mono else wave


def write_audio(path, wave, sample_rate):
    """Float WAV preserves amplitudes; never normalize each separated stem."""
    wave = wave.detach().cpu().float()
    if wave.ndim != 2 or not torch.isfinite(wave).all():
        raise ValueError("Expected finite [C,N] waveform")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, wave.T.numpy(), sample_rate, subtype="FLOAT")


class Spectrogram(nn.Module):
    def __init__(self, n_fft=1024, hop=256):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        self.register_buffer("window", torch.hann_window(n_fft))

    def forward(self, wave):
        shape = wave.shape[:-1]
        z = torch.stft(wave.reshape(-1, wave.shape[-1]).float(), self.n_fft,
                       self.hop, window=self.window, center=True,
                       pad_mode="constant", return_complex=True)
        return z.reshape(*shape, *z.shape[-2:])

    def inverse(self, z, length):
        shape = z.shape[:-2]
        wave = torch.istft(z.reshape(-1, *z.shape[-2:]), self.n_fft,
                          self.hop, window=self.window, center=True, length=length)
        return wave.reshape(*shape, length)


class LogMel(nn.Module):
    """HTK triangular mel, magnitude (not power), natural log; one shared convention."""
    def __init__(self, sr=22050, n_fft=1024, hop=256, n_mels=80):
        super().__init__()
        self.stft = Spectrogram(n_fft, hop)
        self.sr, self.hop, self.n_mels = sr, hop, n_mels
        edges = torch.linspace(0, 2595 * math.log10(1 + sr / 1400), n_mels + 2)
        edges = 700 * (10 ** (edges / 2595) - 1)
        freq = torch.linspace(0, sr / 2, n_fft // 2 + 1)
        up = (freq[None] - edges[:-2, None]) / (edges[1:-1] - edges[:-2])[:, None]
        down = (edges[2:, None] - freq[None]) / (edges[2:] - edges[1:-1])[:, None]
        self.register_buffer("filters", torch.minimum(up, down).clamp_min(0))

    def forward(self, wave):
        magnitude = self.stft(wave).abs()
        return torch.log(torch.matmul(self.filters, magnitude).clamp_min(1e-5))


def length_mask(lengths, width=None):
    width = int(lengths.max()) if width is None else width
    return torch.arange(width, device=lengths.device)[None] < lengths[:, None]


def positional_encoding(length, dim, device):
    if dim % 2:
        raise ValueError("Position encoding requires an even hidden dimension")
    position = torch.arange(length, device=device).float()[:, None]
    frequency = torch.exp(torch.arange(0, dim, 2, device=device) * (-math.log(10000) / dim))
    pe = torch.zeros(length, dim, device=device)
    pe[:, 0::2], pe[:, 1::2] = (position * frequency).sin(), (position * frequency).cos()
    return pe


def pitch_energy(wave, mel):
    """Autocorrelation F0 baseline. Zero denotes unvoiced; not a production tracker."""
    z = mel.stft(wave).abs()
    energy = (z.square().mean(-2) + 1e-8).sqrt().squeeze(0)
    n_fft, hop, sr = mel.stft.n_fft, mel.hop, mel.sr
    frames = torch.nn.functional.pad(wave[0], (n_fft // 2, n_fft // 2)).unfold(0, n_fft, hop)
    frames = (frames - frames.mean(-1, keepdim=True)) * mel.stft.window
    spectrum = torch.fft.rfft(frames, n=2 * n_fft)
    correlation = torch.fft.irfft(spectrum.abs().square(), n=2 * n_fft)[:, :n_fft]
    lo, hi = int(sr / 500), min(int(sr / 60), n_fft - 1)
    peaks, lag = correlation[:, lo:hi + 1].max(-1)
    voiced = (peaks / correlation[:, 0].clamp_min(1e-8) > 0.3) & (energy > 1e-3)
    f0 = torch.where(voiced, sr / (lag + lo).float(), 0.)
    return f0, energy
