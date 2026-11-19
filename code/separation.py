"""Open-Unmix-inspired magnitude estimators, ratio-mask resynthesis (not full Wiener EM)."""
import torch
from torch import nn
from torch.nn import functional as F
from audio import Spectrogram

STEMS = ("vocals", "drums", "bass", "other")


class SourceEstimator(nn.Module):
    def __init__(self, bins, dim=128):
        super().__init__()
        self.input = nn.Linear(2 * bins, dim)
        self.norm = nn.LayerNorm(dim)
        self.rnn = nn.LSTM(dim, dim // 2, num_layers=3, batch_first=True,
                           bidirectional=True, dropout=.1)
        self.output = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU(), nn.Linear(dim, 2 * bins))
        self.bins = bins

    def forward(self, magnitude):
        b, c, f, t = magnitude.shape
        x = torch.log1p(magnitude).permute(0, 3, 1, 2).reshape(b, t, c * f)
        x = torch.tanh(self.norm(self.input(x)))
        recurrent = self.rnn(x)[0]
        y = F.softplus(self.output(torch.cat((x, recurrent), -1)))
        return y.reshape(b, t, c, f).permute(0, 2, 3, 1)


class Separator(nn.Module):
    def __init__(self, n_fft=2048, hop=512, dim=128):
        super().__init__()
        self.stft = Spectrogram(n_fft, hop)
        self.sources = nn.ModuleList([SourceEstimator(n_fft // 2 + 1, dim) for _ in STEMS])

    def magnitudes(self, wave):
        if wave.ndim != 3 or wave.shape[1] != 2:
            raise ValueError("Separation requires stereo [B,2,N]")
        spectrum = self.stft(wave)
        estimates = torch.stack([source(spectrum.abs()) for source in self.sources], dim=1)
        return spectrum, estimates

    def forward(self, wave):
        spectrum, estimates = self.magnitudes(wave)
        power = estimates.square().clamp_min(1e-10)
        masks = power / power.sum(1, keepdim=True)
        return self.stft.inverse(masks * spectrum[:, None], wave.shape[-1])

    def loss(self, mixture, targets):
        _, estimates = self.magnitudes(mixture)
        truth = self.stft(targets).abs()
        return F.l1_loss(estimates, truth)


@torch.inference_mode()
def separate_long(model, wave, chunk=44100 * 6, overlap=44100):
    """wave [2,N] -> [4,2,N]. Positive window weights cover both endpoints."""
    if not 0 <= overlap < chunk:
        raise ValueError("Expected 0 <= overlap < chunk")
    total = wave.shape[-1]
    output = wave.new_zeros(4, 2, total)
    weights = wave.new_zeros(total)
    window = torch.hann_window(chunk, periodic=False, device=wave.device).clamp_min(1e-3)
    for start in range(0, total, chunk - overlap):
        valid = min(chunk, total - start)
        part = F.pad(wave[:, start:start + valid], (0, chunk - valid))
        prediction = model(part.unsqueeze(0))[0, :, :, :valid]
        output[:, :, start:start + valid] += prediction * window[:valid]
        weights[start:start + valid] += window[:valid]
        if start + valid == total:
            break
    return output / weights.clamp_min(1e-8)
