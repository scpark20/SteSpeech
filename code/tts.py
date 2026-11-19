"""FastSpeech 2 teaching variant: frame-level continuous log-F0/log-energy."""
import torch
from torch import nn
from torch.nn import functional as F
from audio import length_mask, positional_encoding


class FFTBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv1, self.conv2 = nn.Conv1d(dim, dim * 4, 3, padding=1), nn.Conv1d(dim * 4, dim, 3, padding=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, valid):
        m = valid[:, :, None]
        q = self.norm1(x)
        x = (x + self.dropout(self.attention(q, q, q, key_padding_mask=~valid,
                                           need_weights=False)[0])) * m
        normalized = self.norm2(x) * m
        y = F.relu(self.conv1(normalized.transpose(1, 2))) * valid[:, None]
        y = self.conv2(self.dropout(y)).transpose(1, 2)
        return (x + self.dropout(y)) * m


class Predictor(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(dim, dim, 3, padding=1) for _ in range(2)])
        self.norms = nn.ModuleList([nn.LayerNorm(dim) for _ in range(2)])
        self.drop = nn.Dropout(0.1)
        self.output = nn.Linear(dim, 1)

    def forward(self, x, valid):
        for conv, norm in zip(self.convs, self.norms):
            x = self.drop(norm(F.relu(conv(x.transpose(1, 2)).transpose(1, 2))))
            x = x * valid[:, :, None]
        return self.output(x).squeeze(-1) * valid


def regulate(x, durations):
    """[B,L,D] -> [B,T,D], T_i=sum(duration_i); integer duration is nondifferentiable."""
    if durations.dtype != torch.long or (durations < 0).any():
        raise ValueError("Durations must be nonnegative int64")
    lengths = durations.sum(1)
    if (lengths <= 0).any() or lengths.max() > 20000:
        raise ValueError("Empty or excessive output length")
    rows = [torch.repeat_interleave(row, count, dim=0) for row, count in zip(x, durations)]
    return nn.utils.rnn.pad_sequence(rows, batch_first=True), lengths


class FastSpeech2(nn.Module):
    def __init__(self, vocab_size, dim=128, layers=4, n_mels=80):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim, padding_idx=0)
        self.encoder = nn.ModuleList([FFTBlock(dim) for _ in range(layers)])
        self.decoder = nn.ModuleList([FFTBlock(dim) for _ in range(layers)])
        self.duration, self.pitch, self.energy = Predictor(dim), Predictor(dim), Predictor(dim)
        self.pitch_embed, self.energy_embed = nn.Linear(1, dim), nn.Linear(1, dim)
        self.mel = nn.Linear(dim, n_mels)

    def forward(self, tokens, lengths, durations=None, pitch=None, energy=None, pace=1.):
        if pace <= 0:
            raise ValueError("pace must be positive")
        valid = length_mask(lengths, tokens.shape[1])
        x = (self.embedding(tokens) + positional_encoding(tokens.shape[1], self.embedding.embedding_dim,
                                                         tokens.device)) * valid[:, :, None]
        for block in self.encoder:
            x = block(x, valid)
        log_duration = self.duration(x, valid)
        if durations is None:
            durations = (torch.expm1(log_duration).clamp(0, 1000) / pace).round().long()
            durations = durations.clamp_min(1) * valid
        elif (durations.masked_select(~valid) != 0).any():
            raise ValueError("Padded tokens must have duration zero")
        x, mel_lengths = regulate(x, durations)
        frame_valid = length_mask(mel_lengths, x.shape[1])
        pred_pitch, pred_energy = self.pitch(x, frame_valid), self.energy(x, frame_valid)
        pitch = pred_pitch if pitch is None else pitch
        energy = pred_energy if energy is None else energy
        if pitch.shape != frame_valid.shape or energy.shape != frame_valid.shape:
            raise ValueError("Frame targets must match sum of durations")
        x = x + self.pitch_embed(pitch.unsqueeze(-1)) + self.energy_embed(energy.unsqueeze(-1))
        x = (x + positional_encoding(x.shape[1], x.shape[2], x.device)) * frame_valid[:, :, None]
        for block in self.decoder:
            x = block(x, frame_valid)
        return {"mel": self.mel(x) * frame_valid[:, :, None], "lengths": mel_lengths,
                "log_duration": log_duration, "pitch": pred_pitch, "energy": pred_energy}


def acoustic_loss(output, batch):
    frame_mask = length_mask(batch["mel_lengths"], batch["mel"].shape[1])
    token_mask = length_mask(batch["lengths"], batch["tokens"].shape[1])
    if not torch.equal(output["lengths"], batch["mel_lengths"]):
        raise ValueError("Alignment and mel lengths disagree")
    losses = {
        "mel": F.l1_loss(output["mel"][frame_mask], batch["mel"][frame_mask]),
        "duration": F.mse_loss(output["log_duration"][token_mask], torch.log1p(batch["durations"].float())[token_mask]),
        "pitch": F.mse_loss(output["pitch"][frame_mask], batch["pitch"][frame_mask]),
        "energy": F.mse_loss(output["energy"][frame_mask], batch["energy"][frame_mask]),
    }
    return sum(losses.values()), losses
