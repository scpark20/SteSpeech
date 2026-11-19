"""Teaching Conformer-CTC: 1-D subsampling + sinusoidal position, explicit deviations."""
import math
import torch
from torch import nn
from torch.nn import functional as F
from audio import length_mask, positional_encoding


class FeedForward(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim * 4), nn.SiLU(),
                                 nn.Dropout(dropout), nn.Linear(dim * 4, dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class ConformerBlock(nn.Module):
    def __init__(self, dim, heads, kernel=15, dropout=0.1):
        super().__init__()
        self.ff1, self.ff2 = FeedForward(dim, dropout), FeedForward(dim, dropout)
        self.attn_norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.conv_norm = nn.LayerNorm(dim)
        self.point_in = nn.Conv1d(dim, 2 * dim, 1)
        self.depth = nn.Conv1d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        # Per-frame LayerNorm avoids batch statistics polluted by padded utterances.
        self.depth_norm = nn.LayerNorm(dim)
        self.point_out = nn.Conv1d(dim, dim, 1)
        self.drop, self.out_norm = nn.Dropout(dropout), nn.LayerNorm(dim)

    def forward(self, x, valid):
        mask = valid[:, :, None]
        x = (x + 0.5 * self.ff1(x)) * mask
        q = self.attn_norm(x)
        a = self.attn(q, q, q, key_padding_mask=~valid, need_weights=False)[0]
        x = (x + self.drop(a)) * mask
        c = self.point_in(self.conv_norm(x).transpose(1, 2))
        c = F.glu(c, dim=1) * valid[:, None]
        c = self.depth(c).transpose(1, 2)
        c = F.silu(self.depth_norm(c)) * mask
        x = (x + self.drop(self.point_out(c.transpose(1, 2)).transpose(1, 2))) * mask
        return self.out_norm(x + 0.5 * self.ff2(x)) * mask


class ConformerCTC(nn.Module):
    def __init__(self, vocab_size, n_mels=80, dim=144, layers=4, heads=4, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(n_mels, dim, 3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(dim, dim, 3, stride=2, padding=1)
        self.blocks = nn.ModuleList([ConformerBlock(dim, heads, dropout=dropout) for _ in range(layers)])
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, features, lengths):
        # features [B,T,80] -> logits [B,ceil(T/4),V]
        x = features * length_mask(lengths, features.shape[1])[:, :, None]
        x = F.silu(self.conv1(x.transpose(1, 2)))
        lengths = (lengths + 1) // 2
        x = x * length_mask(lengths, x.shape[-1])[:, None]
        x = F.silu(self.conv2(x)).transpose(1, 2)
        lengths = (lengths + 1) // 2
        valid = length_mask(lengths, x.shape[1])
        x = (x + positional_encoding(x.shape[1], x.shape[2], x.device)) * valid[:, :, None]
        for block in self.blocks:
            x = block(x, valid)
        return self.output(x), lengths


def ctc_loss(logits, lengths, targets, target_lengths):
    for i, n in enumerate(target_lengths.tolist()):
        y = targets[i, :n]
        if n == 0 or (y == 0).any():
            raise ValueError("CTC target must be nonempty and cannot contain blank=0")
        repeats = (y[1:] == y[:-1]).sum()
        if n + repeats > lengths[i]:
            raise ValueError("CTC alignment impossible after subsampling")
    return F.ctc_loss(logits.float().log_softmax(-1).transpose(0, 1), targets,
                      lengths, target_lengths, blank=0, zero_infinity=False)


def collapse(path, blank=0):
    result, previous = [], None
    for token in path:
        if token != blank and token != previous:
            result.append(token)
        previous = token
    return result


def ctc_forward(log_probs, tokens, blank=0):
    """Small differentiable log-domain DP for teaching; production loss uses PyTorch."""
    labels = [blank]
    for token in tokens:
        labels.extend((int(token), blank))
    if len(tokens) == 0:
        return -log_probs[:, blank].sum()
    state = log_probs.new_full((len(labels),), -torch.inf)
    state[0], state[1] = log_probs[0, blank], log_probs[0, labels[1]]
    for frame in log_probs[1:]:
        next_state = []
        for s, label in enumerate(labels):
            choices = [state[s]]
            if s > 0:
                choices.append(state[s - 1])
            if s > 1 and label != blank and label != labels[s - 2]:
                choices.append(state[s - 2])
            choices = torch.stack(choices)
            value = log_probs.new_tensor(-torch.inf) if torch.isneginf(choices).all() else torch.logsumexp(choices, 0)
            next_state.append(value + frame[label])
        state = torch.stack(next_state)
    return -torch.logsumexp(state[-2:], 0)


def greedy(logits, lengths):
    return [collapse(row[:n].tolist()) for row, n in zip(logits.argmax(-1), lengths)]


def prefix_beam(log_probs, width=8, blank=0):
    """Exact token expansion, approximate pruning by beam width; no external LM."""
    def add(a, b):
        if a == -math.inf:
            return b
        if b == -math.inf:
            return a
        hi = max(a, b)
        return hi + math.log(math.exp(a - hi) + math.exp(b - hi))
    beam = {(): (0., -math.inf)}
    for frame in log_probs.detach().cpu().tolist():
        new = {}
        def accumulate(prefix, pb=-math.inf, pn=-math.inf):
            old_b, old_n = new.get(prefix, (-math.inf, -math.inf))
            new[prefix] = (add(old_b, pb), add(old_n, pn))
        for prefix, (pb, pn) in beam.items():
            total = add(pb, pn)
            accumulate(prefix, pb=total + frame[blank])
            for token, score in enumerate(frame):
                if token == blank:
                    continue
                if prefix and token == prefix[-1]:
                    accumulate(prefix, pn=pn + score)
                    accumulate(prefix + (token,), pn=pb + score)
                else:
                    accumulate(prefix + (token,), pn=total + score)
        beam = dict(sorted(new.items(), key=lambda kv: add(*kv[1]), reverse=True)[:width])
    return list(max(beam, key=lambda key: add(*beam[key])))


def specaugment(x, lengths, time_width=20, freq_width=8):
    x = x.clone()
    for row, n in zip(x, lengths.tolist()):
        for axis, size, max_width in [(0, n, time_width), (1, x.shape[-1], freq_width)]:
            width = int(torch.randint(0, min(size, max_width) + 1, (1,)))
            start = int(torch.randint(0, size - width + 1, (1,)))
            if axis == 0:
                row[start:start + width] = 0
            else:
                row[:n, start:start + width] = 0
    return x
