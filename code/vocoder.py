"""HiFi-GAN structure, compact channels; mel convention is in audio.LogMel."""
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.parametrizations import weight_norm


def conv(*args, **kwargs):
    return weight_norm(nn.Conv1d(*args, **kwargs))


class ResidualBlock(nn.Module):
    def __init__(self, channels, kernel):
        super().__init__()
        self.first = nn.ModuleList([conv(channels, channels, kernel, dilation=d,
                                        padding=(kernel - 1) * d // 2) for d in (1, 3, 5)])
        self.second = nn.ModuleList([conv(channels, channels, kernel, padding=(kernel - 1) // 2) for _ in range(3)])

    def forward(self, x):
        for first, second in zip(self.first, self.second):
            x = x + second(F.leaky_relu(first(F.leaky_relu(x, .1)), .1))
        return x


class Generator(nn.Module):
    def __init__(self, n_mels=80, channels=128):
        super().__init__()
        self.pre = conv(n_mels, channels, 7, padding=3)
        self.ups, self.mrfs = nn.ModuleList(), nn.ModuleList()
        for rate, kernel in ((8, 16), (8, 16), (2, 4), (2, 4)):
            self.ups.append(weight_norm(nn.ConvTranspose1d(channels, channels // 2,
                                                          kernel, stride=rate, padding=(kernel - rate) // 2)))
            channels //= 2
            self.mrfs.append(nn.ModuleList([ResidualBlock(channels, k) for k in (3, 7, 11)]))
        self.post = conv(channels, 1, 7, padding=3)

    def forward(self, mel):
        x = self.pre(mel)
        for up, blocks in zip(self.ups, self.mrfs):
            x = up(F.leaky_relu(x, .1))
            x = sum(block(x) for block in blocks) / len(blocks)
        return torch.tanh(self.post(F.leaky_relu(x, .1)))


class PeriodDiscriminator(nn.Module):
    def __init__(self, period):
        super().__init__()
        self.period = period
        channels = [1, 16, 64, 128, 256]
        self.layers = nn.ModuleList([weight_norm(nn.Conv2d(a, b, (5, 1), (3, 1), (2, 0)))
                                     for a, b in zip(channels[:-1], channels[1:])])
        self.final = weight_norm(nn.Conv2d(channels[-1], 1, (3, 1), padding=(1, 0)))

    def forward(self, wave):
        extra = (-wave.shape[-1]) % self.period
        x = F.pad(wave, (0, extra), mode="replicate")
        x = x.reshape(x.shape[0], 1, -1, self.period)
        maps = []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), .1)
            maps.append(x)
        x = self.final(x)
        return x.flatten(1), maps + [x]


class ScaleDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([conv(1, 16, 15, padding=7), conv(16, 64, 41, 4, 20, groups=4),
                                     conv(64, 128, 41, 4, 20, groups=4), conv(128, 256, 41, 4, 20, groups=4),
                                     conv(256, 256, 5, padding=2)])
        self.final = conv(256, 1, 3, padding=1)

    def forward(self, wave):
        x, maps = wave, []
        for layer in self.layers:
            x = F.leaky_relu(layer(x), .1)
            maps.append(x)
        x = self.final(x)
        return x.flatten(1), maps + [x]


class Discriminators(nn.Module):
    def __init__(self):
        super().__init__()
        self.periods = nn.ModuleList([PeriodDiscriminator(p) for p in (2, 3, 5, 7, 11)])
        self.scales = nn.ModuleList([ScaleDiscriminator() for _ in range(3)])

    def forward(self, wave):
        outputs = [d(wave) for d in self.periods]
        for i, d in enumerate(self.scales):
            if i:
                wave = F.avg_pool1d(wave, 4, 2, padding=1)
            outputs.append(d(wave))
        return outputs


def discriminator_loss(real, fake):
    return sum(((r - 1) ** 2).mean() + (f ** 2).mean() for (r, _), (f, _) in zip(real, fake))


def generator_loss(real, fake):
    adversarial = sum(((score - 1) ** 2).mean() for score, _ in fake)
    feature = sum(F.l1_loss(f, r.detach()) for (_, rs), (_, fs) in zip(real, fake) for r, f in zip(rs, fs))
    return adversarial + 2 * feature
