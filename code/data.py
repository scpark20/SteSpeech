"""Explicit local manifests only. This module never downloads a dataset or model."""
import json
import random
import re
import unicodedata
from pathlib import Path
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from audio import LogMel, read_audio, pitch_energy
from separation import STEMS


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_asr(text, transcription="spelling"):
    index = 1 if transcription == "spelling" else 2
    text = re.sub(r"\(([^()]*)\)/\(([^()]*)\)", lambda m: m.group(index), text)
    text = re.sub(r"\b[blonu]/", " ", text)
    text = re.sub(r"[+*/?!.,~]", "", text)
    return " ".join(unicodedata.normalize("NFC", text).split())


def tts_symbols(text):
    """Input must already have numbers/abbreviations expanded by the author.

    KSS normalized-text field supplies this for training. Jamo is a grapheme
    frontend, not a phoneme G2P system; no unverified pronunciation substitution.
    """
    text = " ".join(unicodedata.normalize("NFKC", text).strip().split())
    if re.search(r"[0-9A-Za-z]", text):
        raise ValueError("Provide normalized Korean text: expand numbers and Latin abbreviations")
    return list(unicodedata.normalize("NFD", text))


class Vocabulary:
    def __init__(self, symbols):
        self.symbols = list(symbols)
        if self.symbols[:2] != ["<blank/pad>", "<unk>"]:
            raise ValueError("Vocabulary IDs 0 and 1 are reserved")
        self.index = {token: i for i, token in enumerate(self.symbols)}

    @classmethod
    def build(cls, sequences):
        return cls(["<blank/pad>", "<unk>"] + sorted(set(token for seq in sequences for token in seq)))

    def encode(self, symbols, strict=False):
        if strict and any(token not in self.index for token in symbols):
            raise ValueError("Text contains a symbol absent from training vocabulary")
        return torch.tensor([self.index.get(token, 1) for token in symbols], dtype=torch.long)

    def decode(self, ids):
        return unicodedata.normalize("NFC", "".join(self.symbols[i] for i in ids if i != 0))

    def save(self, path):
        Path(path).write_text(json.dumps(self.symbols, ensure_ascii=False), encoding="utf-8")


def prepare_asr(index, output, root, transcription="spelling", encoding="utf-8"):
    """Index JSONL: id,audio,transcript,split. Keep official split membership."""
    result = []
    for row in read_jsonl(index):
        if row["split"] not in ("train", "dev", "eval-clean", "eval-other"):
            raise ValueError("Use official KsponSpeech split labels")
        audio, text_path = [(Path(root) / row[key]).resolve() for key in ("audio", "transcript")]
        if not audio.is_file():
            raise FileNotFoundError(audio)
        text = normalize_asr(text_path.read_text(encoding=encoding), transcription)
        if not text:
            raise ValueError("Empty normalized ASR target")
        result.append({"id": row["id"], "audio": str(audio), "text": text, "split": row["split"]})
    write_jsonl(output, result)


def prepare_kss(metadata, alignments, split_file, output, cache):
    """Alignments JSONL: id,tokens,durations (frames at 22050 Hz / hop 256).

    Actual alignments must come from an aligner matching this jamo vocabulary.
    Never synthesize equal-duration labels as a substitute for alignment.
    split_file is JSON mapping utterance IDs to train/dev/test.
    """
    alignment = {row["id"]: row for row in read_jsonl(alignments)}
    split = json.loads(Path(split_file).read_text())
    mel = LogMel()
    root, cache = Path(metadata).parent, Path(cache).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    records = []
    for line in Path(metadata).read_text(encoding="utf-8").splitlines():
        fields = line.split("|")
        if len(fields) < 3:
            raise ValueError("Expected KSS path|text|normalized_text metadata")
        identity, normalized = fields[0], fields[2]
        symbols = tts_symbols(normalized)
        aligned = alignment[identity]
        if aligned["tokens"] != symbols or len(aligned["durations"]) != len(symbols):
            raise ValueError("Alignment tokenization mismatch")
        durations = torch.tensor(aligned["durations"])
        if durations.dtype != torch.int64 or (durations < 0).any():
            raise ValueError("Alignment durations must be nonnegative integers")
        wave = read_audio(root / identity, 22050, mono=True)
        features = mel(wave)[0].T
        if durations.sum() != features.shape[0]:
            raise ValueError("Durations must sum exactly to mel frame count")
        f0, energy = pitch_energy(wave, mel)
        # CWT pitch and quantized embeddings from the paper are replaced explicitly.
        pitch = torch.where(f0 > 0, torch.log(f0.clamp_min(1)) / 6, 0.)
        energy = torch.log1p(energy)
        filename = cache / (identity.replace("/", "_") + ".pt")
        torch.save({"symbols": symbols, "mel": features, "durations": durations,
                    "pitch": pitch, "energy": energy, "wave": wave}, filename)
        if split[identity] not in ("train", "dev", "test"):
            raise ValueError("Unknown KSS split")
        records.append({"id": identity, "features": str(filename), "symbols": symbols,
                        "split": split[identity]})
    write_jsonl(output, records)


def prepare_musdb(root, validation_tracks, output):
    validation = set(Path(validation_tracks).read_text().splitlines())
    records, found = [], set()
    for group in ("train", "test"):
        for folder in sorted((Path(root) / group).iterdir()):
            if not folder.is_dir():
                continue
            paths = {s: str((folder / (s + ".wav")).resolve()) for s in ("mixture", *STEMS)}
            if not all(Path(p).is_file() for p in paths.values()):
                raise ValueError("Every track requires mixture plus all four stems")
            label = "test" if group == "test" else ("dev" if folder.name in validation else "train")
            if label == "dev":
                found.add(folder.name)
            records.append({"id": group + "/" + folder.name, "split": label, **paths})
    if found != validation:
        raise ValueError("Validation list must refer to train tracks only")
    write_jsonl(output, records)


class SpeechDataset(Dataset):
    def __init__(self, manifest, task, split, vocab=None, segment=44100 * 6):
        self.rows = [r for r in read_jsonl(manifest) if r["split"] == split]
        if not self.rows:
            raise ValueError("Requested split is empty")
        ids = [r["id"] for r in self.rows]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate sample IDs")
        self.task, self.vocab, self.segment = task, vocab, segment
        self.training = split == "train"
        self.asr_mel = LogMel(16000, 512, 160)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        if self.task == "asr":
            wave = read_audio(row["audio"], 16000, mono=True)
            feat = self.asr_mel(wave)[0].T
            feat = (feat - feat.mean(0)) / feat.std(0, unbiased=False).clamp_min(1e-5)
            return {"id": row["id"], "features": feat, "tokens": self.vocab.encode(row["text"]),
                    "text": row["text"], "samples": wave.shape[-1]}
        if self.task in ("tts", "vocoder"):
            item = torch.load(row["features"], map_location="cpu", weights_only=True)
            item["id"] = row["id"]
            if self.task == "tts":
                item["tokens"] = self.vocab.encode(item["symbols"], strict=True)
            else:
                # Crop waveform first; recalculate mel to keep boundary conventions identical.
                wave = item["wave"]
                if self.segment == 0:
                    extra = (-wave.shape[-1]) % 256
                    item["wave"] = torch.nn.functional.pad(wave, (0, extra))
                else:
                    start = random.randrange(max(1, wave.shape[-1] - 8192 + 1)) if self.training else 0
                    item["wave"] = torch.nn.functional.pad(wave[:, start:start + 8192],
                                                            (0, max(0, 8192 - wave[:, start:start + 8192].shape[-1])))
            return item
        mixture = read_audio(row["mixture"], 44100)
        targets = torch.stack([read_audio(row[s], 44100) for s in STEMS])
        if mixture.shape[0] != 2 or targets.shape[1:] != mixture.shape:
            raise ValueError("Stereo stem lengths must match mixture")
        if self.segment:
            start = random.randrange(max(1, mixture.shape[-1] - self.segment + 1)) if self.training else 0
            mixture, targets = mixture[..., start:start + self.segment], targets[..., start:start + self.segment]
            extra = self.segment - mixture.shape[-1]
            mixture, targets = torch.nn.functional.pad(mixture, (0, extra)), torch.nn.functional.pad(targets, (0, extra))
        return {"id": row["id"], "mixture": mixture, "targets": targets}


def collate(items):
    result = {"ids": [item["id"] for item in items]}
    for key in ("features", "tokens", "durations", "mel", "pitch", "energy"):
        if key in items[0]:
            result[key] = pad_sequence([item[key] for item in items], batch_first=True)
    if "tokens" in result:
        result["lengths"] = torch.tensor([len(item["tokens"]) for item in items])
    if "features" in result:
        result["feature_lengths"] = torch.tensor([len(item["features"]) for item in items])
    if "mel" in result:
        result["mel_lengths"] = torch.tensor([len(item["mel"]) for item in items])
    for key in ("mixture", "targets", "wave"):
        if key in items[0] and (key != "wave" or "tokens" not in items[0]):
            result[key] = torch.stack([item[key] for item in items])
    if "text" in items[0]:
        result["texts"] = [item["text"] for item in items]
        result["samples"] = sum(item["samples"] for item in items)
    return result
