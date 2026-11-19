"""Complete train/evaluate/infer functions. Called only by explicit CLI subcommands."""
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from audio import LogMel, read_audio, write_audio
from asr import ConformerCTC, ctc_loss, greedy, prefix_beam, specaugment
from data import Vocabulary, SpeechDataset, collate, read_jsonl, tts_symbols
from tts import FastSpeech2, acoustic_loss
from vocoder import Generator, Discriminators, discriminator_loss, generator_loss
from separation import Separator, separate_long, STEMS
from metrics import text_metrics, separation_scores, bss_eval


def build_model(task, config, vocabulary=None):
    size = len(vocabulary) if vocabulary is not None else None
    if task == "asr":
        return ConformerCTC(size, **config)
    if task == "tts":
        return FastSpeech2(size, **config)
    if task == "vocoder":
        return nn.ModuleDict({"generator": Generator(**config), "discriminators": Discriminators()})
    if task == "separation":
        return Separator(**config)
    raise ValueError("Unknown task")


def to_device(batch, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_checkpoint(path, model, task, config, symbols, optimizers=None, scheduler=None,
                    scaler=None, epoch=0, step=0, best=float("inf"), settings=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"format": 1, "task": task, "config": config, "symbols": symbols,
             "model": model.state_dict(), "epoch": epoch, "step": step, "best": best,
             "optimizers": {k: v.state_dict() for k, v in (optimizers or {}).items()},
             "scheduler": scheduler.state_dict() if scheduler else None,
             "scaler": scaler.state_dict() if scaler else None, "rng": rng_state(),
             "settings": settings or {}}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, device):
    # Full resume includes Python/NumPy RNG. Load only your own trusted checkpoint.
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint["format"] != 1:
        raise ValueError("Unsupported checkpoint format")
    model = build_model(checkpoint["task"], checkpoint["config"], checkpoint["symbols"]).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def task_loss(task, model, batch, augment=False):
    if task == "asr":
        features = specaugment(batch["features"], batch["feature_lengths"]) if augment else batch["features"]
        logits, lengths = model(features, batch["feature_lengths"])
        return ctc_loss(logits, lengths, batch["tokens"], batch["lengths"])
    if task == "tts":
        output = model(batch["tokens"], batch["lengths"], batch["durations"], batch["pitch"], batch["energy"])
        return acoustic_loss(output, batch)[0]
    return model.loss(batch["mixture"], batch["targets"])


def vocoder_pair(model, wave, mel):
    # center=True gives N/hop+1 frames. Generator consumes N/hop frames.
    condition = mel(wave[:, 0])[..., :-1]
    fake = model["generator"](condition)
    if fake.shape != wave.shape:
        raise ValueError("Vocoder waveform and reference length mismatch")
    return fake


@torch.no_grad()
def validation_loss(task, model, loader, device, mel):
    model.eval()
    total, count = 0., 0
    for batch in loader:
        batch = to_device(batch, device)
        if task == "vocoder":
            fake = vocoder_pair(model, batch["wave"], mel)
            loss = F.l1_loss(mel(fake[:, 0]), mel(batch["wave"][:, 0]))
        else:
            loss = task_loss(task, model, batch)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite validation loss")
        n = len(batch["ids"])
        total, count = total + float(loss) * n, count + n
    return total / count


def train(args):
    """Epoch-boundary resume. No training is run by verification or PDF building."""
    device = torch.device(args.device)
    seed_everything(args.seed)
    config = json.loads(Path(args.config).read_text()) if args.config else {}
    rows = [r for r in read_jsonl(args.manifest) if r["split"] == "train"]
    symbols = None
    if args.task in ("asr", "tts"):
        sequences = [r["text"] if args.task == "asr" else r["symbols"] for r in rows]
        symbols = Vocabulary.build(sequences).symbols
    checkpoint = None
    if args.resume:
        model, checkpoint = load_checkpoint(args.resume, device)
        if checkpoint["task"] != args.task or symbols != checkpoint["symbols"]:
            raise ValueError("Resume task/vocabulary mismatch")
        if args.config and config != checkpoint["config"]:
            raise ValueError("Resume architecture mismatch")
        config = checkpoint["config"]
    else:
        model = build_model(args.task, config, symbols).to(device)
    vocab = Vocabulary(symbols) if symbols else None
    datasets = [SpeechDataset(args.manifest, args.task, split, vocab) for split in ("train", "dev")]
    loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=(i == 0),
                          collate_fn=collate, num_workers=0) for i, dataset in enumerate(datasets)]
    if args.task == "vocoder":
        optimizers = {name: torch.optim.AdamW(model[name].parameters(), lr=args.lr, betas=(.8, .99))
                      for name in ("generator", "discriminators")}
    else:
        optimizers = {"model": torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)}
    base_optimizer = optimizers.get("model", optimizers.get("generator"))
    scheduler = torch.optim.lr_scheduler.LambdaLR(base_optimizer,
                lambda step: min((step + 1) / args.warmup, (args.warmup / (step + 1)) ** .5))
    amp_enabled = args.amp and device.type == "cuda" and args.task != "vocoder"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    mel = LogMel().to(device)
    settings = {"batch_size": args.batch_size, "seed": args.seed, "lr": args.lr,
                "warmup": args.warmup, "amp": args.amp, "manifest": str(Path(args.manifest).resolve())}
    start, step, best = 0, 0, float("inf")
    if checkpoint:
        if checkpoint["settings"] != settings:
            raise ValueError("Resume settings changed; use the saved batch/lr/warmup/manifest/seed")
        for key, optimizer in optimizers.items():
            optimizer.load_state_dict(checkpoint["optimizers"][key])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        restore_rng(checkpoint["rng"])
        start, step, best = checkpoint["epoch"], checkpoint["step"], checkpoint["best"]
    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    if not args.resume and (directory / "last.pt").exists():
        raise FileExistsError("Use a new output directory or explicitly resume")
    for epoch in range(start, args.epochs):
        model.train()
        total, count = 0., 0
        for batch in loaders[0]:
            batch = to_device(batch, device)
            updated = True
            if args.task == "vocoder":
                fake = vocoder_pair(model, batch["wave"], mel)
                discriminators = model["discriminators"]
                optimizers["discriminators"].zero_grad(set_to_none=True)
                d_loss = discriminator_loss(discriminators(batch["wave"]), discriminators(fake.detach()))
                if not torch.isfinite(d_loss):
                    raise FloatingPointError("Non-finite discriminator loss")
                d_loss.backward()
                nn.utils.clip_grad_norm_(discriminators.parameters(), args.clip, error_if_nonfinite=True)
                optimizers["discriminators"].step()
                discriminators.requires_grad_(False)
                try:
                    optimizers["generator"].zero_grad(set_to_none=True)
                    real_out, fake_out = discriminators(batch["wave"]), discriminators(fake)
                    loss = generator_loss(real_out, fake_out) + 45 * F.l1_loss(mel(fake[:, 0]), mel(batch["wave"][:, 0]))
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite generator loss")
                    loss.backward()
                    nn.utils.clip_grad_norm_(model["generator"].parameters(), args.clip, error_if_nonfinite=True)
                    optimizers["generator"].step()
                finally:
                    discriminators.requires_grad_(True)
            else:
                optimizer = optimizers["model"]
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device.type, enabled=amp_enabled):
                    loss = task_loss(args.task, model, batch, augment=True)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Non-finite training loss")
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.clip, error_if_nonfinite=True)
                previous_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                updated = scaler.get_scale() >= previous_scale
            if updated:
                scheduler.step()
                step += 1
            total, count = total + float(loss.detach()) * len(batch["ids"]), count + len(batch["ids"])
        val = validation_loss(args.task, model, loaders[1], device, mel)
        improved, best = val < best, min(best, val)
        record = {"epoch": epoch + 1, "step": step, "train_loss": total / count, "dev_loss": val}
        with (directory / "history.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        for filename in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            save_checkpoint(directory / filename, model, args.task, config, symbols,
                            optimizers, scheduler, scaler, epoch + 1, step, best, settings)


@torch.inference_mode()
def evaluate(args):
    device = torch.device(args.device)
    model, state = load_checkpoint(args.checkpoint, device)
    task = state["task"]
    model.eval()
    vocabulary = Vocabulary(state["symbols"]) if state["symbols"] else None
    dataset = SpeechDataset(args.manifest, task, args.split, vocabulary, segment=0)
    records, refs, hyps = [], [], []
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    mel = LogMel().to(device)
    for i, item in enumerate(dataset):
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        if task == "asr":
            batch = to_device(collate([item]), device)
            logits, lengths = model(batch["features"], batch["feature_lengths"])
            ids = greedy(logits, lengths)[0]
            hypothesis = vocabulary.decode(ids)
            refs.append(item["text"])
            hyps.append(hypothesis)
            record = {"id": item["id"], "reference": item["text"], "hypothesis": hypothesis}
            seconds = item["samples"] / 16000
        elif task == "separation":
            reference = item["targets"].to(device)
            prediction = separate_long(model, item["mixture"].to(device))
            record = {"id": item["id"], "sources": dict(zip(STEMS, separation_scores(reference, prediction)))}
            if args.bss_eval:
                record["bss_eval_v4"] = bss_eval(reference, prediction)
            seconds = item["mixture"].shape[-1] / 44100
        elif task == "tts":
            batch = to_device(collate([item]), device)
            free = model(batch["tokens"], batch["lengths"])
            conditioned = model(batch["tokens"], batch["lengths"], batch["durations"], batch["pitch"], batch["energy"])
            # Teacher-conditioned mel error is not free-running quality or MOS.
            error, parts = acoustic_loss(conditioned, batch)
            record = {"id": item["id"], "teacher_conditioned_losses": {k: float(v) for k, v in parts.items()},
                      "predicted_frames": int(free["lengths"][0]), "reference_frames": len(item["mel"])}
            torch.save(free["mel"][0].cpu(), output / f"{i:06d}-mel.pt")
            seconds = int(free["lengths"][0]) * 256 / 22050
        else:
            wave = item["wave"][None].to(device)
            fake = vocoder_pair(model, wave, mel)
            record = {"id": item["id"], "log_mel_l1": float(F.l1_loss(mel(fake[:, 0]), mel(wave[:, 0])))}
            write_audio(output / f"{i:06d}-vocoder.wav", fake[0], 22050)
            seconds = wave.shape[-1] / 22050
        if device.type == "cuda":
            torch.cuda.synchronize()
        record["rtf_including_output_io"] = (time.perf_counter() - start) / seconds
        records.append(record)
    report = {"task": task, "split": args.split, "checkpoint": str(args.checkpoint), "records": records}
    if task == "asr":
        report["aggregate"] = text_metrics(refs, hyps)
    if task == "separation":
        report["aggregation"] = "median across tracks, per stem; silent references excluded"
        report["aggregate"] = {stem: {metric: float(np.median(values)) if values else None
            for metric in ("si_sdr", "waveform_sdr")
            for values in [[r["sources"][stem][metric] for r in records if r["sources"][stem][metric] is not None]]}
            for stem in STEMS}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return report


@torch.inference_mode()
def infer(args):
    device = torch.device(args.device)
    model, state = load_checkpoint(args.checkpoint, device)
    model.eval()
    task = state["task"]
    if (task == "tts" and not args.text) or (task != "tts" and not args.audio):
        raise ValueError("TTS requires --text; other tasks require --audio")
    if task == "asr":
        wave = read_audio(args.audio, 16000, True).to(device)
        feat = LogMel(16000, 512, 160).to(device)(wave)[0].T
        feat = (feat - feat.mean(0)) / feat.std(0, unbiased=False).clamp_min(1e-5)
        logits, lengths = model(feat[None], torch.tensor([len(feat)], device=device))
        ids = prefix_beam(logits[0, :lengths[0]].log_softmax(-1), args.beam) if args.beam > 1 else greedy(logits, lengths)[0]
        text = Vocabulary(state["symbols"]).decode(ids)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text, encoding="utf-8")
    elif task == "separation":
        wave = read_audio(args.audio, 44100).to(device)
        sources = separate_long(model, wave)
        for stem, source in zip(STEMS, sources):
            write_audio(Path(args.output) / (stem + ".wav"), source, 44100)
    elif task == "tts":
        vocabulary = Vocabulary(state["symbols"])
        tokens = vocabulary.encode(tts_symbols(args.text), strict=True).to(device)[None]
        output = model(tokens, torch.tensor([tokens.shape[1]], device=device), pace=args.pace)
        if not args.vocoder:
            raise ValueError("TTS waveform inference requires a trained vocoder checkpoint")
        vocoder, voc_state = load_checkpoint(args.vocoder, device)
        if voc_state["task"] != "vocoder":
            raise ValueError("Expected vocoder checkpoint")
        wave = vocoder.eval()["generator"](output["mel"].transpose(1, 2))[0]
        write_audio(args.output, wave, 22050)
    else:
        wave = read_audio(args.audio, 22050, True).to(device)
        original_length = wave.shape[-1]
        wave = F.pad(wave, (0, (-original_length) % 256))
        condition = LogMel().to(device)(wave)[..., :-1]
        generated = model["generator"](condition)[0, :, :original_length]
        write_audio(args.output, generated, 22050)
