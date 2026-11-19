"""Synthetic structural verification only: no datasets, backward or optimizer steps."""
import argparse
import inspect
import hashlib
import json
import tempfile
import sys
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
import torch
from audio import LogMel, Spectrogram, read_audio, write_audio, pitch_energy
from asr import ConformerCTC, collapse, prefix_beam, ctc_loss, ctc_forward
from tts import FastSpeech2, regulate, acoustic_loss
from vocoder import Generator, Discriminators, discriminator_loss, generator_loss
from separation import Separator, separate_long
from data import normalize_asr, tts_symbols, Vocabulary, prepare_asr, prepare_kss, prepare_musdb, SpeechDataset, collate
from metrics import text_metrics, separation_scores
from engine import save_checkpoint, load_checkpoint, infer


@torch.inference_mode()
def verify(device):
    torch.manual_seed(7)
    torch.set_num_threads(2)
    report = {"device": str(device), "torch": torch.__version__, "training_executed": False,
              "dataset_downloaded": False, "backward_executed": False, "checks": {}}
    def record(name, **data):
        report["checks"][name] = data or {"passed": True}
        print(name + ": passed", flush=True)
    wave = torch.sin(torch.arange(4096, device=device) * (2 * torch.pi * 220 / 22050))[None]
    stft = Spectrogram().to(device)
    restored = stft.inverse(stft(wave), wave.shape[-1])
    assert torch.allclose(wave, restored, atol=1e-5)
    mel = LogMel().to(device)
    assert mel(wave).shape == (1, 80, 17)
    f0, energy = pitch_energy(wave, mel)
    assert len(f0) == 17 and (f0 > 0).any()
    record("audio_roundtrip", maximum_error=float((wave - restored).abs().max()))
    assert collapse([2, 2, 0, 2, 3, 3]) == [2, 2, 3]
    probs = torch.tensor([[.1, .2, .7], [.7, .2, .1], [.1, .1, .8]])
    assert prefix_beam(probs.log(), width=30) == [2, 2]
    exact = torch.nn.functional.ctc_loss(probs.log()[:, None], torch.tensor([2, 2]),
             torch.tensor([3]), torch.tensor([2]), reduction="sum")
    assert torch.allclose(ctc_forward(probs.log(), [2, 2]), exact)
    m = text_metrics(["가 나"], ["가 다"])
    assert m["cer"]["rate"] == .5 and m["wer"]["rate"] == .5
    record("ctc_decoder_and_edit_distance")
    asr = ConformerCTC(12).to(device).eval()
    features = torch.randn(2, 96, 80, device=device)
    lens = torch.tensor([96, 73], device=device)
    logits, out_lens = asr(features, lens)
    assert tuple(logits.shape) == (2, 24, 12) and out_lens.tolist() == [24, 19]
    targets = torch.tensor([[2, 3, 4], [3, 4, 0]], device=device)
    loss = ctc_loss(logits, out_lens, targets, torch.tensor([3, 2], device=device))
    assert torch.isfinite(loss)
    features[1, 73:] = 999
    perturbed, _ = asr(features, lens)
    assert torch.allclose(logits[1, :19], perturbed[1, :19], atol=1e-5)
    record("conformer_default", shape=list(logits.shape), parameters=sum(p.numel() for p in asr.parameters()))
    tts = FastSpeech2(12).to(device).eval()
    tokens = torch.tensor([[2, 3, 4], [3, 4, 0]], device=device)
    lengths = torch.tensor([3, 2], device=device)
    durations = torch.tensor([[2, 1, 3], [2, 2, 0]], device=device)
    out = tts(tokens, lengths, durations)
    assert out["mel"].shape == (2, 6, 80) and out["lengths"].tolist() == [6, 4]
    assert out["mel"][1, 4:].count_nonzero() == 0
    batch = {"tokens": tokens, "lengths": lengths, "durations": durations,
             "mel": torch.randn_like(out["mel"]), "mel_lengths": out["lengths"],
             "pitch": torch.zeros(2, 6, device=device), "energy": torch.zeros(2, 6, device=device)}
    total, _ = acoustic_loss(out, batch)
    assert torch.isfinite(total)
    predicted = tts(tokens, lengths)
    assert (predicted["lengths"] > 0).all()
    record("fastspeech_default", shape=list(out["mel"].shape), parameters=sum(p.numel() for p in tts.parameters()))
    gen = Generator().to(device).eval()
    discriminators = Discriminators().to(device).eval()
    fake = gen(torch.randn(1, 80, 16, device=device))
    assert fake.shape == (1, 1, 4096)
    real_out, fake_out = discriminators(wave[None]), discriminators(fake)
    assert len(real_out) == 8
    assert torch.isfinite(generator_loss(real_out, fake_out))
    assert torch.isfinite(discriminator_loss(real_out, fake_out))
    record("hifigan_default", shape=list(fake.shape), parameters=sum(p.numel() for p in gen.parameters()))
    separator = Separator().to(device).eval()
    stereo = torch.cat([wave, wave * .7])
    prediction = separator(stereo[None])[0]
    assert prediction.shape == (4, 2, 4096)
    assert torch.allclose(prediction.sum(0), stereo, atol=2e-5)
    longer = torch.cat([stereo, stereo[:, :1234]], -1)
    chunks = separate_long(separator, longer, chunk=2048, overlap=512)
    assert chunks.shape == (4, 2, 5330)
    assert torch.allclose(chunks.sum(0), longer, atol=2e-5)
    refs = stereo[None].repeat(4, 1, 1) / 4
    assert torch.isfinite(separator.loss(stereo[None], refs[None]))
    scores = separation_scores(refs, refs)
    assert all(s["si_sdr"] > 90 for s in scores)
    assert separation_scores(torch.zeros_like(refs), refs)[0]["si_sdr"] is None
    record("separation_default_and_chunking", shape=list(chunks.shape), parameters=sum(p.numel() for p in separator.parameters()))
    with tempfile.TemporaryDirectory(prefix="stespeech-verify-") as temporary:
        folder = Path(temporary)
        audio_file = folder / "test.wav"
        write_audio(audio_file, wave, 22050)
        assert torch.allclose(read_audio(audio_file, 22050), wave.cpu(), atol=1e-6)
        vocabulary_symbols = ["<blank/pad>", "<unk>", *list("가느더로미")]
        save_checkpoint(folder / "state.pt", asr, "asr", {}, vocabulary_symbols)
        reloaded, state = load_checkpoint(folder / "state.pt", device)
        got, _ = reloaded.eval()(features, lens)
        expected, _ = asr(features, lens)
        assert torch.equal(got, expected)
        record("fresh_model_checkpoint_roundtrip", optimizer_resume_tested=False)
        # Fabricated fixture files exercise adapters, never an actual corpus.
        (folder / "text.txt").write_text("(학교)/(학꾜) b/ 가요+", encoding="utf-8")
        (folder / "index.jsonl").write_text(json.dumps({"id": "synthetic", "audio": "test.wav",
            "transcript": "text.txt", "split": "train"}))
        prepare_asr(folder / "index.jsonl", folder / "asr.jsonl", folder)
        vocabulary = Vocabulary.build(["학교 가요"])
        dataset = SpeechDataset(folder / "asr.jsonl", "asr", "train", vocabulary)
        assert collate([dataset[0]])["features"].shape[-1] == 80
        symbols = tts_symbols("가")
        frames = mel(wave).shape[-1]
        (folder / "metadata.txt").write_text("test.wav|가|가\n", encoding="utf-8")
        (folder / "alignment.jsonl").write_text(json.dumps({"id": "test.wav", "tokens": symbols,
            "durations": [8, frames - 8]}, ensure_ascii=False))
        (folder / "split.json").write_text(json.dumps({"test.wav": "train"}))
        prepare_kss(folder / "metadata.txt", folder / "alignment.jsonl", folder / "split.json",
                    folder / "tts.jsonl", folder / "cache")
        dataset = SpeechDataset(folder / "tts.jsonl", "tts", "train", Vocabulary.build([symbols]))
        assert collate([dataset[0]])["durations"].sum() == frames
        voc_dataset = SpeechDataset(folder / "tts.jsonl", "vocoder", "train")
        assert voc_dataset[0]["wave"].shape == (1, 8192)
        full_voc = SpeechDataset(folder / "tts.jsonl", "vocoder", "train", segment=0)
        assert full_voc[0]["wave"].shape == (1, 4096)
        for group in ("train", "test"):
            for name in ("mixture", "vocals", "drums", "bass", "other"):
                value = stereo.cpu() if name == "mixture" else stereo.cpu() / 4
                write_audio(folder / "musdb" / group / "synthetic" / (name + ".wav"), value, 44100)
        (folder / "valid.txt").write_text("")
        prepare_musdb(folder / "musdb", folder / "valid.txt", folder / "musdb.jsonl")
        dataset = SpeechDataset(folder / "musdb.jsonl", "separation", "train", segment=2048)
        assert dataset[0]["targets"].shape == (4, 2, 2048)
        record("synthetic_manifest_adapters")
        vocoder = torch.nn.ModuleDict({"generator": gen, "discriminators": discriminators})
        save_checkpoint(folder / "vocoder.pt", vocoder, "vocoder", {}, None)
        save_checkpoint(folder / "tts.pt", tts, "tts", {}, vocabulary_symbols)
        args = SimpleNamespace(checkpoint=folder / "state.pt", audio=str(audio_file), text=None,
                               device=str(device), output=folder / "transcript.txt", beam=1, pace=1., vocoder=None)
        infer(args)
        assert (folder / "transcript.txt").is_file()
        args.checkpoint, args.output = folder / "vocoder.pt", folder / "resynthesis.wav"
        infer(args)
        assert read_audio(args.output, 22050).shape[-1] == 4096
        args.checkpoint, args.vocoder = folder / "tts.pt", folder / "vocoder.pt"
        args.audio, args.text, args.output = None, "가", folder / "tts.wav"
        infer(args)
        assert read_audio(args.output, 22050).shape[-1] > 0
        record("synthetic_inference_file_paths", quality_evaluated=False)
    assert normalize_asr("(학교)/(학꾜) b/ 가요+") == "학교 가요"
    if device.type == "cuda":
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 1024 ** 2
        report["gpu"] = torch.cuda.get_device_name(device)
    root = Path(__file__).resolve().parents[1]
    files = sorted((root / "code").glob("*.py")) + [root / "scripts/ste_speech.py", Path(__file__)]
    report["source_sha256"] = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    report["python"] = sys.executable
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    result = verify(torch.device(args.device))
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(result, ensure_ascii=False, indent=2))
