"""SteSpeech: local data preparation, training, evaluation and inference."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    asr = sub.add_parser("prepare-asr", help="Prepare an existing KsponSpeech index; never download")
    asr.add_argument("--index", required=True)
    asr.add_argument("--root", required=True)
    asr.add_argument("--output", required=True)
    asr.add_argument("--transcription", choices=("spelling", "pronunciation"), default="spelling")
    asr.add_argument("--encoding", default="utf-8")
    tts = sub.add_parser("prepare-kss")
    for name in ("metadata", "alignments", "split-file", "output", "cache"):
        tts.add_argument("--" + name, required=True)
    separation = sub.add_parser("prepare-musdb")
    for name in ("root", "validation-tracks", "output"):
        separation.add_argument("--" + name, required=True)
    train = sub.add_parser("train", help="Run real training only when explicitly invoked")
    train.add_argument("--task", choices=("asr", "tts", "vocoder", "separation"), required=True)
    train.add_argument("--manifest", required=True)
    train.add_argument("--output", required=True)
    train.add_argument("--config")
    train.add_argument("--resume")
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--epochs", type=int, required=True)
    train.add_argument("--batch-size", type=int, default=2)
    train.add_argument("--lr", type=float, default=2e-4)
    train.add_argument("--warmup", type=int, default=1000)
    train.add_argument("--clip", type=float, default=1.)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--amp", action="store_true")
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--split", required=True)
    evaluation.add_argument("--device", default="cuda:0")
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--bss-eval", action="store_true")
    inference = sub.add_parser("infer")
    inference.add_argument("--checkpoint", required=True)
    inputs = inference.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--audio")
    inputs.add_argument("--text")
    inference.add_argument("--output", required=True)
    inference.add_argument("--vocoder")
    inference.add_argument("--device", default="cuda:0")
    inference.add_argument("--beam", type=int, default=1)
    inference.add_argument("--pace", type=float, default=1.)
    return root


def main():
    args = parser().parse_args()
    if args.command.startswith("prepare"):
        import data
        values = vars(args).copy()
        values.pop("command")
        functions = {"prepare-asr": data.prepare_asr, "prepare-kss": data.prepare_kss,
                     "prepare-musdb": data.prepare_musdb}
        functions[args.command](**values)
    else:
        import engine
        if args.command == "train" and min(args.epochs, args.batch_size, args.warmup, args.lr, args.clip) <= 0:
            raise ValueError("Training settings must be positive")
        {"train": engine.train, "evaluate": engine.evaluate, "infer": engine.infer}[args.command](args)


if __name__ == "__main__":
    main()
