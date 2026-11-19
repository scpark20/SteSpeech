"""Metric definitions are explicit; model evaluation is never run on import."""
import math
import torch


def edit_counts(reference, hypothesis):
    """Levenshtein S,D,I with deterministic tie breaking; sequences may be words."""
    table = [[(0, 0, j) for j in range(len(hypothesis) + 1)]]
    for i, ref in enumerate(reference, 1):
        row = [(0, i, 0)]
        for j, hyp in enumerate(hypothesis, 1):
            if ref == hyp:
                row.append(table[i - 1][j - 1])
            else:
                s, d, ins = table[i - 1][j - 1]
                candidates = [(s + 1, d, ins)]
                s, d, ins = table[i - 1][j]
                candidates.append((s, d + 1, ins))
                s, d, ins = row[j - 1]
                candidates.append((s, d, ins + 1))
                row.append(min(candidates, key=sum))
        table.append(row)
    return table[-1][-1]


def text_metrics(references, hypotheses):
    result = {}
    for name, split in (("cer", lambda s: list(s.replace(" ", ""))), ("wer", str.split)):
        counts, total = [0, 0, 0], 0
        for reference, hypothesis in zip(references, hypotheses):
            ref, hyp = split(reference), split(hypothesis)
            counts = [a + b for a, b in zip(counts, edit_counts(ref, hyp))]
            total += len(ref)
        result[name] = {"rate": sum(counts) / total if total else None,
                        "substitutions": counts[0], "deletions": counts[1],
                        "insertions": counts[2], "reference_units": total}
    return result


def separation_scores(reference, estimate):
    """Per source, joint stereo-vector projection; silent references return None.

    These scores are NOT museval BSS Eval SDR. Channels are flattened together,
    so one scale is shared by left/right, preserving spatial-balance errors.
    """
    records = []
    for ref, est in zip(reference.double().flatten(1), estimate.double().flatten(1)):
        ref, est = ref - ref.mean(), est - est.mean()
        if ref.square().mean() < 1e-12:
            records.append({"si_sdr": None, "waveform_sdr": None, "silent_reference": True})
            continue
        target = torch.dot(est, ref) / ref.square().sum() * ref
        si = 10 * torch.log10((target.square().sum() + 1e-12) / ((est - target).square().sum() + 1e-12))
        sd = 10 * torch.log10(ref.square().sum() / ((ref - est).square().sum() + 1e-12))
        records.append({"si_sdr": float(si), "waveform_sdr": float(sd), "silent_reference": False})
    return records


def bss_eval(reference, estimate):
    try:
        import museval
    except ImportError as error:
        raise RuntimeError("BSS Eval requires the optional museval package") from error
    # museval [source,sample,channel], 1-second window/hop; protocol recorded in output.
    arrays = [x.detach().cpu().numpy().transpose(0, 2, 1) for x in (reference, estimate)]
    values = museval.evaluate(*arrays, win=44100, hop=44100)
    return {key: [[float(v) if math.isfinite(v) else None for v in source] for source in array]
            for key, array in zip(("SDR", "ISR", "SIR", "SAR"), values)}
