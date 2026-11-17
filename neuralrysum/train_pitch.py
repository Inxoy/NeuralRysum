"""Train the pitch estimator on synthetic material (no dataset needed).

    python -m neuralrysum.train_pitch --out runs/pitch [--steps 30000]

Resumes from <out>/last.pt. Prints frame-level precision / recall / pitch
error on fresh synthetic batches while it trains.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .pitch import PitchConfig, PitchEstimator, peaks
from .pitch_data import make_batch


def loss_fn(sal_logit, ons_logit, batch):
    w_s = 1.0 + 9.0 * batch.salience
    w_o = 1.0 + 19.0 * batch.onset
    ls = F.binary_cross_entropy_with_logits(sal_logit, batch.salience, weight=w_s)
    lo = F.binary_cross_entropy_with_logits(ons_logit, batch.onset, weight=w_o)
    return ls, lo


def frame_metrics(sal: np.ndarray, midi: np.ndarray, active: np.ndarray, bin_midi, thresh=0.5, tol=0.5):
    """sal (B, nb, F); midi / active (B, V, F) -> precision, recall, median |cents| of hits."""
    tp = fp = fn = 0
    errs = []
    b, _, fr = sal.shape
    for i in range(b):
        for f in range(0, fr, 4):
            est = [m for m, _ in peaks(sal[i, :, f], bin_midi, thresh)]
            ref = [m for m, a in zip(midi[i, :, f], active[i, :, f]) if a and bin_midi[0] <= m <= bin_midi[-1]]
            used = set()
            for r in ref:
                best = None
                for j, e in enumerate(est):
                    if j not in used and abs(e - r) < tol and (best is None or abs(e - r) < abs(est[best] - r)):
                        best = j
                if best is None:
                    fn += 1
                else:
                    used.add(best)
                    tp += 1
                    errs.append(abs(est[best] - r) * 100)
            fp += len(est) - len(used)
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return p, r, float(np.median(errs)) if errs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/pitch")
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--batch", type=int, default=12)
    ap.add_argument("--crop", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=16)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = PitchConfig(channels=a.channels)
    model = PitchEstimator(cfg).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, s / 500) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps)))
    step = 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=a.device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        step = ck["step"]
        print(f"resumed at step {step}")
    print(f"pitch estimator: {sum(p.numel() for p in model.parameters())} parameters, "
          f"{cfg.n_bins} bins, windows {cfg.windows}", flush=True)
    (out / "config.json").write_text(json.dumps({"pitch": cfg.to_dict(), "args": vars(a)}, indent=2))
    bin_midi = cfg.bin_midi()

    def save():
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "config": cfg.to_dict(), "kind": "pitch"}, last)

    t0 = time.time()
    while step < a.steps:
        batch = make_batch(cfg, a.batch, a.crop, a.device)
        sal, ons = model(batch.audio)
        ls, lo = loss_fn(sal, ons, batch)
        loss = ls + lo
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
        step += 1
        if step % 200 == 0:
            msg = f"step {step} | sal {ls.item():.4f} ons {lo.item():.4f}"
            if step % 1000 == 0:
                with torch.no_grad():
                    p, r, c = frame_metrics(torch.sigmoid(sal).cpu().numpy(), batch.midi.cpu().numpy(),
                                            batch.active.cpu().numpy(), bin_midi)
                msg += f" | P {p:.3f} R {r:.3f} err {c:.1f} cents"
            print(f"{msg} | {200 / (time.time() - t0):.2f} it/s", flush=True)
            t0 = time.time()
        if step % 2000 == 0 or step == a.steps:
            save()
    save()


if __name__ == "__main__":
    main()
