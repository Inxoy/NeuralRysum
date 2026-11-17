"""Train the v2 organ synth from organ clips with MIDI notes.

    python -m neuralrysum.organ_data data/raw data/organ_v2      # once
    python -m neuralrysum.train_synth --data data/organ_v2 --out runs/synth

The synth hears only the voices (key, gate, timing) and the timbre latent of
the clip, never the waveform, so it has to produce the organ itself. Resumes
from <out>/last.pt.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .descriptors import descriptors
from .losses import MultiResDiscriminator, d_loss, feature_matching, g_adv_loss, multires_stft_loss
from .model import NeuralRysum
from .organ_data import OrganMidiDataset
from .synth import OrganSynth, SynthConfig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=150_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--preroll", type=int, default=65536)
    ap.add_argument("--crop", type=int, default=32768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup_adv", type=int, default=50_000)
    ap.add_argument("--kl_weight", type=float, default=1e-2)
    ap.add_argument("--kl_warmup", type=int, default=20_000)
    ap.add_argument("--desc_weight", type=float, default=1.0)
    ap.add_argument("--adv_weight", type=float, default=1.0)
    ap.add_argument("--fm_weight", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--save_every", type=int, default=5000)
    ap.add_argument("--log_every", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    random.seed(a.seed)
    torch.backends.cudnn.benchmark = True
    dev = a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = SynthConfig()
    assert (a.preroll + a.crop) % cfg.slow_hop == 0 and a.preroll % cfg.hop == 0
    model = OrganSynth(cfg).to(dev)
    disc = MultiResDiscriminator().to(dev)
    opt_g = torch.optim.AdamW(model.parameters(), a.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), a.lr, betas=(0.8, 0.99))
    step = 0
    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=dev)
        model.load_state_dict(ck["model"])
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"resumed from {last} at step {step}")

    ds = OrganMidiDataset(a.data, "train", a.preroll, a.crop, cfg.hop, cfg.voices)
    vs = OrganMidiDataset(a.data, "valid", a.preroll, a.crop, cfg.hop, cfg.voices, gain_db=(0.0, 0.0))
    print(f"{len(ds.clips)} train clips ({ds.hours:.2f} h), {len(vs.clips)} valid clips", flush=True)
    print(f"synth: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M parameters", flush=True)
    dl = DataLoader(ds, batch_size=a.batch, num_workers=a.workers, drop_last=True,
                    persistent_workers=a.workers > 0)
    random.seed(1234)
    valid = [vs[0] for _ in range(32)]  # fixed validation windows
    random.seed(a.seed + step)
    (out / "config.json").write_text(json.dumps({"synth": cfg.to_dict(), "train": vars(a)}, indent=2))
    log = (out / "log.csv").open("a")
    pre = a.preroll

    def save(path):
        torch.save({"model": model.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
                    "opt_d": opt_d.state_dict(), "step": step, "config": cfg.to_dict(), "kind": "synth"}, path)

    @torch.no_grad()
    def validate():
        model.eval()
        losses = []
        for i in range(0, len(valid), a.batch):
            x, key, gate, on_age, off_age = (torch.stack(t).to(dev) for t in zip(*valid[i: i + a.batch]))
            desc = descriptors(x, cfg.slow_hop, cfg.sample_rate)
            mean, _, _ = model.encode(x)
            c = model.hold(NeuralRysum.delay_one(torch.cat([mean, desc], 1)), key.shape[-1])
            y = model.render(key, gate, on_age, off_age, c)
            losses.append(multires_stft_loss(y[..., pre:], x[..., pre:]).item())
        model.train()
        return float(np.mean(losses))

    t0 = time.time()
    while step < a.steps:
        for x, key, gate, on_age, off_age in dl:
            x, key, gate, on_age, off_age = (t.to(dev, non_blocking=True) for t in (x, key, gate, on_age, off_age))
            with torch.no_grad():
                desc = descriptors(x, cfg.slow_hop, cfg.sample_rate)
            o = model(x, desc, key, gate, on_age, off_age)
            y, xr = o["y"][..., pre:], x[..., pre:]
            use_adv = step >= a.warmup_adv
            if use_adv:
                loss_d = d_loss(disc(xr), disc(y.detach()))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()
            l_spec = multires_stft_loss(y, xr)
            l_desc = torch.nn.functional.mse_loss(o["desc_pred"], desc)
            kl_w = a.kl_weight * min(1.0, step / max(1, a.kl_warmup))
            loss = l_spec + kl_w * o["kl"] + a.desc_weight * l_desc
            stats = {"spec": l_spec.item(), "kl": o["kl"].item(), "desc": l_desc.item()}
            if use_adv:
                fake, real = disc(y), disc(xr)
                l_adv, l_fm = g_adv_loss(fake), feature_matching(real, fake)
                loss = loss + a.adv_weight * l_adv + a.fm_weight * l_fm
                stats.update(adv=l_adv.item(), fm=l_fm.item(), d=loss_d.item())
            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt_g.step()
            step += 1
            if step % a.log_every == 0:
                rate = a.log_every / (time.time() - t0)
                t0 = time.time()
                line = " ".join(f"{k} {v:.4f}" for k, v in stats.items())
                print(f"step {step} | {line} | {rate:.2f} it/s", flush=True)
                log.write(json.dumps({"step": step, **stats}) + "\n")
                log.flush()
            if step % a.save_every == 0 or step >= a.steps:
                v = validate()
                print(f"step {step} | valid spec {v:.4f}", flush=True)
                log.write(json.dumps({"step": step, "valid_spec": v}) + "\n")
                save(last)
                if step % (a.save_every * 5) == 0:
                    save(out / f"step_{step}.pt")
            if step >= a.steps:
                break
    save(last)


if __name__ == "__main__":
    main()
