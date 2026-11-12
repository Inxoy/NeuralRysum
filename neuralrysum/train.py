"""Train a NeuralRysum model on a folder of recordings.

    python -m neuralrysum.train --data path/to/organ_wavs --out runs/rysum

Resumes automatically if ``<out>/last.pt`` exists.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .augment import perturb_timbre
from .config import ModelConfig, TrainConfig, large_config
from .data import CropDataset
from .descriptors import descriptors
from .losses import (
    MultiResDiscriminator,
    d_loss,
    feature_matching,
    g_adv_loss,
    multires_stft_loss,
)
from .model import NeuralRysum


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for name, f in TrainConfig.__dataclass_fields__.items():
        p.add_argument(f"--{name}", type=type(f.default), default=f.default)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p


def train(tc: TrainConfig, device: str = "cpu", mc: ModelConfig | None = None) -> Path:
    torch.manual_seed(tc.seed)
    out = Path(tc.out)
    out.mkdir(parents=True, exist_ok=True)
    mc = mc or (large_config() if tc.size == "large" else ModelConfig())
    assert tc.crop % mc.slow_hop == 0, f"--crop must be a multiple of {mc.slow_hop}"

    model = NeuralRysum(mc).to(device)
    disc = MultiResDiscriminator().to(device)
    opt_g = torch.optim.AdamW(model.parameters(), tc.lr, betas=(0.8, 0.99))
    opt_d = torch.optim.AdamW(disc.parameters(), tc.lr, betas=(0.8, 0.99))
    step = 0

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"])
        disc.load_state_dict(ck["disc"])
        opt_g.load_state_dict(ck["opt_g"])
        opt_d.load_state_dict(ck["opt_d"])
        step = ck["step"]
        print(f"resumed from {last} at step {step}")

    ds = CropDataset(tc.data, tc.crop, mc.sample_rate)
    print(f"{len(ds.files)} files, {ds.hours:.2f} h of audio")
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model: {n_par / 1e6:.2f} M parameters, hop {mc.hop} samples "
          f"({1000 * mc.hop / mc.sample_rate:.2f} ms)")
    dl = DataLoader(ds, batch_size=tc.batch, num_workers=tc.workers, drop_last=True,
                    persistent_workers=tc.workers > 0)
    (out / "config.json").write_text(json.dumps({"model": mc.to_dict(), "train": asdict(tc)}, indent=2))
    log = (out / "log.csv").open("a")

    def save(path: Path) -> None:
        torch.save(
            {"model": model.state_dict(), "disc": disc.state_dict(), "opt_g": opt_g.state_dict(),
             "opt_d": opt_d.state_dict(), "step": step, "config": mc.to_dict()},
            path,
        )

    t0 = time.time()
    while step < tc.steps:
        for x in dl:
            x = x.to(device)
            with torch.no_grad():
                desc = descriptors(x, mc.slow_hop, mc.sample_rate)
                x_fast = None
                if torch.rand(()) < tc.augment_prob:
                    x_fast = perturb_timbre(x, mc.sample_rate)
            use_adv = step >= tc.warmup_adv

            o = model(x, desc, x_fast)
            y = o["y"]

            if use_adv:
                loss_d = d_loss(disc(x), disc(y.detach()))
                opt_d.zero_grad(set_to_none=True)
                loss_d.backward()
                opt_d.step()

            l_spec = multires_stft_loss(y, x)
            l_desc = torch.nn.functional.mse_loss(o["desc_pred"], desc)
            # the prior learns on frozen codes: it must not shape the encoder
            l_prior = model.prior.nll(o["e"].detach(), o["c"].detach())
            kl_w = tc.kl_weight * min(1.0, step / max(1, tc.kl_warmup))
            loss = l_spec + kl_w * o["kl"] + tc.desc_weight * l_desc + tc.prior_weight * l_prior
            stats = {"spec": l_spec.item(), "kl": o["kl"].item(), "desc": l_desc.item(),
                     "prior": l_prior.item()}
            if use_adv:
                fake, real = disc(y), disc(x)
                l_adv, l_fm = g_adv_loss(fake), feature_matching(real, fake)
                loss = loss + tc.adv_weight * l_adv + tc.fm_weight * l_fm
                stats.update(adv=l_adv.item(), fm=l_fm.item(), d=loss_d.item())

            opt_g.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt_g.step()
            step += 1

            if step % tc.log_every == 0:
                rate = tc.log_every / (time.time() - t0)
                t0 = time.time()
                line = " ".join(f"{k} {v:.4f}" for k, v in stats.items())
                print(f"step {step} | {line} | {rate:.2f} it/s", flush=True)
                log.write(json.dumps({"step": step, **stats}) + "\n")
                log.flush()
            if step % tc.save_every == 0 or step == tc.steps:
                save(last)
                if step % (tc.save_every * 10) == 0:
                    save(out / f"step_{step}.pt")
            if step >= tc.steps:
                break
    save(last)
    return last


def main() -> None:
    args = vars(build_argparser().parse_args())
    device = args.pop("device")
    train(TrainConfig(**args), device)


if __name__ == "__main__":
    main()
