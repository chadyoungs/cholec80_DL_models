#!/usr/bin/env python
"""Train a CycleGAN between two image domains (e.g. Cholec80 -> private data).

Expects two flat image directories (any extension), unpaired::

    python scripts/train_cyclegan.py \
        --a-dir data/cyclegan/cholec/trainA --b-dir data/cyclegan/private/trainB \
        --epochs 100 --size 256 --batch-size 4

Outputs ``runs/cyclegan/`` with ``samples/``, ``metrics.csv`` and a single
``checkpoints/{last,best}.pt`` containing all four networks.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80.data import UnalignedImageDataset  # noqa: E402
from cholec80.engine import autocast, make_amp  # noqa: E402
from cholec80.models import Discriminator, Generator, ReplayBuffer, linear_decay_lambda, weights_init_normal  # noqa: E402
from cholec80.utils import CSVLogger, AverageMeter, describe_device, dump_json, make_run_dir, resolve_device, set_seed  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--a-dir", required=True, help="domain A images")
    p.add_argument("--b-dir", required=True, help="domain B images")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--decay-epoch", type=int, default=50, help="epoch where the LR starts decaying")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-blocks", type=int, default=9)
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="cyclegan")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=0, help="limit batches per epoch (debug)")
    p.add_argument("--unpaired", action="store_true", default=True)
    return p.parse_args(argv)


def build_transforms(size: int):
    train_tf = transforms.Compose([
        transforms.Resize(int(size * 1.12), Image.BICUBIC),
        transforms.RandomCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return train_tf


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | amp: {amp}")

    transform = build_transforms(args.size)
    dataset = UnalignedImageDataset(args.a_dir, args.b_dir, transform, unaligned=True)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"dataset: {len(dataset)} pairs ({len(dataset.files_a)} A / {len(dataset.files_b)} B)")

    g_ab = Generator(3, 3, n_blocks=args.n_blocks).to(device)
    g_ba = Generator(3, 3, n_blocks=args.n_blocks).to(device)
    d_a = Discriminator(3).to(device)
    d_b = Discriminator(3).to(device)
    for net in (g_ab, g_ba, d_a, d_b):
        net.apply(weights_init_normal)

    opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()), lr=args.lr, betas=(0.5, 0.999))
    opt_da = torch.optim.Adam(d_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_db = torch.optim.Adam(d_b.parameters(), lr=args.lr, betas=(0.5, 0.999))

    lambda_fn = linear_decay_lambda(args.epochs, args.decay_epoch)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda_fn)
    sched_da = torch.optim.lr_scheduler.LambdaLR(opt_da, lambda_fn)
    sched_db = torch.optim.lr_scheduler.LambdaLR(opt_db, lambda_fn)

    _, scaler_g = make_amp(device, amp)
    _, scaler_da = make_amp(device, amp)
    _, scaler_db = make_amp(device, amp)

    criterion_gan = nn.MSELoss()      # LSGAN objective (as in the reference repo)
    criterion_l1 = nn.L1Loss()
    buffer_fake_a, buffer_fake_b = ReplayBuffer(50), ReplayBuffer(50)

    run_dir = make_run_dir(args.out, args.name, seed=args.seed)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    dump_json({"args": vars(args)}, run_dir / "args.json")
    logger = CSVLogger(
        run_dir / "metrics.csv",
        ["epoch", "loss_G", "loss_G_gan", "loss_G_cycle", "loss_G_identity", "loss_D", "seconds"],
    )

    start_epoch = 0
    if args.resume:
        state = torch.load(args.resume, map_location=device, weights_only=False)
        g_ab.load_state_dict(state["g_ab"])
        g_ba.load_state_dict(state["g_ba"])
        d_a.load_state_dict(state["d_a"])
        d_b.load_state_dict(state["d_b"])
        opt_g.load_state_dict(state["opt_g"])
        opt_da.load_state_dict(state["opt_da"])
        opt_db.load_state_dict(state["opt_db"])
        start_epoch = int(state.get("epoch", 0)) + 1
        print(f"resumed from {args.resume} (epoch {start_epoch})")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        meters = {k: AverageMeter() for k in ("G", "G_gan", "G_cycle", "G_identity", "D")}
        steps = 0

        for batch in loader:
            if args.max_steps and steps >= args.max_steps:
                break
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)

            # ---- generators -------------------------------------------------
            opt_g.zero_grad(set_to_none=True)
            with autocast(device, amp):
                same_b = g_ab(real_b)
                same_a = g_ba(real_a)
                loss_id = (criterion_l1(same_b, real_b) + criterion_l1(same_a, real_a)) * args.lambda_identity

                fake_b = g_ab(real_a)
                pred_b = d_b(fake_b)
                loss_gan_ab = criterion_gan(pred_b, torch.ones_like(pred_b))
                fake_a = g_ba(real_b)
                pred_a = d_a(fake_a)
                loss_gan_ba = criterion_gan(pred_a, torch.ones_like(pred_a))

                rec_a = g_ba(fake_b)
                rec_b = g_ab(fake_a)
                loss_cycle = (criterion_l1(rec_a, real_a) + criterion_l1(rec_b, real_b)) * args.lambda_cycle
                loss_g = loss_id + loss_gan_ab + loss_gan_ba + loss_cycle

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            sched_g.step()

            # ---- discriminators ---------------------------------------------
            opt_da.zero_grad(set_to_none=True)
            fake_a_buffered = buffer_fake_a.push_and_pop(fake_a)
            with autocast(device, amp):
                pred_real_a = d_a(real_a)
                pred_fake_a = d_a(fake_a_buffered.detach())
                loss_da = 0.5 * (criterion_gan(pred_real_a, torch.ones_like(pred_real_a))
                                 + criterion_gan(pred_fake_a, torch.zeros_like(pred_fake_a)))
            scaler_da.scale(loss_da).backward()
            scaler_da.step(opt_da)
            scaler_da.update()
            sched_da.step()

            opt_db.zero_grad(set_to_none=True)
            fake_b_buffered = buffer_fake_b.push_and_pop(fake_b)
            with autocast(device, amp):
                pred_real_b = d_b(real_b)
                pred_fake_b = d_b(fake_b_buffered.detach())
                loss_db = 0.5 * (criterion_gan(pred_real_b, torch.ones_like(pred_real_b))
                                 + criterion_gan(pred_fake_b, torch.zeros_like(pred_fake_b)))
            scaler_db.scale(loss_db).backward()
            scaler_db.step(opt_db)
            scaler_db.update()
            sched_db.step()

            meters["G"].update(float(loss_g.detach()))
            meters["G_gan"].update(float(loss_gan_ab.detach() + loss_gan_ba.detach()))
            meters["G_cycle"].update(float(loss_cycle.detach()))
            meters["G_identity"].update(float(loss_id.detach()))
            meters["D"].update(float(loss_da.detach() + loss_db.detach()))
            steps += 1

        row = {
            "epoch": epoch + 1,
            "loss_G": meters["G"].avg,
            "loss_G_gan": meters["G_gan"].avg,
            "loss_G_cycle": meters["G_cycle"].avg,
            "loss_G_identity": meters["G_identity"].avg,
            "loss_D": meters["D"].avg,
            "seconds": round(time.time() - t0, 1),
        }
        logger.log(row)
        print(f"[{epoch + 1:3d}/{args.epochs}] G {row['loss_G']:.3f} "
              f"(gan {row['loss_G_gan']:.3f}, cyc {row['loss_G_cycle']:.3f}) "
              f"D {row['loss_D']:.3f} | {row['seconds']}s")

        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs:
            with torch.no_grad(), autocast(device, amp):
                grid = torch.cat([real_a, g_ab(real_a), real_b, g_ba(real_b)], dim=0)
            save_image(grid.float(), samples_dir / f"epoch_{epoch + 1:03d}.jpg",
                       nrow=real_a.size(0), normalize=True, value_range=(-1, 1))

        torch.save(
            {
                "g_ab": g_ab.state_dict(), "g_ba": g_ba.state_dict(),
                "d_a": d_a.state_dict(), "d_b": d_b.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_da": opt_da.state_dict(), "opt_db": opt_db.state_dict(),
                "epoch": epoch, "args": vars(args),
            },
            ckpt_dir / "last.pt",
        )

    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
