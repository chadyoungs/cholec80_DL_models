#!/usr/bin/env python
"""Train a sim-to-real texture translator (CycleGAN + surgery-specific guards).

    python scripts/train_sim2real.py \
        --a-dir data/sim/frames --b-dir data/frames \
        --a-mask-dir data/sim/instrument_masks \
        --temporal-weight 2.0 --epochs 100 --size 256 --batch-size 4

Domain A = simulated frames, domain B = real frames (e.g. Cholec80 frames
extracted by ``prepare_dataset.py``).  Only G_AB is needed at inference;
G_BA / D_A exist purely to make the cycle loss work.

THE FOUR LOSSES AND WHY EACH ONE IS HERE
========================================
Let G = G_AB (sim->real), F = G_BA (real->sim), D_B = real-domain critic.

1. ADVERSARIAL (LSGAN):   E[(D_B(G(a)) - 1)^2]
   The only term that actually pushes towards "looks real".  We use the
   least-squares form (Mao et al., ICCV 2017) instead of BCE: gradients do
   not saturate when the discriminator is confident, which makes GAN
   training far less brittle.  D_B is a PatchGAN: it classifies every
   ~70x70 receptive-field patch independently, so it supervises TEXTURE
   statistics rather than global layout -- exactly what we want to transfer.

2. CYCLE CONSISTENCY:   ||F(G(a)) - a||_1
   Without it the adversarial loss alone admits any mapping that lands in
   the real distribution -- the generator could map every input to the
   same realistic frame (mode collapse) or repaint anatomy freely.  The
   round trip forces the translation to be (approximately) invertible,
   which pins the content: shape, position, layout survive; only style
   changes.  L1 (not L2) because it tolerates a few large deviations
   (specular highlights) instead of blurring everything.

3. IDENTITY:   ||G(b) - b||_1
   Feed a REAL frame to the sim->real generator: it should do nothing.
   This regularises the color composition -- without it CycleGAN famously
   flips hues / invents tint shifts, because the cycle loss is invariant
   to any invertible color transform.  For surgery, keeping the tissue
   hue anchored to the real domain matters a lot.

4. TEMPORAL CONSISTENCY (this script's addition, cf. tempCycleGAN):
       || warp(G(a_t), flow_t->t+1) - G(a_{t+1}) ||_1
   restricted to pixels where (a) the flow passes the forward-backward
   check and (b) nothing is protected.  Two consecutive frames translated
   independently produce uncorrelated texture noise = flicker.  This loss
   says: after compensating for the true scene motion, the translation
   must be the SAME.  The flow is computed from the INPUT frames (which
   the generator cannot influence), so it is an honest supervision signal.

INSTRUMENT PROTECTION
=====================
Every generator output is composited with the original frame through the
protection mask BEFORE any loss is computed (see blend_protected).  The
generator receives zero gradient on instrument pixels, and the
discriminator judges exactly the composite we will emit at inference.
Semantically-labelled content (instruments, blood decals you placed
yourself) therefore survives bit-exact -- labels stay valid.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cholec80_dl.data import Sim2RealDataset  # noqa: E402
from cholec80_dl.engine import autocast, make_amp  # noqa: E402
from cholec80_dl.models import Discriminator, Generator, ReplayBuffer, linear_decay_lambda, weights_init_normal  # noqa: E402
from cholec80_dl.sim2real import batch_flow_and_validity, blend_protected, warp_images  # noqa: E402
from cholec80_dl.utils import CSVLogger, AverageMeter, describe_device, dump_json, make_run_dir, resolve_device, set_seed  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--a-dir", required=True, help="simulated frames (recursive, per-video subfolders)")
    p.add_argument("--b-dir", required=True, help="real frames, e.g. data/frames from prepare_dataset")
    p.add_argument("--a-mask-dir", default=None,
                   help="protection masks mirroring a-dir layout, .png, white=keep original")
    p.add_argument("--temporal-weight", type=float, default=0.0,
                   help="weight of the flow-warped temporal loss (0 = off, try 1-5)")
    p.add_argument("--flow-threshold", type=float, default=1.5,
                   help="forward-backward round-trip tolerance in pixels")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--decay-epoch", type=int, default=50, help="LR starts linear decay here")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--n-blocks", type=int, default=9, help="residual blocks in the generator")
    p.add_argument("--lambda-cycle", type=float, default=10.0)
    p.add_argument("--lambda-identity", type=float, default=10.0,
                   help="higher than vanilla CycleGAN (5): anchor colors/structure harder")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--sample-every", type=int, default=5)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default="runs")
    p.add_argument("--name", default="sim2real")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-steps", type=int, default=0, help="limit batches per epoch (debug)")
    return p.parse_args(argv)


def masked_l1(x: torch.Tensor, y: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """L1 averaged ONLY over pixels with weight > 0.

    ``weight`` is (B,1,H,W) and broadcasts over channels.  Dividing by the
    weight sum (not the pixel count) keeps the loss scale independent of
    how much of the frame happens to be valid -- otherwise clips with big
    occlusions would silently contribute weaker gradients.
    """
    return (weight * (x - y).abs()).sum() / (weight.sum() * x.shape[1] + 1e-6)


def main(argv=None) -> int:
    args = parse_args(argv)
    set_seed(args.seed)
    device = resolve_device(args.device)
    amp = not args.no_amp and device.type == "cuda"
    print(f"device: {describe_device(device)} | amp: {amp}")

    temporal = args.temporal_weight > 0
    dataset = Sim2RealDataset(args.a_dir, args.b_dir, size=args.size, train=True,
                              mask_dir=args.a_mask_dir, temporal=temporal)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.workers, pin_memory=(device.type == "cuda"),
                        drop_last=True)
    n_pairs = sum(dataset.has_next)
    print(f"dataset: {len(dataset.files_a)} sim / {len(dataset.files_b)} real frames | "
          f"masks: {args.a_mask_dir or 'none'} | temporal pairs: {n_pairs if temporal else 'off'}")

    # ------------------------------------------------------------------ #
    # Networks.  G: ResNet generator (2 down, n_blocks residual, 2 up).
    # D: 70x70 PatchGAN.  Normal init N(0, 0.02) is the DCGAN recipe.
    # ------------------------------------------------------------------ #
    g_ab = Generator(3, 3, n_blocks=args.n_blocks).to(device)
    g_ba = Generator(3, 3, n_blocks=args.n_blocks).to(device)
    d_a = Discriminator(3).to(device)
    d_b = Discriminator(3).to(device)
    for net in (g_ab, g_ba, d_a, d_b):
        net.apply(weights_init_normal)

    # Adam with beta1=0.5: the GAN literature standard.  Lower momentum
    # dampens the oscillation of the min-max game (high momentum makes the
    # two players overshoot each other).
    opt_g = torch.optim.Adam(list(g_ab.parameters()) + list(g_ba.parameters()),
                             lr=args.lr, betas=(0.5, 0.999))
    opt_da = torch.optim.Adam(d_a.parameters(), lr=args.lr, betas=(0.5, 0.999))
    opt_db = torch.optim.Adam(d_b.parameters(), lr=args.lr, betas=(0.5, 0.999))

    # Constant LR for the first half, linear to zero afterwards: the decay
    # phase stabilises texture details once the coarse mapping is learned.
    lambda_fn = linear_decay_lambda(args.epochs, args.decay_epoch)
    sched_g = torch.optim.lr_scheduler.LambdaLR(opt_g, lambda_fn)
    sched_da = torch.optim.lr_scheduler.LambdaLR(opt_da, lambda_fn)
    sched_db = torch.optim.lr_scheduler.LambdaLR(opt_db, lambda_fn)

    _, scaler_g = make_amp(device, amp)
    _, scaler_da = make_amp(device, amp)
    _, scaler_db = make_amp(device, amp)

    criterion_gan = nn.MSELoss()   # LSGAN: see module docstring, loss #1
    criterion_l1 = nn.L1Loss()
    # Replay buffers: D is trained on a mix of fresh and 50-step-old fakes.
    # Prevents D from over-fitting to the generator's latest quirks, which
    # otherwise causes the classic G/D oscillation (Shrivastava et al. 2017).
    buffer_fake_a, buffer_fake_b = ReplayBuffer(50), ReplayBuffer(50)

    run_dir = make_run_dir(args.out, args.name, seed=args.seed)
    samples_dir = run_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    dump_json({"args": vars(args), "model_config": {"n_blocks": args.n_blocks, "size": args.size}},
              run_dir / "args.json")
    logger = CSVLogger(run_dir / "metrics.csv",
                       ["epoch", "loss_G", "loss_G_gan", "loss_G_cycle",
                        "loss_G_identity", "loss_G_temporal", "loss_D", "seconds"])

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
        meters = {k: AverageMeter() for k in
                  ("G", "G_gan", "G_cycle", "G_identity", "G_temporal", "D")}
        steps = 0

        for batch in loader:
            if args.max_steps and steps >= args.max_steps:
                break
            real_a = batch["A"].to(device, non_blocking=True)
            real_b = batch["B"].to(device, non_blocking=True)
            mask_a = batch["mask_a"].to(device, non_blocking=True)

            # ============================================================== #
            # GENERATOR STEP
            # ============================================================== #
            opt_g.zero_grad(set_to_none=True)
            with autocast(device, amp):
                # --- identity: real frames must pass through unchanged ----
                same_b = g_ab(real_b)
                same_a = g_ba(real_a)
                loss_id = (criterion_l1(same_b, real_b)
                           + criterion_l1(same_a, real_a)) * args.lambda_identity

                # --- forward translation + instrument protection ----------
                # blend BEFORE every loss: no gradient on protected pixels,
                # and D_B sees exactly the inference-time composite.
                fake_b = blend_protected(g_ab(real_a), real_a, mask_a)
                pred_b = d_b(fake_b)
                loss_gan_ab = criterion_gan(pred_b, torch.ones_like(pred_b))

                # backward direction has no masks (real frames carry none)
                fake_a = g_ba(real_b)
                pred_a = d_a(fake_a)
                loss_gan_ba = criterion_gan(pred_a, torch.ones_like(pred_a))

                # --- cycle: content must survive the round trip -----------
                rec_a = blend_protected(g_ba(fake_b), fake_b, mask_a)
                rec_b = g_ab(fake_a)
                loss_cycle = (criterion_l1(rec_a, real_a)
                              + criterion_l1(rec_b, real_b)) * args.lambda_cycle

                loss_g = loss_id + loss_gan_ab + loss_gan_ba + loss_cycle
                loss_temporal = torch.zeros((), device=device)

            # --- temporal consistency (computed in fp32 outside autocast:
            #     flow/warp involve coordinate arithmetic that hates fp16) --
            if temporal:
                a_next = batch["A_next"].to(device, non_blocking=True)
                pair_valid = batch["pair_valid"].to(device)
                if bool(pair_valid.any()):
                    # Flow is a constant target derived from the INPUTS.
                    with torch.no_grad():
                        flow, occ = batch_flow_and_validity(real_a, a_next, args.flow_threshold)
                    # Translate t+1 with the same protected composite.
                    fake_next = blend_protected(g_ab(a_next), a_next, mask_a)
                    warped = warp_images(fake_b.float(), flow)
                    # Valid where: real temporal pair AND flow trustworthy
                    # AND pixel not protected.  (mask_a is the mask at time
                    # t; combined with the occlusion check this is a
                    # conservative approximation -- newly appearing
                    # instruments fail the flow check anyway.)
                    weight = occ * (1.0 - mask_a) * pair_valid.float().view(-1, 1, 1, 1)
                    loss_temporal = masked_l1(warped, fake_next.float(), weight) * args.temporal_weight
                    loss_g = loss_g + loss_temporal

            scaler_g.scale(loss_g).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            sched_g.step()

            # ============================================================== #
            # DISCRIMINATOR STEPS (real=1, fake=0; half-scale for stability)
            # ============================================================== #
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
            meters["G_temporal"].update(float(loss_temporal.detach()))
            meters["D"].update(float(loss_da.detach() + loss_db.detach()))
            steps += 1

        row = {
            "epoch": epoch + 1,
            "loss_G": meters["G"].avg,
            "loss_G_gan": meters["G_gan"].avg,
            "loss_G_cycle": meters["G_cycle"].avg,
            "loss_G_identity": meters["G_identity"].avg,
            "loss_G_temporal": meters["G_temporal"].avg,
            "loss_D": meters["D"].avg,
            "seconds": round(time.time() - t0, 1),
        }
        logger.log(row)
        print(f"[{epoch + 1:3d}/{args.epochs}] G {row['loss_G']:.3f} "
              f"(gan {row['loss_G_gan']:.3f}, cyc {row['loss_G_cycle']:.3f}, "
              f"temp {row['loss_G_temporal']:.3f}) D {row['loss_D']:.3f} | {row['seconds']}s")

        # Sample grid rows: sim input | protected translation | real | real->sim.
        if (epoch + 1) % args.sample_every == 0 or epoch + 1 == args.epochs:
            with torch.no_grad(), autocast(device, amp):
                grid = torch.cat([real_a, fake_b, real_b, g_ba(real_b)], dim=0)
            save_image(grid.float(), samples_dir / f"epoch_{epoch + 1:03d}.jpg",
                       nrow=real_a.size(0), normalize=True, value_range=(-1, 1))

        torch.save(
            {
                "g_ab": g_ab.state_dict(), "g_ba": g_ba.state_dict(),
                "d_a": d_a.state_dict(), "d_b": d_b.state_dict(),
                "opt_g": opt_g.state_dict(), "opt_da": opt_da.state_dict(),
                "opt_db": opt_db.state_dict(),
                "epoch": epoch, "args": vars(args),
                "model_config": {"n_blocks": args.n_blocks, "size": args.size},
            },
            ckpt_dir / "last.pt",
        )

    print(f"run dir: {run_dir}")
    print(f"translate frames/video with: python scripts/translate_video.py "
          f"--checkpoint {ckpt_dir / 'last.pt'} --input <sim frames or video> --out-dir output/translated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
