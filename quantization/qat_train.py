"""Quantization Aware Training fine-tuning script for LLVC.

Fine-tunes a pretrained LLVC model with FakeQuantize nodes to learn
quantization-friendly weights. Uses adversarial training with
HiFi-GAN discriminator (same losses as train.py).

Single-GPU only, no AMP (conflicts with fake quantization).

Usage:
    python quantization/qat_train.py \
        --checkpoint llvc_models/models/checkpoints/llvc_hfg/LibriSpeech_Female_8312.pth \
        --config experiments/llvc_hfg/config.json \
        --output_dir quantization/qat_output \
        --num_steps 10000 --lr 5e-5 --batch_size 4
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from dataset import LLVCDataset as Dataset
from quantization.qat_model import (
    convert_to_quantized,
    load_qat_checkpoint,
    prepare_qat_model,
    save_qat_checkpoint,
)
import utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def qat_train(
    checkpoint_path: str,
    config_path: str,
    output_dir: str,
    num_steps: int = 10000,
    lr: float = 5e-5,
    batch_size: int = 4,
    grad_clip: float = 1.0,
    save_every: int = 1000,
    freeze_bn_at: float = 0.6,
    freeze_observer_at: float = 0.8,
    resume_from: str = None,
):
    """Run QAT fine-tuning.

    Args:
        checkpoint_path: Path to pretrained float32 .pth checkpoint
        config_path: Path to config.json
        output_dir: Directory for QAT checkpoints
        num_steps: Total training steps
        lr: Learning rate (lower than original training)
        batch_size: Batch size
        grad_clip: Gradient clipping threshold
        save_every: Save checkpoint every N steps
        freeze_bn_at: Fraction of training to freeze BN stats
        freeze_observer_at: Fraction of training to freeze observers
        resume_from: Path to QAT checkpoint to resume from
    """
    os.makedirs(output_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    with open(config_path) as f:
        config = json.load(f)

    # --- Prepare QAT model ---
    logger.info("Preparing QAT model...")
    qat_model = prepare_qat_model(checkpoint_path, config_path, device)
    qat_model.to(device)

    param_count = sum(p.numel() for p in qat_model.parameters()) / 1e6
    logger.info(f"QAT model: {param_count:.2f}M parameters on {device}")

    # --- Load discriminator (stays float32) ---
    if config["discriminator"] == "hfg":
        from hfg_disc import ComboDisc, discriminator_loss, generator_loss, feature_loss
        net_d = ComboDisc()
    else:
        from discriminators import (
            MultiPeriodDiscriminator,
            discriminator_loss,
            generator_loss,
            feature_loss,
        )
        net_d = MultiPeriodDiscriminator(periods=config["periods"])
    net_d.to(device)

    # Load discriminator weights if available
    disc_ckpt = checkpoint_path.replace("G_", "D_").replace(
        "LibriSpeech_", "D_LibriSpeech_"
    )
    # Try to find matching discriminator checkpoint
    ckpt_dir = os.path.dirname(checkpoint_path)
    disc_files = [
        f for f in os.listdir(ckpt_dir)
        if f.startswith("D_") and f.endswith(".pth")
    ] if os.path.isdir(ckpt_dir) else []
    if disc_files:
        disc_path = os.path.join(ckpt_dir, sorted(disc_files)[-1])
        try:
            disc_state = torch.load(disc_path, map_location=device, weights_only=False)
            net_d.load_state_dict(disc_state["model"])
            logger.info(f"Loaded discriminator from {disc_path}")
        except Exception as e:
            logger.warning(f"Could not load discriminator: {e}. Training from scratch.")
    else:
        logger.info("No discriminator checkpoint found. Training discriminator from scratch.")

    # --- Optimizers ---
    optim_g = torch.optim.AdamW(qat_model.parameters(), lr=lr, weight_decay=0.0)
    optim_d = torch.optim.AdamW(net_d.parameters(), lr=lr, weight_decay=0.0)

    # --- Resume from checkpoint ---
    global_step = 0
    start_epoch = 0
    if resume_from and os.path.exists(resume_from):
        qat_model, optim_g, lr, start_epoch, global_step = load_qat_checkpoint(
            resume_from, qat_model, optim_g
        )
        logger.info(f"Resumed from {resume_from} at step {global_step}")

    # --- Dataset ---
    data_train = Dataset(**config["data"], dset="train")
    logger.info(f"Training dataset: {len(data_train)} samples")

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=device == "cuda",
        persistent_workers=num_workers > 0,
        drop_last=True,
    )

    # --- Milestone steps ---
    freeze_bn_step = int(num_steps * freeze_bn_at)
    freeze_obs_step = int(num_steps * freeze_observer_at)
    logger.info(f"Training for {num_steps} steps")
    logger.info(f"  Freeze BN at step {freeze_bn_step} ({freeze_bn_at*100:.0f}%)")
    logger.info(f"  Freeze observers at step {freeze_obs_step} ({freeze_observer_at*100:.0f}%)")
    logger.info(f"  Save every {save_every} steps")
    logger.info(f"  LR: {lr}, batch_size: {batch_size}, grad_clip: {grad_clip}")

    # --- Training loop ---
    qat_model.train()
    net_d.train()
    bn_frozen = False
    obs_frozen = False
    t_start = time.time()

    while global_step < num_steps:
        for batch in train_loader:
            if global_step >= num_steps:
                break

            og, gt = batch
            og = og.to(device, non_blocking=True)
            gt = gt.to(device, non_blocking=True)

            # --- Freeze BN stats ---
            if not bn_frozen and global_step >= freeze_bn_step:
                logger.info(f"Step {global_step}: Freezing BatchNorm stats")
                qat_model.apply(torch.nn.intrinsic.qat.freeze_bn_stats)
                bn_frozen = True

            # --- Freeze observers ---
            if not obs_frozen and global_step >= freeze_obs_step:
                logger.info(f"Step {global_step}: Freezing observers")
                qat_model.apply(torch.ao.quantization.disable_observer)
                obs_frozen = True

            # --- Forward through QAT model ---
            output = qat_model(og)

            # --- Discriminator step ---
            # Random segment for discriminator
            if config["segment_size"] < output.shape[-1]:
                start_idx = random.randint(
                    0, output.shape[-1] - config["segment_size"] - 1
                )
                gt_sliced = gt[:, :, start_idx : start_idx + config["segment_size"]]
                output_sliced = output.detach()[
                    :, :, start_idx : start_idx + config["segment_size"]
                ]
            else:
                gt_sliced = gt
                output_sliced = output.detach()

            y_d_hat_r, y_d_hat_g, _, _ = net_d(output_sliced, gt_sliced)
            loss_disc, _, _ = discriminator_loss(y_d_hat_r, y_d_hat_g)

            optim_d.zero_grad(set_to_none=True)
            loss_disc.backward()
            torch.nn.utils.clip_grad_norm_(net_d.parameters(), grad_clip)
            optim_d.step()

            # --- Generator step ---
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(gt, output)
            loss_gen, _ = generator_loss(y_d_hat_g)
            loss_gen = loss_gen * config.get("disc_loss_c", 1)
            loss_fm = feature_loss(fmap_r, fmap_g) * config.get("feature_loss_c", 2)

            # Mel loss
            if config["aux_mel"]["c"] > 0:
                loss_mel = utils.aux_mel_loss(output, gt, config) * config["aux_mel"]["c"]
            else:
                loss_mel = torch.tensor(0.0, device=device)

            loss_total = loss_gen + loss_fm + loss_mel

            optim_g.zero_grad(set_to_none=True)
            loss_total.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), grad_clip)
            optim_g.step()

            global_step += 1

            # --- Logging ---
            if global_step % 100 == 0:
                elapsed = time.time() - t_start
                steps_per_sec = global_step / max(elapsed, 1e-6)
                logger.info(
                    f"Step {global_step}/{num_steps} | "
                    f"loss_total={loss_total.item():.4f} "
                    f"loss_gen={loss_gen.item():.4f} "
                    f"loss_fm={loss_fm.item():.4f} "
                    f"loss_mel={loss_mel.item():.4f} "
                    f"loss_disc={loss_disc.item():.4f} | "
                    f"{steps_per_sec:.2f} steps/s"
                )

            # --- Save checkpoint ---
            if global_step % save_every == 0:
                ckpt_path = os.path.join(output_dir, f"G_qat_{global_step}.pth")
                save_qat_checkpoint(qat_model, optim_g, 0, global_step, lr, ckpt_path)

    # --- Final save ---
    final_path = os.path.join(output_dir, "G_qat_final.pth")
    save_qat_checkpoint(qat_model, optim_g, 0, global_step, lr, final_path)

    # --- Convert to quantized ---
    logger.info("Converting QAT model to quantized int8...")
    quantized = convert_to_quantized(qat_model)
    quantized_path = os.path.join(output_dir, "G_quantized_int8.pth")
    torch.save({"model": quantized.state_dict()}, quantized_path)
    logger.info(f"Saved quantized model to {quantized_path}")

    # --- Report size comparison ---
    orig_size = os.path.getsize(checkpoint_path) / (1024 * 1024)
    qat_size = os.path.getsize(final_path) / (1024 * 1024)
    q_size = os.path.getsize(quantized_path) / (1024 * 1024)
    logger.info(f"Original checkpoint:  {orig_size:.1f} MB")
    logger.info(f"QAT checkpoint:       {qat_size:.1f} MB")
    logger.info(f"Quantized int8:       {q_size:.1f} MB")

    logger.info("QAT training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QAT fine-tuning for LLVC")
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to pretrained float32 .pth checkpoint",
    )
    parser.add_argument(
        "--config",
        default="experiments/llvc_hfg/config.json",
        help="Path to config.json",
    )
    parser.add_argument(
        "--output_dir",
        default="quantization/qat_output",
        help="Output directory for QAT checkpoints",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default=10000,
        help="Total training steps (default: 10000)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=5e-5,
        help="Learning rate (default: 5e-5, 10x lower than original)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size (default: 4)",
    )
    parser.add_argument(
        "--grad_clip",
        type=float,
        default=1.0,
        help="Gradient clipping threshold (default: 1.0)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save checkpoint every N steps (default: 1000)",
    )
    parser.add_argument(
        "--freeze_bn_at",
        type=float,
        default=0.6,
        help="Fraction of training to freeze BN stats (default: 0.6)",
    )
    parser.add_argument(
        "--freeze_observer_at",
        type=float,
        default=0.8,
        help="Fraction of training to freeze observers (default: 0.8)",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to QAT checkpoint to resume from",
    )
    args = parser.parse_args()

    qat_train(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        output_dir=args.output_dir,
        num_steps=args.num_steps,
        lr=args.lr,
        batch_size=args.batch_size,
        grad_clip=args.grad_clip,
        save_every=args.save_every,
        freeze_bn_at=args.freeze_bn_at,
        freeze_observer_at=args.freeze_observer_at,
        resume_from=args.resume,
    )
