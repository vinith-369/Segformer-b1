"""
SegFormer-B1 Training Script for Cityscapes.

Supports:
  - Mac (MPS/CPU): single-device training
  - NVIDIA single GPU: single-device CUDA training
  - NVIDIA multi-GPU: DDP training (when launched with torchrun)

Usage:
    # Single device (Mac/CPU/single-GPU):
    python train.py --config config.yaml

    # Multi-GPU:
    torchrun --nproc_per_node=2 train.py --config config.yaml
"""

import argparse
import datetime
import os
import random
import json

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from model import SegFormer
from data import CityscapesSegDataset
from utils import scores, PolyWarmupAdamW, setup_logger, get_logger


# ======================== Distributed Utilities ========================

def is_ddp():
    return 'WORLD_SIZE' in os.environ and int(os.environ['WORLD_SIZE']) > 1


def get_rank():
    return dist.get_rank() if is_ddp() else 0


def is_main():
    return get_rank() == 0


def get_device():
    """Auto-detect the best available device."""
    if is_ddp():
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(device)
        return device
    elif torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


# ======================== ETA Calculation ========================

def cal_eta(time0, cur_iter, total_iter):
    time_now = datetime.datetime.now().replace(microsecond=0)
    elapsed = time_now - time0
    scale = (total_iter - cur_iter) / float(cur_iter)
    eta = elapsed * scale
    return str(elapsed), str(eta)


# ======================== Validation ========================

def validate(model, criterion, data_loader, device, num_classes):
    """Run validation and compute metrics."""
    logger = get_logger()
    val_loss = 0.0
    preds, gts = [], []
    model.eval()

    with torch.no_grad():
        for _, inputs, labels in tqdm(data_loader, total=len(data_loader),
                                       ncols=100, ascii=" >=",
                                       disable=not is_main(),
                                       desc="Validating"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            labels = labels.long().to(outputs.device)

            resized_outputs = F.interpolate(
                outputs, size=labels.shape[1:],
                mode='bilinear', align_corners=False)

            loss = criterion(resized_outputs, labels)
            val_loss += loss.item()

            preds += list(torch.argmax(resized_outputs, dim=1).cpu().numpy().astype(np.int16))
            gts += list(labels.cpu().numpy().astype(np.int16))

    score = scores(gts, preds, num_classes=num_classes)
    avg_loss = val_loss / max(len(data_loader), 1)

    return avg_loss, score


# ======================== Checkpoint Saving ========================

def save_checkpoint(model, optimizer, iteration, val_score, checkpoint_dir, is_best=False, is_ddp_model=False):
    """Save model weights and full training state."""
    os.makedirs(checkpoint_dir, exist_ok=True)

    model_to_save = model.module if is_ddp_model else model

    # Save full checkpoint (for resuming training)
    ckpt = {
        'iteration': iteration,
        'model_state_dict': model_to_save.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_score': val_score,
    }
    ckpt_path = os.path.join(checkpoint_dir, f'checkpoint_iter_{iteration}.pth')
    torch.save(ckpt, ckpt_path)

    # Save model weights only (for inference)
    weights_path = os.path.join(checkpoint_dir, f'model_iter_{iteration}.pth')
    torch.save(model_to_save.state_dict(), weights_path)

    # Save latest symlink / copy
    latest_ckpt = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(ckpt, latest_ckpt)

    latest_model = os.path.join(checkpoint_dir, 'latest_model.pth')
    torch.save(model_to_save.state_dict(), latest_model)

    if is_best:
        best_ckpt = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(ckpt, best_ckpt)
        best_model = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model_to_save.state_dict(), best_model)

    return ckpt_path, weights_path


# ======================== Main Training ========================

def train(cfg):
    device = get_device()
    use_ddp = is_ddp()
    logger = get_logger()

    num_workers = 0 if device.type in ('mps', 'cpu') else 4

    if use_ddp:
        dist.init_process_group(backend='nccl')

    setup_seed(cfg.train.seed)

    time0 = datetime.datetime.now().replace(microsecond=0)

    # ---- Log config ----
    if is_main():
        logger.info("=" * 70)
        logger.info("SegFormer-B1 Cityscapes Training")
        logger.info("=" * 70)
        logger.info(f"Device: {device} | DDP: {use_ddp}")
        logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # ---- Dataset ----
    train_dataset = CityscapesSegDataset(
        root_dir=cfg.dataset.root_dir,
        split=cfg.train.split,
        stage='train',
        aug=True,
        resize_range=cfg.dataset.resize_range,
        rescale_range=cfg.dataset.rescale_range,
        crop_size=cfg.dataset.crop_size,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    val_dataset = CityscapesSegDataset(
        root_dir=cfg.dataset.root_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    # ---- DataLoaders ----
    if use_ddp:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.train.batch_size,
            num_workers=num_workers, pin_memory=True,
            drop_last=True, sampler=train_sampler, prefetch_factor=4)
    else:
        train_sampler = None
        train_loader = DataLoader(
            train_dataset, batch_size=cfg.train.batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=(device.type == 'cuda'), drop_last=True)

    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == 'cuda'), drop_last=False)

    if is_main():
        logger.info(f"Train: {len(train_dataset)} images | Val: {len(val_dataset)} images")

    # ---- Model ----
    pretrained_path = cfg.exp.pretrained_path if cfg.exp.pretrained else None

    model = SegFormer(
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.exp.embedding_dim,
        pretrained_path=pretrained_path,
    )

    param_groups = model.get_param_groups()
    model.to(device)

    if use_ddp:
        model = DistributedDataParallel(model, device_ids=[device], find_unused_parameters=True)

    # ---- Log model info ----
    if is_main():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        logger.info(f"Total params: {total_params:,} | Trainable: {trainable_params:,}")

    # ---- Optimizer ----
    optimizer = PolyWarmupAdamW(
        params=[
            {"params": param_groups[0], "lr": cfg.optimizer.learning_rate,
             "weight_decay": cfg.optimizer.weight_decay},
            {"params": param_groups[1], "lr": cfg.optimizer.learning_rate,
             "weight_decay": 0.0},
            {"params": param_groups[2], "lr": cfg.optimizer.learning_rate * 10,
             "weight_decay": cfg.optimizer.weight_decay},
        ],
        lr=cfg.optimizer.learning_rate,
        weight_decay=cfg.optimizer.weight_decay,
        betas=cfg.optimizer.betas,
        warmup_iter=cfg.scheduler.warmup_iter,
        max_iter=cfg.train.max_iters,
        warmup_ratio=cfg.scheduler.warmup_ratio,
        power=cfg.scheduler.power,
    )

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index).to(device)

    # ---- Training Loop ----
    best_miou = 0.0
    checkpoint_dir = cfg.logging.checkpoint_dir

    if train_sampler:
        train_sampler.set_epoch(0)
    train_loader_iter = iter(train_loader)

    if is_main():
        logger.info("Starting training...")
        logger.info("-" * 70)

    for n_iter in range(cfg.train.max_iters):
        model.train()

        try:
            _, inputs, labels = next(train_loader_iter)
        except StopIteration:
            if train_sampler:
                train_sampler.set_epoch(n_iter)
            train_loader_iter = iter(train_loader)
            _, inputs, labels = next(train_loader_iter)

        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(inputs)
        outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=False)
        seg_loss = criterion(outputs, labels.long())

        optimizer.zero_grad()
        seg_loss.backward()
        optimizer.step()

        # ---- Logging ----
        if (n_iter + 1) % cfg.train.log_iters == 0 and is_main():
            elapsed, eta = cal_eta(time0, n_iter + 1, cfg.train.max_iters)
            lr = optimizer.param_groups[0]['lr']
            logger.info(
                f"Iter: {n_iter + 1}/{cfg.train.max_iters} | "
                f"Loss: {seg_loss.item():.4f} | "
                f"LR: {lr:.3e} | "
                f"Elapsed: {elapsed} | ETA: {eta}"
            )

        # ---- Validation + Checkpoint ----
        if (n_iter + 1) % cfg.train.eval_iters == 0:
            if is_main():
                logger.info("=" * 40 + " VALIDATION " + "=" * 40)

            val_loss, val_score = validate(
                model=model, criterion=criterion,
                data_loader=val_loader, device=device,
                num_classes=cfg.dataset.num_classes)

            if is_main():
                miou = val_score['Mean IoU']
                pixel_acc = val_score['Pixel Accuracy']
                mean_acc = val_score['Mean Accuracy']

                logger.info(f"Val Loss: {val_loss:.4f}")
                logger.info(f"Pixel Accuracy: {pixel_acc:.4f}")
                logger.info(f"Mean Accuracy:  {mean_acc:.4f}")
                logger.info(f"Mean IoU:       {miou:.4f}")

                # Per-class IoU
                from data.cityscapes import CITYSCAPES_CLASSES
                logger.info("Per-class IoU:")
                for cls_id, cls_iou in val_score['Class IoU'].items():
                    cls_name = CITYSCAPES_CLASSES[cls_id] if cls_id < len(CITYSCAPES_CLASSES) else f"class_{cls_id}"
                    logger.info(f"  {cls_name:20s}: {cls_iou:.4f}")

                is_best = miou > best_miou
                if is_best:
                    best_miou = miou
                    logger.info(f"*** New best mIoU: {best_miou:.4f} ***")

                ckpt_path, weights_path = save_checkpoint(
                    model=model, optimizer=optimizer,
                    iteration=n_iter + 1, val_score=val_score,
                    checkpoint_dir=checkpoint_dir,
                    is_best=is_best,
                    is_ddp_model=use_ddp)

                logger.info(f"Saved checkpoint: {ckpt_path}")
                logger.info(f"Saved model weights: {weights_path}")
                logger.info("=" * 92)

        # ---- Periodic save (without validation) ----
        if (n_iter + 1) % cfg.train.save_iters == 0 and \
           (n_iter + 1) % cfg.train.eval_iters != 0 and is_main():
            save_checkpoint(
                model=model, optimizer=optimizer,
                iteration=n_iter + 1, val_score=None,
                checkpoint_dir=checkpoint_dir,
                is_best=False, is_ddp_model=use_ddp)
            logger.info(f"Periodic checkpoint saved at iter {n_iter + 1}")

    # ---- Final save ----
    if is_main():
        logger.info("=" * 70)
        logger.info("Training complete!")
        logger.info(f"Best mIoU: {best_miou:.4f}")

        save_checkpoint(
            model=model, optimizer=optimizer,
            iteration=cfg.train.max_iters, val_score=None,
            checkpoint_dir=checkpoint_dir,
            is_best=False, is_ddp_model=use_ddp)
        logger.info(f"Final model saved to {checkpoint_dir}/")
        logger.info("=" * 70)

    if use_ddp:
        dist.destroy_process_group()


# ======================== Entry Point ========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer-B1 Cityscapes Training")
    parser.add_argument("--config", default="config.yaml", type=str, help="Path to config YAML")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)

    if is_main():
        logger = setup_logger(
            log_dir=cfg.logging.log_dir,
            log_name=None,  # auto-timestamp
        )
    else:
        logger = get_logger()

    train(cfg)
