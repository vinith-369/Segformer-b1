"""
SegFormer-B1 Evaluation Script for Cityscapes.

Loads a trained model checkpoint and evaluates on the validation set.

Usage:
    python test.py --config config.yaml --checkpoint checkpoints/best_model.pth
    python test.py --config config.yaml --checkpoint checkpoints/best_checkpoint.pth
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from model import SegFormer
from data import CityscapesSegDataset
from data.cityscapes import CITYSCAPES_CLASSES
from utils import scores, setup_logger, get_logger


def evaluate(cfg, checkpoint_path):
    logger = get_logger()

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    logger.info(f"Device: {device}")

    # ---- Dataset ----
    val_dataset = CityscapesSegDataset(
        root_dir=cfg.dataset.root_dir,
        split=cfg.val.split,
        stage='val',
        aug=False,
        ignore_index=cfg.dataset.ignore_index,
        num_classes=cfg.dataset.num_classes,
    )

    num_workers = 0 if device.type in ('mps', 'cpu') else 4
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == 'cuda'))

    logger.info(f"Validation set: {len(val_dataset)} images")

    # ---- Model ----
    model = SegFormer(
        num_classes=cfg.dataset.num_classes,
        embedding_dim=cfg.exp.embedding_dim,
        pretrained_path=None,
    )

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')

    if 'model_state_dict' in ckpt:
        # Full checkpoint
        model.load_state_dict(ckpt['model_state_dict'])
        iteration = ckpt.get('iteration', '?')
        logger.info(f"Loaded checkpoint from iteration {iteration}: {checkpoint_path}")
    else:
        # Model weights only
        model.load_state_dict(ckpt)
        logger.info(f"Loaded model weights: {checkpoint_path}")

    model.to(device)
    model.eval()

    # ---- Evaluate ----
    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.dataset.ignore_index).to(device)
    val_loss = 0.0
    preds, gts = [], []

    logger.info("Running evaluation...")
    with torch.no_grad():
        for _, inputs, labels in tqdm(val_loader, total=len(val_loader),
                                       ncols=100, ascii=" >=", desc="Evaluating"):
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

    # ---- Compute Metrics ----
    score = scores(gts, preds, num_classes=cfg.dataset.num_classes)
    avg_loss = val_loss / max(len(val_loader), 1)

    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Val Loss:        {avg_loss:.4f}")
    logger.info(f"Pixel Accuracy:  {score['Pixel Accuracy']:.4f}")
    logger.info(f"Mean Accuracy:   {score['Mean Accuracy']:.4f}")
    logger.info(f"Mean IoU:        {score['Mean IoU']:.4f}")
    logger.info("-" * 70)
    logger.info("Per-class IoU:")
    for cls_id, cls_iou in score['Class IoU'].items():
        cls_name = CITYSCAPES_CLASSES[cls_id] if cls_id < len(CITYSCAPES_CLASSES) else f"class_{cls_id}"
        logger.info(f"  {cls_name:20s}: {cls_iou:.4f}")
    logger.info("=" * 70)

    return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SegFormer-B1 Cityscapes Evaluation")
    parser.add_argument("--config", default="config.yaml", type=str, help="Path to config YAML")
    parser.add_argument("--checkpoint", required=True, type=str,
                        help="Path to model checkpoint (.pth) — can be full checkpoint or model weights")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    logger = setup_logger(log_dir=cfg.logging.log_dir, log_name='evaluation.log')
    evaluate(cfg, args.checkpoint)
