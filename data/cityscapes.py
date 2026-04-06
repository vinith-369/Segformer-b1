# ---------------------------------------------------------------
# Cityscapes dataset for SegFormer-B1 semantic segmentation.
# ---------------------------------------------------------------

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import transforms as T


# ---- Cityscapes label ID -> trainId mapping (19 classes + 255 ignore) ----
# Based on cityscapesScripts/helpers/labels.py
CITYSCAPES_LABEL_MAP = {
    -1: 255,  0: 255,  1: 255,  2: 255,  3: 255,  4: 255,
    5: 255,   6: 255,  7: 0,    8: 1,    9: 255,  10: 255,
    11: 2,   12: 3,   13: 4,   14: 255,  15: 255,  16: 255,
    17: 5,   18: 255,  19: 6,   20: 7,   21: 8,   22: 9,
    23: 10,  24: 11,  25: 12,  26: 13,  27: 14,  28: 15,
    29: 255,  30: 255,  31: 16,  32: 17,  33: 18,
}

# Numpy LUT for O(1) label mapping
_LABEL_LUT = np.full(256, 255, dtype=np.uint8)
for _k, _v in CITYSCAPES_LABEL_MAP.items():
    if 0 <= _k < 256:
        _LABEL_LUT[_k] = _v

# 19 Cityscapes class names (in trainId order)
CITYSCAPES_CLASSES = [
    'road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
    'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
    'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle'
]


def map_labels_to_trainid(label):
    """Map Cityscapes raw label IDs to train IDs using LUT."""
    return _LABEL_LUT[label]


def get_file_list(root_dir, split='train'):
    """Scan leftImg8bit/{split}/ to find all image/label pairs."""
    img_dir = os.path.join(root_dir, 'leftImg8bit', split)
    gt_dir = os.path.join(root_dir, 'gtFine', split)

    items = []
    for city in sorted(os.listdir(img_dir)):
        city_img_dir = os.path.join(img_dir, city)
        if not os.path.isdir(city_img_dir):
            continue
        for fname in sorted(os.listdir(city_img_dir)):
            if fname.endswith('_leftImg8bit.png'):
                base = fname.replace('_leftImg8bit.png', '')
                img_path = os.path.join(city_img_dir, fname)
                # Prefer _labelTrainIds.png (already mapped); fall back to _labelIds.png
                label_trainid_path = os.path.join(gt_dir, city, base + '_gtFine_labelTrainIds.png')
                label_id_path = os.path.join(gt_dir, city, base + '_gtFine_labelIds.png')
                label_path = label_trainid_path if os.path.exists(label_trainid_path) else label_id_path
                use_trainids = os.path.exists(label_trainid_path)
                items.append((base, img_path, label_path, use_trainids))

    return items


class CityscapesSegDataset(Dataset):
    """Cityscapes semantic segmentation dataset with augmentations.

    Args:
        root_dir: Path to Cityscapes root (containing leftImg8bit/ and gtFine/).
        split: 'train' or 'val'.
        stage: 'train' (with augmentation) or 'val' (no augmentation).
        crop_size: Random crop size for training.
        resize_range: [min, max] resize range.
        rescale_range: [min, max] random scale ratio.
        ignore_index: Label value to ignore in loss (255).
        num_classes: Number of segmentation classes (19).
        aug: Whether to apply augmentation.
    """

    def __init__(self, root_dir, split='train', stage='train',
                 resize_range=(512, 2048), rescale_range=(0.5, 2.0),
                 crop_size=1024, ignore_index=255, num_classes=19, aug=False):
        super().__init__()
        self.root_dir = root_dir
        self.stage = stage
        self.split = split
        self.items = get_file_list(root_dir, split)

        self.aug = aug
        self.ignore_index = ignore_index
        self.resize_range = resize_range
        self.rescale_range = rescale_range
        self.crop_size = crop_size
        self.num_classes = num_classes
        self.color_jittor = T.PhotoMetricDistortion()

        print(f"[Cityscapes] {split}: {len(self.items)} images found")

    def __len__(self):
        return len(self.items)

    def _apply_transforms(self, image, label):
        """Apply training augmentations or validation preprocessing."""
        if self.aug:
            if self.rescale_range:
                image, label = T.random_scaling(
                    image, label,
                    scale_range=self.rescale_range,
                    size_range=self.resize_range)
            image, label = T.random_fliplr(image, label)
            image = self.color_jittor(image)
            if self.crop_size:
                image, label = T.random_crop(
                    image, label,
                    crop_size=self.crop_size,
                    mean_rgb=(123.675, 116.28, 103.53),
                    ignore_index=self.ignore_index)

        if self.stage != 'train':
            image = T.img_resize_short(image, min_size=min(self.resize_range))

        image = T.normalize_img(image)
        image = np.transpose(image, (2, 0, 1))  # HWC -> CHW
        return image, label

    def __getitem__(self, idx):
        name, img_path, label_path, use_trainids = self.items[idx]

        image = np.asarray(Image.open(img_path).convert('RGB'))
        label = np.asarray(Image.open(label_path))

        # Map raw labelIds to trainIds if needed
        if not use_trainids:
            label = map_labels_to_trainid(label)

        image, label = self._apply_transforms(image, label)

        return name, image, label
