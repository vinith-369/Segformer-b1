# ---------------------------------------------------------------
# Image augmentation and preprocessing utilities.
# Pure NumPy/PIL — no mmcv dependency.
# ---------------------------------------------------------------

import random
import numpy as np
from PIL import Image
import cv2


def normalize_img(img, mean=(123.675, 116.28, 103.53), std=(58.395, 57.12, 57.375)):
    """Normalize image with ImageNet mean/std."""
    img = np.asarray(img, dtype=np.float32)
    proc = np.empty_like(img)
    proc[..., 0] = (img[..., 0] - mean[0]) / std[0]
    proc[..., 1] = (img[..., 1] - mean[1]) / std[1]
    proc[..., 2] = (img[..., 2] - mean[2]) / std[2]
    return proc


def random_scaling(image, label, scale_range, size_range):
    """Random scale image and label."""
    min_ratio, max_ratio = scale_range
    ratio = random.uniform(min_ratio, max_ratio)

    h, w = label.shape
    new_scale = [int(ratio * w), int(ratio * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    new_image = np.asarray(new_image).astype(np.float32)

    new_label = Image.fromarray(label).resize(new_scale, resample=Image.NEAREST)
    new_label = np.asarray(new_label)

    return new_image, new_label


def random_fliplr(image, label):
    """Random horizontal flip."""
    if random.random() > 0.5:
        image = np.fliplr(image).copy()
        label = np.fliplr(label).copy()
    return image, label


def random_crop(image, label, crop_size, mean_rgb=(123.675, 116.28, 103.53), ignore_index=255):
    """Random crop with padding if necessary. Uses cat_max_ratio to ensure class diversity."""
    h, w = label.shape

    H = max(crop_size, h)
    W = max(crop_size, w)

    pad_image = np.zeros((H, W, 3), dtype=np.float32)
    pad_label = np.ones((H, W), dtype=np.float32) * ignore_index

    pad_image[:, :, 0] = mean_rgb[0]
    pad_image[:, :, 1] = mean_rgb[1]
    pad_image[:, :, 2] = mean_rgb[2]

    H_pad = int(np.random.randint(H - h + 1))
    W_pad = int(np.random.randint(W - w + 1))

    pad_image[H_pad:(H_pad + h), W_pad:(W_pad + w), :] = image
    pad_label[H_pad:(H_pad + h), W_pad:(W_pad + w)] = label

    def get_random_cropbox(cat_max_ratio=0.75):
        for _ in range(10):
            H_start = random.randrange(0, H - crop_size + 1, 1)
            H_end = H_start + crop_size
            W_start = random.randrange(0, W - crop_size + 1, 1)
            W_end = W_start + crop_size

            temp_label = pad_label[H_start:H_end, W_start:W_end]
            index, cnt = np.unique(temp_label, return_counts=True)
            cnt = cnt[index != ignore_index]
            if len(cnt > 1) and np.max(cnt) / np.sum(cnt) < cat_max_ratio:
                break
        return H_start, H_end, W_start, W_end

    H_start, H_end, W_start, W_end = get_random_cropbox()

    image = pad_image[H_start:H_end, W_start:W_end, :]
    label = pad_label[H_start:H_end, W_start:W_end]

    return image, label


def img_resize_short(image, min_size=512):
    """Resize image so its shorter side is at least min_size."""
    h, w, _ = image.shape
    if min(h, w) >= min_size:
        return image

    scale = float(min_size) / min(h, w)
    new_scale = [int(scale * w), int(scale * h)]

    new_image = Image.fromarray(image.astype(np.uint8)).resize(new_scale, resample=Image.BILINEAR)
    return np.asarray(new_image).astype(np.float32)


class PhotoMetricDistortion:
    """Photometric distortion augmentation (brightness, contrast, saturation, hue).

    Pure OpenCV implementation — no mmcv dependency.
    """

    def __init__(self, brightness_delta=32, contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5), hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def _convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if np.random.randint(2):
            return self._convert(
                img, beta=random.uniform(-self.brightness_delta, self.brightness_delta))
        return img

    def contrast(self, img):
        if np.random.randint(2):
            return self._convert(
                img, alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        if np.random.randint(2):
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 1] = self._convert(
                hsv[:, :, 1],
                alpha=random.uniform(self.saturation_lower, self.saturation_upper))
            img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def hue(self, img):
        if np.random.randint(2):
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV)
            hsv[:, :, 0] = (hsv[:, :, 0].astype(int) +
                            np.random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        return img

    def __call__(self, img):
        img = self.brightness(img)
        mode = np.random.randint(2)
        if mode == 1:
            img = self.contrast(img)
        img = self.saturation(img)
        img = self.hue(img)
        if mode == 0:
            img = self.contrast(img)
        return img
