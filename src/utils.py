import numpy as np
from PIL import Image
import os

def load_image(path):
    return np.array(Image.open(path).convert('L'))  # 單通道灰階

def image_to_vectors(img, block_size=(4,4)):
    H, W = img.shape
    bh, bw = block_size
    vectors = []
    for i in range(0, H, bh):
        for j in range(0, W, bw):
            block = img[i:i+bh, j:j+bw]
            if block.shape == (bh, bw):
                vectors.append(block.flatten())
    return np.stack(vectors, axis=0)  # shape=(M, bh*bw)

def mse(img1, img2):
    return np.mean((img1.astype(float) - img2.astype(float))**2)

def psnr(img1, img2):
    _mse = mse(img1, img2)
    if _mse == 0: return float('inf')
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(_mse))
