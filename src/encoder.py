from src.utils import image_to_vectors

import numpy as np

def encode_image(img, codebook, block_size=(4,4)):
    vectors = image_to_vectors(img, block_size)
    # 計算每個向量對應 codebook index
    dists = np.linalg.norm(
        vectors[:, None, :] - codebook[None, :, :], axis=2
    )
    indices = np.argmin(dists, axis=1)
    return indices  # 長度 = (H/bh)*(W/bw)
