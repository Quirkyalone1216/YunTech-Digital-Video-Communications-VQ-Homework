import numpy as np

def decode_indices(indices, codebook, img_shape, block_size=(4,4)):
    H, W = img_shape
    bh, bw = block_size
    recon = np.zeros((H, W), dtype=np.uint8)
    idx = 0
    for i in range(0, H, bh):
        for j in range(0, W, bw):
            recon[i:i+bh, j:j+bw] = codebook[indices[idx]].reshape(bh, bw)
            idx += 1
    return recon
