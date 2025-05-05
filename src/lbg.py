import numpy as np

def lbg(vectors, Nc, epsilon=1e-5, max_iter=100):
    """
    vectors: shape=(M, D)
    Nc: codebook 大小
    """
    M, D = vectors.shape
    # Step1 隨機初始化 Nc 個 codeword
    indices = np.random.choice(M, Nc, replace=False)
    codebook = vectors[indices].astype(float)  # shape=(Nc, D)

    prev_distortion = np.inf
    for it in range(max_iter):
        # Step2 分群
        dists = np.linalg.norm(
            vectors[:, None, :] - codebook[None, :, :], axis=2
        )  # shape=(M, Nc)
        labels = np.argmin(dists, axis=1)       # shape=(M,)

        # Step3 計算 distortion
        distortion = np.mean(np.min(dists, axis=1))
        if abs(prev_distortion - distortion) < epsilon:
            break
        prev_distortion = distortion

        # Step4 更新 codebook
        for i in range(Nc):
            cluster = vectors[labels == i]
            if len(cluster) > 0:
                codebook[i] = cluster.mean(axis=0)

    return codebook, labels
