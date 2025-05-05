import numpy as np
import faiss  # 使用 Faiss 加速最近鄰搜尋
from sklearn.decomposition import PCA  # 用於向量降維

class LBG:
    def __init__(self, Nc, epsilon=1e-5, max_iter=100, pca_dim=None):
        """
        Linde–Buzo–Gray (LBG) 演算法實作，結合 Splitting 初始化、PCA 降維與 Faiss 加速。

        參數:
        - Nc: 目標 codebook 大小
        - epsilon: 失真改變的收斂門檻
        - max_iter: 每次分裂後最大迭代次數
        - pca_dim: 若不為 None，會先將向量降維到此維度
        """
        self.Nc = Nc
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.pca_dim = pca_dim
        self.codebook = None

    def fit(self, vectors):
        """
        建立 codebook:
        1. (選用) PCA 降維到 pca_dim
        2. Splitting 初始化：由 1 個中心開始，倍增到 Nc
        3. 每次倍增後做 LBG 迭代至收斂
        4. 回傳回到原始維度的 codebook
        """
        # 1. PCA 降維
        if self.pca_dim and self.pca_dim < vectors.shape[1]:
            self.pca = PCA(n_components=self.pca_dim)
            vecs = self.pca.fit_transform(vectors)  # 降維後向量
        else:
            vecs = vectors.copy()
            self.pca = None

        # 2. Splitting 初始化：先從所有向量均值做為單一中心
        centroid = np.mean(vecs, axis=0, keepdims=True)
        codebook = centroid

        # 3. 迭代倍增並收斂
        while codebook.shape[0] < self.Nc:
            # 分裂：每個中心左右微調 ±1% 作為新中心
            codebook = np.vstack([codebook * (1 + 1e-2), codebook * (1 - 1e-2)])
            # 對當前中心執行 LBG 迭代
            codebook = self._lbg_iterations(vecs, codebook)
            # 若超出 Nc，則截斷
            if codebook.shape[0] > self.Nc:
                codebook = codebook[:self.Nc]
                break

        # 儲存結果
        self.codebook = codebook
        # 4. 若有 PCA，回投影回原始維度
        return self._deproject(codebook)

    def _lbg_iterations(self, vecs, codebook):
        """
        LBG 核心：在固定中心數量下，反覆分群與更新中心，直到失真不再顯著變化。
        使用 Faiss 做最近鄰搜尋以加速。  
        輸入:
        - vecs: 降維後或原始向量矩陣 (M×D)
        - codebook: 當前中心 (Ncᵢ×D)
        回傳:
        - 收斂後中心陣列 (Ncᵢ×D)
        """
        prev_dist = np.inf
        n_centroids = codebook.shape[0]
        for it in range(self.max_iter):
            # 3.1 分群：用 Faiss 快速找到每個 vec 的最近中心
            index = faiss.IndexFlatL2(vecs.shape[1])
            index.add(codebook.astype('float32'))
            _, labels = index.search(vecs.astype('float32'), 1)
            labels = labels.flatten()

            # 3.2 計算失真 (平均 MSE)
            dist = np.mean((vecs - codebook[labels])**2)
            # 3.3 判斷收斂
            if abs(prev_dist - dist) < self.epsilon:
                break
            prev_dist = dist

            # 3.4 更新中心：對每個群計算新均值
            new_cb = []
            for i in range(n_centroids):
                points = vecs[labels == i]
                if len(points) > 0:
                    new_cb.append(points.mean(axis=0))
                else:
                    # 若無點屬於該中心，保留舊中心
                    new_cb.append(codebook[i])
            codebook = np.vstack(new_cb)

        return codebook

    def _deproject(self, codebook):
        """
        若有 PCA，將中心從降維空間投影回原始向量空間。
        否則原樣回傳。
        """
        if self.pca is not None:
            return self.pca.inverse_transform(codebook)
        return codebook

# 使用方式示例：
# from src.lbg_updated import LBG
# lbg = LBG(Nc=256, epsilon=1e-4, max_iter=50, pca_dim=8)
# codebook = lbg.fit(training_vectors)
