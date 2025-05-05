import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from src.utils import load_image, image_to_vectors, mse, psnr
from src.lbg import LBG
from src.encoder import encode_image
from src.decoder import decode_indices

# -----------------------------
# 設定區
# -----------------------------
BLOCK_SIZE = (4, 4)
BLOCK_AREA = BLOCK_SIZE[0] * BLOCK_SIZE[1]
NC_LIST = [128, 256, 512, 1024]
PCA_DIM = 8               # PCA 降維後向量維度
TRAIN_PATHS = ['./Img/Train/1.png', './Img/Train/3.png']
TEST_PATHS  = ['./Img/Test/2.png', './Img/Test/4.png']

# 輸出資料夾
CODEBOOK_DIR = 'results/codebooks'
RECON_DIR    = 'results/reconstructions'
CSV_PATH     = 'results/metrics.csv'
PLOT1_PATH   = 'results/PSNR_vs_bitrate.png'
PLOT2_PATH   = 'results/Time_vs_Nc.png'

# -----------------------------
# 功能函式定義
# -----------------------------

def ensure_dirs():
    """建立所需的結果資料夾"""
    os.makedirs(CODEBOOK_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)


def load_training_vectors(paths, block_size):
    """讀取並切割所有訓練影像，回傳堆疊後的向量陣列"""
    vecs = []
    for p in paths:
        img = load_image(p)
        vecs.append(image_to_vectors(img, block_size))
    return np.vstack(vecs)


def process_single_nc(vectors, nc, test_paths, block_size, pca_dim):
    """
    使用 Splitting+PCA+Faiss 訓練 codebook，
    並對測試影像編碼、解碼，測量訓練時間
    回傳 metrics list。
    """
    # 建立 LBG 物件
    lbg = LBG(Nc=nc, epsilon=1e-4, max_iter=50, pca_dim=pca_dim)

    # 量測 codebook 訓練時間
    tic_train = time.time()
    codebook = lbg.fit(vectors)
    time_train = time.time() - tic_train

    # 存 codebook
    np.save(f'{CODEBOOK_DIR}/codebook_{nc}.npy', codebook)

    results = []
    for tp in test_paths:
        img   = load_image(tp)
        idxs  = encode_image(img, codebook, block_size)
        recon = decode_indices(idxs, codebook, img.shape, block_size)

        name = os.path.basename(tp).replace('.png', f'_Nc{nc}.png')
        Image.fromarray(recon).save(f'{RECON_DIR}/{name}')

        m = mse(img, recon)
        p = psnr(img, recon)
        results.append((name, nc, m, p, time_train))

    return results


def run_experiments():
    """執行所有 Nc 的實驗並回傳整合後的 DataFrame"""
    all_metrics = []
    training_vecs = load_training_vectors(TRAIN_PATHS, BLOCK_SIZE)

    for nc in NC_LIST:
        metrics = process_single_nc(training_vecs, nc, TEST_PATHS, BLOCK_SIZE, PCA_DIM)
        all_metrics.extend(metrics)

    df = pd.DataFrame(all_metrics,
        columns=['file','Nc','MSE','PSNR','Time_train'])
    df.to_csv(CSV_PATH, index=False)
    return df


def plot_results(df):
    """根據結果 DataFrame 繪製並儲存圖表"""
    df['bits_per_index'] = np.ceil(np.log2(df['Nc']))
    df['bitrate_bpp']    = df['bits_per_index'] / BLOCK_AREA

    summary = df.groupby('Nc').agg({
        'bitrate_bpp': 'mean',
        'PSNR': 'mean',
        'Time_train': 'mean'
    }).reset_index()

    # 純文字輸出
    print("\n===== PSNR vs. Bitrate =====")
    print(summary[['Nc','bitrate_bpp','PSNR']].to_string(
        index=False, header=["Nc","Bitrate(bpp)","PSNR(dB)"]))
    print("\n===== Training Time vs. Nc =====")
    print(summary[['Nc','Time_train']].to_string(
        index=False, header=["Nc","Train(s)"]))
    print()

    # PSNR vs. Bitrate 圖
    plt.figure(figsize=(8,4))
    plt.scatter(summary['bitrate_bpp'], summary['PSNR'])
    for _, row in summary.iterrows():
        plt.text(row['bitrate_bpp'], row['PSNR'], f"Nc={int(row['Nc'])}",
                 fontsize=9, va='bottom', ha='right')
    plt.xlabel('Bit Rate (bits/pixel)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Bitrate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT1_PATH)
    plt.show()

    # Training Time vs. Nc 圖
    plt.figure(figsize=(8,4))
    plt.plot(summary['Nc'], summary['Time_train'], marker='o')
    for _, row in summary.iterrows():
        plt.text(row['Nc'], row['Time_train'], f"Nc={int(row['Nc'])}", fontsize=9, va='bottom', ha='right')
    plt.xlabel('Codebook Size Nc')
    plt.ylabel('Training Time (s)')
    plt.title('LBG Training Time vs. Nc')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT2_PATH)
    plt.show()


# -----------------------------
# 主程式
# -----------------------------
if __name__ == '__main__':
    ensure_dirs()
    df = run_experiments()
    print("完成！結果儲存在 results/ 下面。")
    plot_results(df)
