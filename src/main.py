import os
import time
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from utils import load_image, image_to_vectors, mse, psnr
from lbg import LBG
from encoder import encode_image
from decoder import decode_indices

# -----------------------------
# 設定區
# -----------------------------
BLOCK_SIZE = (4, 4)
BLOCK_AREA = BLOCK_SIZE[0] * BLOCK_SIZE[1]
NC_LIST = [128, 256, 512, 1024]
PCA_DIM = 8               # PCA 降維後向量維度
TRAIN_PATHS = ['./Img/Train/1.png', './Img/Train/3.png']
TEST_PATHS  = ['./Img/Test/2.png', './Img/Test/4.png']

# 輸出資料夾與檔案
CODEBOOK_DIR    = 'results/codebooks'
RECON_DIR       = 'results/reconstructions'
CSV_PATH        = 'results/metrics.csv'
PLOT1_PATH      = 'results/PSNR_vs_bitrate.png'
PLOT2_PATH      = 'results/Time_vs_Nc.png'
RD_CSV_PATH     = 'results/rd_curve.csv'
RD_PLOT_PATH    = 'results/rd_curve.png'

# -----------------------------
# 功能函式定義
# -----------------------------

def ensure_dirs():
    """建立所需的結果資料夾"""
    os.makedirs(CODEBOOK_DIR, exist_ok=True)
    os.makedirs(RECON_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)


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
    並對測試影像編碼、解碼，測量訓練時間、壓縮時間、解碼時間
    回傳 metrics list。
    """
    # 訓練時間
    tic_train = time.time()
    lbg = LBG(Nc=nc, epsilon=1e-4, max_iter=50, pca_dim=pca_dim)
    codebook = lbg.fit(vectors)
    time_train = time.time() - tic_train
    np.save(f'{CODEBOOK_DIR}/codebook_{nc}.npy', codebook)

    results = []
    for tp in test_paths:
        img = load_image(tp)
        orig = os.path.basename(tp)

        # 壓縮 (encode) 時間
        tic_enc = time.time()
        idxs = encode_image(img, codebook, block_size)
        time_encode = time.time() - tic_enc

        # 解碼 (decode) 時間
        tic_dec = time.time()
        recon = decode_indices(idxs, codebook, img.shape, block_size)
        time_decode = time.time() - tic_dec

        # 儲存重建影像
        name = orig.replace('.png', f'_Nc{nc}.png')
        Image.fromarray(recon).save(f'{RECON_DIR}/{name}')

        # 計算指標
        m = mse(img, recon)
        p = psnr(img, recon)
        results.append((orig, nc, m, p, time_train, time_encode, time_decode))

    return results


def run_experiments():
    """執行所有 Nc 的實驗並回傳整合後的 DataFrame"""
    all_metrics = []
    training_vecs = load_training_vectors(TRAIN_PATHS, BLOCK_SIZE)

    for nc in NC_LIST:
        metrics = process_single_nc(training_vecs, nc, TEST_PATHS, BLOCK_SIZE, PCA_DIM)
        all_metrics.extend(metrics)

    df = pd.DataFrame(all_metrics,
        columns=['file','Nc','MSE','PSNR','Time_train','Time_encode','Time_decode'])
    df.to_csv(CSV_PATH, index=False)
    return df


def plot_results(df):
    """繪製 PSNR vs. Bit Rate 與 Training/Encode/Decode 時間 vs. Nc 圖表"""
    df['bits_per_index'] = np.ceil(np.log2(df['Nc']))
    df['bitrate_bpp']    = df['bits_per_index'] / BLOCK_AREA

    # PSNR vs. Bit Rate
    summary = df.groupby('Nc').agg({'bitrate_bpp':'mean','PSNR':'mean'}).reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(summary['bitrate_bpp'], summary['PSNR'], marker='o')
    for _, row in summary.iterrows():
        plt.text(row['bitrate_bpp'], row['PSNR'], f"Nc={row['Nc']}\nPSNR={row['PSNR']:.2f}",
                 fontsize=8, va='bottom', ha='right')
    plt.xlabel('Bit Rate (bits/pixel)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Bit Rate')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT1_PATH)
    plt.show()

    # 時間 vs. Nc
    time_summary = df.groupby('Nc').agg({
        'Time_train':'mean','Time_encode':'mean','Time_decode':'mean'
    }).reset_index()
    plt.figure(figsize=(8,4))
    plt.plot(time_summary['Nc'], time_summary['Time_train'], marker='o', label='Training')
    for _, row in time_summary.iterrows():
        plt.text(row['Nc'], row['Time_train'], f"{row['Time_train']:.2f}s", fontsize=7, va='bottom', ha='right')
    plt.xlabel('Codebook Size Nc')
    plt.ylabel('Time (s)')
    plt.title('Processing Time vs. Codebook Size')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOT2_PATH)
    plt.show()


def export_rd_curve(df):
    """計算並匯出 RD Curve 並在圖上標示每點數值"""
    # 保留與 csv 相同
    df['bpp'] = np.log2(df['Nc']) / BLOCK_AREA
    df[['file','bpp','PSNR']].to_csv(RD_CSV_PATH, index=False)

    plt.figure(figsize=(8,4))
    for fname, grp in df.groupby('file'):
        grp_sorted = grp.sort_values('bpp')
        plt.plot(grp_sorted['bpp'], grp_sorted['PSNR'], marker='o', label=fname)
        for _, row in grp_sorted.iterrows():
            plt.text(row['bpp'], row['PSNR'], f"Nc={row['Nc']}\n{row['PSNR']:.2f}dB", fontsize=7,
                     va='bottom', ha='left')
    plt.xlabel('Bit Rate (bits/pixel)')
    plt.ylabel('PSNR (dB)')
    plt.title('PSNR vs. Bit Rate for Each Test Image')
    plt.legend(title='Test Image')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(RD_PLOT_PATH)
    plt.show()

# -----------------------------
# 主程式
# -----------------------------
if __name__ == '__main__':
    ensure_dirs()
    df = run_experiments()
    print("完成！結果儲存在 results/ 下面。")
    plot_results(df)
    export_rd_curve(df)
