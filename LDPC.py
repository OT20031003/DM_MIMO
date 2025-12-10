import numpy as np
from PIL import Image
from pyldpc import make_ldpc, encode, decode
import os, time

path = "./sentimg" # 送信画像dirのpath
pathr = "./outputs/LDPC" # 受信画像dirのpath

# --- 修正点 1: 保存先ディレクトリを自動で作成 ---
os.makedirs(pathr, exist_ok=True)
# -----------------------------------------

"""
    列の数: n
    行の数: m = n * d_v / d_c
    1列あたりの1の数: d_v
    """
# --- パラメータ設定 ---
BLOCK_SIZE = 1024
CODE_RATE = 0.5 # 符号率
d_v, d_c = 2, 4 # LDPCのパラメータ


n = int(BLOCK_SIZE / CODE_RATE) # 符号長
k = BLOCK_SIZE # 情報長

# systematic（組織符号）の場合、Gの形状は (k, n)
try:
    H, G = make_ldpc(n, d_v, d_c, systematic=True, sparse=True)
    print(f"LDPC matrix created for block size {k} -> code length {n}")
except Exception as e:
    print(f"Error creating LDPC matrix: {e}")
    exit()

k_actual = G.shape[1]        # G の列数 (pyldpcのsparse Gは n x k)
n_actual = G.shape[0]        # G の行数
print(f"Actual k = {k_actual}, n = {n_actual}")
dic_time = {}
dic_num = {}

# --- 修正点 2: ファイルリストをソートする ---
try:
    file_list = sorted(os.listdir(path))
except FileNotFoundError:
    print(f"Error: Input directory not found at {path}")
    exit()

for d in file_list:
    # 画像ファイル以外をスキップ (念のため)
    if not (d.endswith('.png') or d.endswith('.jpg') or d.endswith('.jpeg')):
        print(f"Skipping non-image file: {d}")
        continue
    
    try:
        image = Image.open(os.path.join(path, d)).resize((256, 256))
    except Exception as e:
        print(f"Error opening image {d}: {e}")
        continue # このファイルをスキップ
        
    image_id = ""
    for i in range(len(d)):
        if d[i].isdigit():
            image_id += d[i]
    
    if not image_id:
        print(f"Skipping file {d}: No digits found in filename.")
        continue

    print("Processing image ID:", image_id)
    image_data = np.array(image)
    original_shape = image_data.shape
    #print(image_data.shape)  # (256, 256)
    binary_image = np.unpackbits(image_data)
    original_bits_len = len(binary_image)
    rem = len(binary_image) % k_actual
    if rem != 0:
        padded_binary_image = np.hstack((binary_image, np.zeros(k_actual - rem, dtype=np.uint8)))
    else:
        padded_binary_image = binary_image
    
    blocks = padded_binary_image.reshape(-1, k_actual)
    
    # --- 修正点 3: ログのkをk_actualに修正 (任意) ---
    print(f"Image {d} divided into {blocks.shape[0]} blocks of size {k_actual} bits.")
    
    
    for snr in range(-10,20, 1):
        decoded_blocks = []
        print(f"SNR = {snr} dB")
        start_time = time.time()
        for i, block in enumerate(blocks):
            if (i+1) % 100 == 0:
                print(f"Encoding block {i+1}/{blocks.shape[0]}")
            
            encoded_block = encode(G, block, snr)
            decoded_block = decode(H, encoded_block, snr, maxiter=100)
            
            # (Remember to apply the fix from last time here!)
            message_block = decoded_block[:k_actual] 
            decoded_blocks.append(message_block)
        decoded_binary_image = np.hstack(decoded_blocks)
        decoded_binary_image = decoded_binary_image[:original_bits_len]
        reconstructed_image_data = np.packbits(decoded_binary_image).reshape(original_shape)
        reconstructed_image = Image.fromarray(reconstructed_image_data.astype(np.uint8))
        end_time = time.time()
        elapsed_time = end_time - start_time
        dic_time[snr] = dic_time.get(snr, 0) + elapsed_time
        dic_num[snr] = dic_num.get(snr, 0) + 1
        print(f"execute time = {elapsed_time}")
        reconstructed_image.save(os.path.join(pathr, f"output_{snr}_{image_id}.png"))

for k, v in dic_time.items():
    print(f"SNR = {k}, Average Execute Time = {v/dic_num[k]}")