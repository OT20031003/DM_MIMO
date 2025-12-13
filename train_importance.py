import argparse, os, sys, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random  # 【修正】ここを追加しました
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import utils as vutils
import lpips
from einops import rearrange

# LDM imports (CompVis/latent-diffusion リポジトリの構造を前提)
from ldm.util import instantiate_from_config

# ==========================================
#  1. Helper Functions
# ==========================================
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if verbose:
        if len(m) > 0: print("missing keys:", m)
        if len(u) > 0: print("unexpected keys:", u)
    model.cuda()
    model.eval()
    return model

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ==========================================
#  2. Visualization Function (ViT対応版)
# ==========================================
def save_epoch_visualization(imgs, gt_imp_block, pred_imp_block, epoch, save_dir, scale_factor):
    """
    可視化用関数
    gt_imp_block, pred_imp_block: (B, 1, 8, 8) のブロック単位重要度
    """
    with torch.no_grad():
        # Predを逆スケールして値のレンジを合わせる（表示用）
        pred_imp_unscaled = pred_imp_block / scale_factor
        
        # 入力画像サイズ (256x256想定)
        target_size = (imgs.shape[2], imgs.shape[3])
        
        # 8x8 のマップを 256x256 に拡大 (Nearest Neighborでブロック感を残す)
        gt_up = torch.nn.functional.interpolate(gt_imp_block, size=target_size, mode='nearest')
        pred_up = torch.nn.functional.interpolate(pred_imp_unscaled, size=target_size, mode='nearest')

        # 正規化 (0.0 - 1.0) してヒートマップ化
        def normalize_batch(t):
            mn = t.view(t.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
            mx = t.view(t.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
            denom = mx - mn
            denom[denom == 0] = 1.0 
            return (t - mn) / denom

        gt_vis = normalize_batch(gt_up)
        pred_vis = normalize_batch(pred_up)

        # Grayscale -> RGB
        gt_vis = gt_vis.repeat(1, 3, 1, 1)
        pred_vis = pred_vis.repeat(1, 3, 1, 1)

        # 画像結合: [元画像, 正解(ブロック), 予測(ブロック)]
        combined = torch.cat([imgs.cpu(), gt_vis.cpu(), pred_vis.cpu()], dim=0)
        
        batch_size = imgs.shape[0]
        # グリッド作成 (各行が1つのサンプル)
        grid = vutils.make_grid(combined, nrow=batch_size, padding=2, normalize=False)
        
        save_path = os.path.join(save_dir, f"vis_epoch_{epoch+1}.jpg")
        vutils.save_image(grid, save_path)
        print(f"Saved visualization to {save_path}")

# ==========================================
#  3. ViT Model Architecture (Latent ViT)
# ==========================================
class LatentViTImportance(nn.Module):
    def __init__(self, in_channels=4, patch_size=4, latent_size=32, dim=128, depth=4, heads=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_size = patch_size
        # 32x32 / 4x4 = 8x8 = 64 patches
        self.num_patches = (latent_size // patch_size) ** 2
        self.grid_size = latent_size // patch_size # 8
        
        # 1. Patch Embedding & Flatten
        # Conv2d (stride=patch_size) でパッチ分割と埋め込みを同時に行う
        self.patch_to_embedding = nn.Conv2d(
            in_channels, dim, 
            kernel_size=patch_size, stride=patch_size
        )
        
        # 2. Positional Embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches, dim))
        self.dropout = nn.Dropout(dropout)
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, 
            nhead=heads, 
            dim_feedforward=mlp_dim, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # 4. Output Head -> 重要度スコア (非負)
        self.to_importance = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1),
            nn.ReLU() # 重要度は「量」なのでReLUで非負にする
        )

    def forward(self, x):
        # x: (B, 4, 32, 32)
        
        # Embedding: (B, dim, 8, 8)
        x = self.patch_to_embedding(x)
        
        # Flatten: (B, dim, 64) -> Transpose: (B, 64, dim)
        x = x.flatten(2).transpose(1, 2)
        
        # Add Position
        x += self.pos_embedding
        x = self.dropout(x)
        
        # Transformer
        x = self.transformer(x) # (B, 64, dim)
        
        # Head
        x = self.to_importance(x) # (B, 64, 1)
        
        # Reshape back to grid: (B, 1, 8, 8)
        return rearrange(x, 'b (h w) c -> b c h w', h=self.grid_size, w=self.grid_size)

# ==========================================
#  4. Dataset
# ==========================================
class COCOImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        # jpg, png対応
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                           glob.glob(os.path.join(root_dir, "*.png")) + \
                           glob.glob(os.path.join(root_dir, "*.jpeg"))
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        try:
            image = Image.open(path).convert("RGB")
            return self.transform(image)
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 256, 256)

# ==========================================
#  5. Gradient Calculation (Ground Truth)
# ==========================================
def compute_ground_truth_importance(model, lpips_loss_fn, img_tensor, device):
    """
    入力画像に対するLatentのLPIPS勾配を計算する
    戻り値:
      z_norm: (B, 4, 32, 32) 正規化されたLatent
      importance_map: (B, 4, 32, 32) 各画素の勾配絶対値
    """
    with torch.no_grad():
        # LDMは [-1, 1] 入力を期待
        input_img = img_tensor.to(device) * 2.0 - 1.0
        
        z_raw = model.encode_first_stage(input_img)
        z_raw = model.get_first_stage_encoding(z_raw).detach()
        
        z_mean = z_raw.mean(dim=(1, 2, 3), keepdim=True)
        z_var = torch.var(z_raw, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        eps = 1e-7
        
        z_norm = (z_raw - z_mean) / (torch.sqrt(z_var) + eps)
    
    # 勾配計算のため requires_grad
    z_norm.requires_grad = True
    
    # デコード
    z_restored = z_norm * (torch.sqrt(z_var) + eps) + z_mean
    scale_factor = model.scale_factor 
    z_scaled = (1.0 / scale_factor) * z_restored
    rec_img = model.first_stage_model.decode(z_scaled)
    
    # LPIPS Loss
    loss = lpips_loss_fn(rec_img, input_img).mean()
    
    model.zero_grad()
    loss.backward()
    
    importance_map = z_norm.grad.abs().detach()
    
    return z_norm.detach(), importance_map

# ==========================================
#  6. Main Training Loop
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, default="val2017", help="Path to COCO images")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/text2img-large/model.ckpt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/importance_vit")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    
    opt = parser.parse_args()
    seed_everything(opt.seed)
    
    os.makedirs(opt.save_dir, exist_ok=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # --- Load LDM ---
    config = OmegaConf.load(opt.config)
    model = load_model_from_config(config, opt.ckpt)
    for param in model.parameters():
        param.requires_grad = False

    # --- Load LPIPS ---
    print("Loading LPIPS...")
    lpips_loss_fn = lpips.LPIPS(net='alex').to(device)
    lpips_loss_fn.eval()
    for param in lpips_loss_fn.parameters():
        param.requires_grad = False

    # --- Student Model (ViT) ---
    print("Initializing Vision Transformer for Latent Importance...")
    student_model = LatentViTImportance(
        in_channels=4, 
        patch_size=4, 
        latent_size=32, 
        dim=128, 
        depth=4, 
        heads=4
    ).to(device)
    
    optimizer = optim.Adam(student_model.parameters(), lr=opt.lr)
    criterion = nn.L1Loss() # MAE

    # 勾配スケーリングファクター (勾配値が小さいため)
    TARGET_SCALE = 1000.0 

    # --- Data ---
    dataset = COCOImageDataset(opt.coco_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    print(f"Start Training: {len(dataset)} images, {opt.epochs} epochs.")

    # 可視化用に直前のバッチデータを保持する変数
    last_imgs, last_gt_block, last_pred_block = None, None, None

    for epoch in range(opt.epochs):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for imgs in pbar:
            if imgs.shape[0] == 0: continue
            
            # 1. Generate GT Importance (High Res: 32x32)
            z_input, gt_importance_highres = compute_ground_truth_importance(model, lpips_loss_fn, imgs, device)
            
            # 2. Downsample GT to Block Importance (8x8)
            # (B, 4, 32, 32) -> Channel Mean -> (B, 1, 32, 32) -> AvgPool -> (B, 1, 8, 8)
            gt_imp_mean = gt_importance_highres.mean(dim=1, keepdim=True)
            gt_target_block = torch.nn.functional.avg_pool2d(gt_imp_mean, kernel_size=4, stride=4)
            
            # Scale Target
            gt_target_scaled = gt_target_block * TARGET_SCALE
            
            # 3. ViT Prediction
            optimizer.zero_grad()
            # input z is (B, 4, 32, 32), output is (B, 1, 8, 8)
            pred_importance = student_model(z_input)
            
            # 4. Loss
            loss = criterion(pred_importance, gt_target_scaled)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
            
            # 保持 (可視化用)
            last_imgs = imgs
            last_gt_block = gt_target_scaled
            last_pred_block = pred_importance

        # === エポック終了時の可視化と保存 ===
        if last_imgs is not None:
            # GTはスケール済みなのでそのまま、Predもスケール済みが出ている
            save_epoch_visualization(last_imgs, last_gt_block, last_pred_block, epoch, opt.save_dir, TARGET_SCALE)

        # Checkpoint Save
        save_path = os.path.join(opt.save_dir, f"vit_importance_epoch_{epoch+1}.pth")
        torch.save(student_model.state_dict(), save_path)
        print(f"Epoch {epoch+1} Saved. Avg Loss: {total_loss / len(dataloader):.6f}")

    print("Training Finished.")