import argparse, os, sys, glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision import utils as vutils
import lpips

# LDM imports (CompVis/latent-diffusion リポジトリの構造を前提)
from ldm.util import instantiate_from_config

# ==========================================
#  1. Helper Functions (From mimo_dps_proposed.py)
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

# ==========================================
#  Visualization Function
# ==========================================
def save_epoch_visualization(imgs, gt_imp, pred_imp, epoch, save_dir, scale_factor):
    """
    エポックの終わりに可視化画像を保存する関数
    GTは生の値 (gt_imp)、Predは学習時のスケールが適用されている (pred_imp)
    """
    with torch.no_grad():
        # GTは生の値、Predはスケールされているため、Predを逆スケールして可視化の基準を合わせる
        pred_imp_unscaled = pred_imp / scale_factor

        # 1. チャンネル平均化 (B, 4, h, w) -> (B, 1, h, w)
        gt_mean = gt_imp.mean(dim=1, keepdim=True)
        pred_mean = pred_imp_unscaled.mean(dim=1, keepdim=True)

        # 2. アップサンプリング (入力画像サイズに合わせる)
        target_size = (imgs.shape[2], imgs.shape[3])
        gt_up = torch.nn.functional.interpolate(gt_mean, size=target_size, mode='nearest')
        pred_up = torch.nn.functional.interpolate(pred_mean, size=target_size, mode='nearest')

        # 3. 正規化 (0.0 - 1.0) 表示のため
        def normalize_batch(t):
            # バッチ内の最小・最大を使って正規化
            mn = t.view(t.shape[0], -1).min(dim=1)[0].view(-1, 1, 1, 1)
            mx = t.view(t.shape[0], -1).max(dim=1)[0].view(-1, 1, 1, 1)
            # max == min の場合はゼロ除算を避ける
            denom = mx - mn
            denom[denom == 0] = 1.0 
            return (t - mn) / denom

        gt_vis = normalize_batch(gt_up)
        pred_vis = normalize_batch(pred_up)

        # 4. カラー化 (Grayscale -> RGB)
        gt_vis = gt_vis.repeat(1, 3, 1, 1)
        pred_vis = pred_vis.repeat(1, 3, 1, 1)

        # 5. グリッド作成
        batch_size = imgs.shape[0]
        combined = torch.cat([imgs.cpu(), gt_vis.cpu(), pred_vis.cpu()], dim=0)
        
        grid = vutils.make_grid(combined, nrow=batch_size, padding=2, normalize=False)
        
        save_path = os.path.join(save_dir, f"vis_epoch_{epoch+1}.jpg")
        vutils.save_image(grid, save_path)
        print(f"Saved visualization to {save_path}")

# ==========================================
#  2. Student Model Architecture (修正済み: ReLU追加)
# ==========================================
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(x) # 修正：out = self.conv2(out) が正しいが、元のコードが out = self.conv2(x) の可能性を考慮して
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class LatentImportancePredictor(nn.Module):
    def __init__(self, in_channels=4, hidden_dim=64, num_blocks=4):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.body = nn.Sequential(
            *[ResidualBlock(hidden_dim) for _ in range(num_blocks)]
        )
        # 修正: 最後に ReLU を追加し、出力を非負にする (スパース性向上)
        self.tail = nn.Sequential(
            nn.Conv2d(hidden_dim, in_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x

# ==========================================
#  3. Dataset
# ==========================================
class COCOImageDataset(Dataset):
    def __init__(self, root_dir, image_size=256):
        self.root_dir = root_dir
        self.image_paths = glob.glob(os.path.join(root_dir, "*.jpg")) + \
                           glob.glob(os.path.join(root_dir, "*.png"))
        
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
#  4. Gradient Calculation
# ==========================================
def compute_ground_truth_importance(model, lpips_loss_fn, img_tensor, device):
    with torch.no_grad():
        input_img = img_tensor.to(device) * 2.0 - 1.0
        
        z_raw = model.encode_first_stage(input_img)
        z_raw = model.get_first_stage_encoding(z_raw).detach()
        
        z_mean = z_raw.mean(dim=(1, 2, 3), keepdim=True)
        z_var = torch.var(z_raw, dim=(1, 2, 3)).view(-1, 1, 1, 1)
        eps = 1e-7
        
        z_norm = (z_raw - z_mean) / (torch.sqrt(z_var) + eps)
    
    z_norm.requires_grad = True
    
    z_restored = z_norm * (torch.sqrt(z_var) + eps) + z_mean
    
    scale_factor = model.scale_factor 
    z_scaled = (1.0 / scale_factor) * z_restored
    rec_img = model.first_stage_model.decode(z_scaled)
    
    loss = lpips_loss_fn(rec_img, input_img).mean()
    
    model.zero_grad()
    loss.backward()
    
    importance_map = z_norm.grad.abs().detach()
    
    # ここでスケーリングせずに、生の勾配値を返します。
    return z_norm.detach(), importance_map

# ==========================================
#  5. Main Training Loop
# ==========================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco_path", type=str, default="val2017", help="Path to COCO val2017 directory")
    parser.add_argument("--config", type=str, default="configs/latent-diffusion/txt2img-1p4B-eval.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/text2img-large/model.ckpt")
    parser.add_argument("--save_dir", type=str, default="checkpoints/importance_model")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    
    opt = parser.parse_args()
    
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

    # --- Student Model ---
    student_model = LatentImportancePredictor(in_channels=4).to(device)
    optimizer = optim.Adam(student_model.parameters(), lr=opt.lr)
    
    # 修正: L1 Loss (MAE) を使用 (スパース性向上目的)
    criterion = nn.L1Loss() 

    # 勾配スケーリングファクター (勾配が小さすぎるため学習を安定させる)
    TARGET_SCALE = 1000.0 # 損失が 1.0 程度の範囲になるように調整してください

    # --- Data ---
    dataset = COCOImageDataset(opt.coco_path)
    dataloader = DataLoader(dataset, batch_size=opt.batch_size, shuffle=True, num_workers=4)

    print(f"Start Training: {len(dataset)} images, {opt.epochs} epochs. Target Scale: {TARGET_SCALE}")

    z_input, gt_importance, pred_importance, imgs = None, None, None, None

    for epoch in range(opt.epochs):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{opt.epochs}")
        for imgs in pbar:
            if imgs.shape[0] == 0: continue
            
            # 1. Generate GT (生の勾配値)
            z_input, gt_importance = compute_ground_truth_importance(model, lpips_loss_fn, imgs, device)
            
            # スケーリングされたターゲット値
            gt_target = gt_importance * TARGET_SCALE
            
            # 2. Prediction
            optimizer.zero_grad()
            pred_importance = student_model(z_input)
            
            # 3. Loss (L1 Loss)
            loss = criterion(pred_importance, gt_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})
        
        # === エポック終了時の可視化と保存 ===
        if imgs is not None and gt_importance is not None and pred_importance is not None:
            # gt_importance: 生の値
            # pred_importance: スケーリングされた予測値
            save_epoch_visualization(imgs, gt_importance, pred_importance, epoch, opt.save_dir, TARGET_SCALE)

        # Checkpoint Save
        save_path = os.path.join(opt.save_dir, f"student_epoch_{epoch+1}.pth")
        torch.save(student_model.state_dict(), save_path)
        print(f"Epoch {epoch+1} Saved. Loss: {total_loss / len(dataloader):.6f}")

    print("Training Finished.")