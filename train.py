import os, sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import matplotlib.pyplot as plt
import kornia.color as kc
from kornia.losses import SSIMLoss
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import ToPILImage

# --- PATH SETUP ---
# root/
#   dataset/
#   model/
#   train/
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "dataset"))
sys.path.append(str(project_root / "model"))


from dataset.kaist_dataset_video_split import KAIST_ThermalEventRGBDatasetVideo
from model.THEV_Net import ThermalEvent2RGBNet 
from torch.utils.data import DataLoader

# === ARGS ===
parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, choices=["full","thermal-only","event-only"], default="full")
parser.add_argument("--encoder_type", type=str, choices=["2d","3d"], default="3d")
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--fusion", type=str, choices=["gated","add","concat"], default="gated")

# split video
parser.add_argument("--split_json", type=str, default=str(project_root / "utils" / "video_split_day_global.json"))

# loss weights
parser.add_argument("--w_l1", type=float, default=0.6)
parser.add_argument("--w_ssim", type=float, default=0.4)
parser.add_argument("--w_ab", type=float, default=0.8)

# toggle loss terms
parser.add_argument("--smooth_l1", action="store_true")
parser.add_argument("--ms_ssim", action="store_true")
parser.add_argument("--ab_loss", action="store_true")

args = parser.parse_args()

print(f"""
=== CONFIG ===
• MODE           : {args.mode}
• ENCODER TYPE   : {args.encoder_type}
• FUSION         : {args.fusion}
• SPLIT JSON     : {args.split_json}
• WEIGHTS (L1/SSIM/AB): {args.w_l1}/{args.w_ssim}/{args.w_ab}
• LOSS TERMS     : smooth_l1={args.smooth_l1} | ms_ssim={args.ms_ssim} | ab_loss={args.ab_loss}
• GPU            : {args.device}
""")

# === CONFIG ENV ===
# CHANGE WITH YOUR PAHTS
os.environ["TORCH_HOME"] = "/medias/db/ImagingSecurity_misc/melcarne/.torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = torch.device(f"cuda:{args.device}")

# === PATHS ===
# CHANGE WITH YOUR PAHTS
thermal_dir = "/medias/db/kaist/rgbt-ped-detection/data/kaist-rgbt/images"
rgb_dir     = "/medias/db/kaist/rgbt-ped-detection/data/kaist-rgbt/images"
event_dir   = "/medias/db/ImagingSecurity_misc/kaist-rgbt/test/voxel_grid_soft_rgb"

# === Build ablation tag for naming ===
abl_off = []
if not args.smooth_l1: abl_off.append("smoothl1")
if not args.ms_ssim:   abl_off.append("msssim")
if not args.ab_loss:   abl_off.append("ab")
abl_tag = "" if not abl_off else "_ablation_" + "_".join(abl_off) 

# === Use the tag in paths/names ===
run_tag   = f"videoSplit_{args.encoder_type}_fusion{args.fusion}_{args.mode}{abl_tag}"
save_dir  = str(project_root / f"outputs/train_results_{run_tag}")
ckpt_dir  = "/medias/db/ImagingSecurity_misc/melcarne/Multimodal-Data-Fusion/Kaist-EV/checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)
model_name = f"best_{run_tag}"
os.makedirs(save_dir, exist_ok=True)

# === DATASETS ===
train_dataset = KAIST_ThermalEventRGBDatasetVideo(
    thermal_root=thermal_dir, rgb_root=rgb_dir, voxel_root=event_dir,
    split_json=args.split_json, mode="train", target_size=(256,256),
    skip_empty_voxel=True, use_filter_csv=True, filter_csv_path=None
)
val_dataset = KAIST_ThermalEventRGBDatasetVideo(
    thermal_root=thermal_dir, rgb_root=rgb_dir, voxel_root=event_dir,
    split_json=args.split_json, mode="val", target_size=(256,256),
    skip_empty_voxel=True, use_filter_csv=True, filter_csv_path=None
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True,
                          num_workers=8, pin_memory=True,
                          persistent_workers=True, drop_last=True)
val_loader   = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=8, pin_memory=True)

# === MODEL ===
use_events  = args.mode in ["full","event-only"]
use_thermal = args.mode in ["full","thermal-only"]

model = ThermalEvent2RGBNet(
    event_bins=5, 
    dropout=0.2,
    encoder_type=args.encoder_type, 
    base=32,
    norm='gn',
    fusion=args.fusion 
).to(device)

# === LOSSES  ===

class MSSSIM_L_Loss(nn.Module):
    """ MS-SSIM only over  Luminance (Lab), robust to drift ±px. """
    def __init__(self, window=11):
        super().__init__()
        self.ssim = SSIMLoss(window_size=window, reduction='mean')
    def forward(self, pred_rgb, gt_rgb):
        pred_lab = kc.rgb_to_lab(pred_rgb)
        gt_lab   = kc.rgb_to_lab(gt_rgb)
        Lp = pred_lab[:, :1] / 100.0
        Lg = gt_lab[:, :1] / 100.0
        # 3 scale (1, 1/2, 1/4)
        loss = self.ssim(Lp, Lg)
        Lp2, Lg2 = F.avg_pool2d(Lp, 2), F.avg_pool2d(Lg, 2)
        Lp4, Lg4 = F.avg_pool2d(Lp2, 2), F.avg_pool2d(Lg2, 2)
        loss = 0.5*loss + 0.3*self.ssim(Lp2, Lg2) + 0.2*self.ssim(Lp4, Lg4)
        return loss

class EventGuidedABLoss(nn.Module):
    """
    L1 in ab of Lab, weighted by a mask event-driven.
    """
    def __init__(self, blur_ks=9):
        super().__init__()
        self.l1 = nn.L1Loss(reduction='none')
        self.blur_ks = blur_ks
    def forward(self, out_lab, gt_rgb, events):
        # out_lab: [B,3,H,W] with L∈[0,1],ab∈[-1,1]
        gt_lab = kc.rgb_to_lab(gt_rgb)
        gt_ab  = gt_lab[:, 1:] / 128.0
        pred_ab = out_lab[:, 1:]  # already in [-1,1]

        if events is None:
            return self.l1(pred_ab, gt_ab).mean()

        # events: [B,T,H,W] o [B,T,1,H,W]
        if events.dim() == 5:
            events = events.squeeze(2)
        dens = events.abs().mean(dim=1, keepdim=True)  # [B,1,H,W]
        dens = dens / (dens.amax(dim=[2,3], keepdim=True) + 1e-8)
        # light blur 
        pad = self.blur_ks // 2
        dens = F.avg_pool2d(F.pad(dens, (pad,pad,pad,pad), mode='replicate'), self.blur_ks, stride=1)
        mask = dens.clamp(0,1).expand(-1,2,-1,-1)      # [B,2,H,W]

        l1_map = self.l1(pred_ab, gt_ab)               # [B,2,H,W]
        return (l1_map * (0.2 + 0.8*mask)).mean()       

class HybridLoss(nn.Module):
    def __init__(self, enable_smooth_l1: bool, enable_ms_ssim: bool, enable_ab: bool,
                 w_l1=0.6, w_ssim=0.4, w_ab=0.8, blur_ks=9):
        super().__init__()
        self.enable_s1   = enable_smooth_l1
        self.enable_mss  = enable_ms_ssim
        self.enable_ab   = enable_ab
        self.w_l1, self.w_ssim, self.w_ab = w_l1, w_ssim, w_ab

        if self.enable_s1:
            self.smooth_l1 = nn.SmoothL1Loss(beta=0.01)
        if self.enable_mss:
            self.ms_ssim_L = MSSSIM_L_Loss(window=11)
        if self.enable_ab:
            self.ab_loss   = EventGuidedABLoss(blur_ks=blur_ks)

    def forward(self, out_lab, pred_rgb, gt_rgb, events=None):
        terms = []

        if self.enable_s1:
            terms.append(self.w_l1 * self.smooth_l1(pred_rgb, gt_rgb))

        if self.enable_mss:
            terms.append(self.w_ssim * self.ms_ssim_L(pred_rgb, gt_rgb))

        if self.enable_ab:
            if events is not None:
                terms.append(self.w_ab * self.ab_loss(out_lab, gt_rgb, events))
            else:
                # Fallback (ablation)
                # O ( thermal-only)
                pass

        if len(terms) == 0:
            return torch.tensor(0.0, device=pred_rgb.device, requires_grad=True), 0.0

        # sum
        loss = torch.stack([t if torch.is_tensor(t) else torch.tensor(t, device=pred_rgb.device)
                            for t in terms]).sum()
        return loss, 0.0

hybrid_loss = HybridLoss(
    enable_smooth_l1=args.smooth_l1,
    enable_ms_ssim=args.ms_ssim,
    enable_ab=args.ab_loss,
    w_l1=args.w_l1, w_ssim=args.w_ssim, w_ab=args.w_ab
) 

# === OPTIMIZER & SCHEDULER ===
optimizer = optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=5, verbose=True, min_lr=1e-7
)

# === EMA (Exponential Moving Average) ===
ema_decay = 0.999
ema = {k: v.detach().clone() for k, v in model.state_dict().items()}

def update_ema(model, ema_dict, decay=0.999):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_dict[k].mul_(decay).add_(v.detach(), alpha=1 - decay)

def load_ema(model, ema_dict):
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(ema_dict, strict=False)
    return backup

# === TRAIN LOOP ===
best_val = float('inf')
epochs_no_improve = 0  
to_pil = ToPILImage()
train_hist, val_hist = [], []

def lab_to_rgb(out_lab):
    return kc.lab_to_rgb(torch.cat([out_lab[:, :1]*100.0, out_lab[:, 1:]*128.0], dim=1)).clamp(0,1)

for epoch in range(0, 200):
    # ---- TRAIN ----
    model.train()
    tot = 0.0
    for ib, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch} [TRAIN]")):
        thermal = batch["thermal"].to(device)
        events  = batch["events"].to(device)
        gt      = batch["rgb_gt"].to(device)

        if not use_events:  events = None
        if not use_thermal: thermal = None

        optimizer.zero_grad()
        out_lab = model(thermal, events)
        pred_rgb = lab_to_rgb(out_lab)

        loss, _ = hybrid_loss(out_lab, pred_rgb, gt, events if use_events else None)

        loss.backward()
        optimizer.step()
        update_ema(model, ema, ema_decay) 
        tot += loss.item()

    tr_loss = tot / max(1, len(train_loader))
    train_hist.append(tr_loss)

    # ---- VAL ----
    model.eval()
    backup = load_ema(model, ema) 

    tot_val, ps, st = 0.0, [], []
    with torch.no_grad():
        for batch in val_loader:
            thermal = batch["thermal"].to(device)
            events  = batch["events"].to(device)
            gt      = batch["rgb_gt"].to(device)

            if not use_events:  events = None
            if not use_thermal: thermal = None

            out_lab = model(thermal, events)
            pred_rgb = lab_to_rgb(out_lab)

            vloss, _ = hybrid_loss(out_lab, pred_rgb, gt, events if use_events else None)

            tot_val += vloss.item()

            p = pred_rgb.cpu().numpy().transpose(0,2,3,1)
            g = gt.cpu().numpy().transpose(0,2,3,1)
            for pi,gi in zip(p,g):
                ps.append(psnr(gi, pi, data_range=1.0))
                st.append(ssim(gi, pi, channel_axis=-1, data_range=1.0))

    model.load_state_dict(backup, strict=False) 

    val_loss = tot_val / max(1, len(val_loader))
    val_hist.append(val_loss)
    scheduler.step(val_loss)
    
    # Check fusion alpha only if gated mode is active
    alphas_info = ""
    if hasattr(model, 'fusion') and model.fusion == 'gated' and hasattr(model, 'f1'):
         # Retrieve alpha values safely
         a1 = model.f1.alpha_mean.item() if model.f1.alpha_mean is not None else -1
         a2 = model.f2.alpha_mean.item() if model.f2.alpha_mean is not None else -1
         a3 = model.f3.alpha_mean.item() if model.f3.alpha_mean is not None else -1
         a4 = model.f4.alpha_mean.item() if model.f4.alpha_mean is not None else -1
         alphas_info = f"| Fusion α: {a1:.2f}, {a2:.2f}, {a3:.2f}, {a4:.2f}"

    print(f"[Epoch {epoch}] LR: {scheduler.optimizer.param_groups[0]['lr']:.2e} | "
          f"Train: {tr_loss:.4f} | Val: {val_loss:.4f} | PSNR: {np.mean(ps):.2f} | SSIM: {np.mean(st):.4f} "
          f"{alphas_info}")

    if val_loss < best_val:
        best_val = val_loss

        # save EMA state dict
        temp_backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(ema, strict=False)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_val_loss": best_val
        }, os.path.join(ckpt_dir, f"{model_name}.pth"))
        model.load_state_dict(temp_backup, strict=False)

        print("✓ Best model saved.")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"No improvement for {epochs_no_improve} epoch(s)")

    if epochs_no_improve >= 15:
        print("Early stopping triggered.")
        break

    # sample each 5 epochs
    if (epoch % 5) == 0:
        sample = next(iter(val_loader))
        with torch.no_grad():
            thermal = sample["thermal"].to(device)
            events  = sample["events"].to(device)
            if not use_events:  events = None
            if not use_thermal: thermal = None
            out_lab = model(thermal, events)
            out_rgb = lab_to_rgb(out_lab).clamp(0,1).cpu()
            pred_dir = os.path.join(save_dir, f"epoch{epoch:03d}_samples")
            os.makedirs(pred_dir, exist_ok=True)
            ToPILImage()(out_rgb[0]).save(os.path.join(pred_dir, "val_pred.png"))

# save curves
log_df = pd.DataFrame({"epoch": list(range(1, len(train_hist)+1)),
                       "train_loss": train_hist, "val_loss": val_hist})
log_df.to_csv(os.path.join(save_dir, "loss_curves.csv"), index=False)
plt.figure(figsize=(8,5))
plt.plot(log_df["epoch"], log_df["train_loss"], label="Train", linewidth=2)
plt.plot(log_df["epoch"], log_df["val_loss"], label="Val", linewidth=2)
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training/Validation")
plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig(os.path.join(save_dir, "loss_plot.png")); plt.close()
print("✓ Loss plot saved.")

# Example run command:
# python train.py --fusion gated --mode full --encoder_type 3d --smooth_l1 --ms_ssim --ab_loss --device 0