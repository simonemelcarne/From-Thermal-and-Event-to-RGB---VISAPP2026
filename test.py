import os
import sys
import argparse
from pathlib import Path
import numpy as np
from tqdm import tqdm
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.utils import make_grid, save_image
from torchvision.transforms import ToPILImage

from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import kornia.color as kc
from skimage.color import rgb2lab as sk_rgb2lab, deltaE_ciede2000 

# --- PATH SETUP ---
# root/
#   dataset/
#   model/
#   test.py (this file)
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "dataset"))
sys.path.append(str(project_root / "model"))

from dataset.kaist_dataset_video_split import KAIST_ThermalEventRGBDatasetVideo
from model.THEV_Net import ThermalEvent2RGBNet


# =============== ARGS ===============
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=int, default=0)
parser.add_argument("--mode", type=str, choices=["full","thermal-only","event-only"], default="full")
parser.add_argument("--encoder_type", type=str, choices=["2d","3d"], default="3d")
parser.add_argument("--norm", type=str, choices=["bn","gn"], default="gn")

# Paths (Adjust defaults as needed)
parser.add_argument("--split_json", type=str, default="utils/video_split_day_global.json")
parser.add_argument("--thermal_dir", type=str, required=True, help="Path to KAIST thermal images")
parser.add_argument("--rgb_dir",     type=str, required=True, help="Path to KAIST RGB images")
parser.add_argument("--event_dir",   type=str, required=True, help="Path to Event Voxel Grids")

parser.add_argument("--fusion", type=str, choices=["gated","add","concat"], default="gated")

parser.add_argument("--ckpt_dir", type=str, default="checkpoints")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--num_workers", type=int, default=8)
parser.add_argument("--save_every", type=int, default=50)

# Loss terms tags (used only to find the correct checkpoint filename)
parser.add_argument("--smooth_l1", action="store_true")
parser.add_argument("--ms_ssim", action="store_true")
parser.add_argument("--ab_loss", action="store_true")

parser.add_argument("--save_separate", action="store_true", help="Save individual images to separate folders")
parser.add_argument("--out_dir", type=str, default="outputs/eval_results", help="Root folder for results")

args = parser.parse_args()


print(f"""
=== EVAL CONFIG ===
• MODE            : {args.mode}
• ENCODER TYPE    : {args.encoder_type}
• NORM            : {args.norm}
• SPLIT JSON      : {args.split_json}
• CKPT DIR        : {args.ckpt_dir}
• BATCH / WORKERS : {args.batch_size} / {args.num_workers}
• FUSION          : {args.fusion}
• SAVE EVERY      : {args.save_every}
• SAVE SEPARATE   : {args.save_separate}
""")


# =============== ENV/DEVICE ===============
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# =============== UTILS ===============
to_pil = ToPILImage()

def lab_to_rgb(out_lab: torch.Tensor) -> torch.Tensor:
    """
    out_lab: [B,3,H,W], L in [0,1], ab in [-1,1]
    returns RGB in [0,1]
    """
    lab = torch.cat([out_lab[:, :1] * 100.0, out_lab[:, 1:] * 128.0], dim=1)
    rgb = kc.lab_to_rgb(lab).clamp(0, 1)
    return rgb

def events_to_vis(events: torch.Tensor) -> torch.Tensor:
    """
    Visualization of events: Red (positive) / Blue (negative)
    """
    if events is None:
        return None
    if events.dim() == 5:         # [B,T,1,H,W] -> [B,T,H,W]
        events = events.squeeze(2)

    proj = events.sum(dim=1)      # [B,H,W]

    B, H, W = proj.shape
    abs_flat = proj.abs().view(B, -1)
    q = torch.quantile(abs_flat, 0.99, dim=1).clamp_min(1e-6)  # [B]
    proj_norm = (proj / q.view(B, 1, 1)).clamp(-1, 1)          # [-1,1]

    img = torch.ones(B, 3, H, W, device=proj.device, dtype=proj.dtype)

    pos = proj_norm > 0
    neg = proj_norm < 0

    x_pos = proj_norm[pos]
    img[:, 1, :, :][pos] = (1.0 - x_pos)
    img[:, 2, :, :][pos] = (1.0 - x_pos)

    x_neg = proj_norm[neg]
    img[:, 0, :, :][neg] = (1.0 + x_neg)
    img[:, 1, :, :][neg] = (1.0 + x_neg)

    return img.clamp(0, 1)


def thermal_to_vis(thermal: torch.Tensor) -> torch.Tensor:
    if thermal is None:
        return None
    return thermal.repeat(1, 3, 1, 1)

def compute_metrics_batch(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor, lpips_fn=None):
    pred_np = pred_rgb.detach().cpu().numpy().transpose(0,2,3,1)
    gt_np   = gt_rgb.detach().cpu().numpy().transpose(0,2,3,1)
    B = pred_np.shape[0]

    ps_list, ss_list, lp_list = [], [], []
    for i in range(B):
        ps_list.append(psnr(gt_np[i], pred_np[i], data_range=1.0))
        ss_list.append(ssim(gt_np[i], pred_np[i], channel_axis=-1, data_range=1.0))
    if lpips_fn is not None:
        # LPIPS expects [-1,1]
        lp = lpips_fn(pred_rgb*2-1, gt_rgb*2-1).squeeze().detach().cpu().numpy()
        if np.isscalar(lp): lp = [float(lp)]
        lp_list = list(np.array(lp).reshape(-1).tolist())
    else:
        lp_list = [np.nan]*B
    return ps_list, ss_list, lp_list

def compute_deltaE_batch(pred_rgb: torch.Tensor, gt_rgb: torch.Tensor):
    pred_np = pred_rgb.detach().cpu().numpy().transpose(0,2,3,1)
    gt_np   = gt_rgb.detach().cpu().numpy().transpose(0,2,3,1)
    B = pred_np.shape[0]
    de_list = []
    for i in range(B):
        lab_pred = sk_rgb2lab(pred_np[i])  # (H,W,3) in Lab
        lab_gt   = sk_rgb2lab(gt_np[i])
        de = deltaE_ciede2000(lab_gt, lab_pred)  # (H,W)
        de_list.append(float(np.mean(de)))
    return de_list

def make_panel(gt_rgb, thermal_vis, events_vis, pred_rgb):
    tiles = []
    tiles.append(gt_rgb)
    if thermal_vis is not None: tiles.append(thermal_vis)
    else: tiles.append(torch.zeros_like(gt_rgb))
    if events_vis is not None: tiles.append(events_vis)
    else: tiles.append(torch.zeros_like(gt_rgb))
    tiles.append(pred_rgb)
    grid = make_grid(torch.cat(tiles, dim=0), nrow=4, padding=4, pad_value=1.0)
    return grid

def save_summary_csv(df: pd.DataFrame, split_name: str, out_dir: str):
    rows = []
    metrics_present = [m for m in ["psnr", "ssim", "lpips", "deltaE00"] if m in df.columns]
    for m in metrics_present:
        s = df[m].dropna()
        rows.append({
            "metric": m,
            "mean": float(s.mean()) if len(s) else float("nan"),
            "std": float(s.std(ddof=1)) if len(s) > 1 else float("nan"),
            "count": int(s.count())
        })
    summary = pd.DataFrame(rows)
    summary_path = os.path.join(out_dir, f"{split_name}_metrics_summary.csv")
    summary.to_csv(summary_path, index=False)
    print(f"✓ Saved summary: {summary_path}")
    return summary

# =============== DATASETS/LOADERS ===============
def make_loader(mode_split: str):
    ds = KAIST_ThermalEventRGBDatasetVideo(
        thermal_root=args.thermal_dir,
        rgb_root=args.rgb_dir,
        voxel_root=args.event_dir,
        split_json=args.split_json,
        mode=mode_split,                   # 'train'|'val'|'test'
        target_size=(256,256),
        skip_empty_voxel=True,
        use_filter_csv=True,
        filter_csv_path=None
    )
    dl = torch.utils.data.DataLoader(
        ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )
    return ds, dl

val_dataset, val_loader   = make_loader("val")
test_dataset, test_loader = make_loader("test")

# =============== MODEL & CKPT ===============
# Initialized cleanly without CBAM/FiLM
model = ThermalEvent2RGBNet(
    event_bins=5, 
    dropout=0.2,
    encoder_type=args.encoder_type, 
    base=32,
    norm=args.norm,
    fusion=args.fusion
).to(device)


# === Build ablation tag for naming to find CKPT ===
abl_off = []
if not args.smooth_l1: abl_off.append("smoothl1")
if not args.ms_ssim:   abl_off.append("msssim")
if not args.ab_loss:   abl_off.append("ab")
abl_tag = "" if not abl_off else "_ablation_" + "_".join(abl_off) 

# New naming convention consistent with train.py
# Removed "cbam" and "film" from the tag
run_tag_ckpt   = f"videoSplit_{args.encoder_type}_fusion{args.fusion}_{args.mode}{abl_tag}"
run_tag_eval   = f"{args.encoder_type}_fusion{args.fusion}_{args.mode}{abl_tag}"

ckpt_name = f"best_{run_tag_ckpt}.pth"
ckpt_path = os.path.join(args.ckpt_dir, ckpt_name)

if not os.path.exists(ckpt_path):
    print(f"ERROR: Checkpoint not found at {ckpt_path}")
    print("Please check arguments (fusion, mode, loss terms) or check your checkpoints folder.")
    sys.exit(1)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model_state_dict"], strict=False)
model.eval()

# LPIPS
lpips_fn = None
try:
    import lpips
    lpips_fn = lpips.LPIPS(net='alex').to(device).eval()
    print("LPIPS loaded (alex).")
except Exception as e:
    print("WARNING: LPIPS not available. Install with: pip install lpips")
    lpips_fn = None

# =============== EVAL LOOP ===============
out_dir = args.out_dir
os.makedirs(out_dir, exist_ok=True)
vis_dir = os.path.join(out_dir, "visuals")
os.makedirs(vis_dir, exist_ok=True)

# Save single 256x256 image in specific folders
separate_dirs = None
if args.save_separate:
    separate_dirs = {
        "visible": os.path.join(out_dir, "separate", "visible"),
        "thermal": os.path.join(out_dir, "separate", "thermal"),
        "event":   os.path.join(out_dir, "separate", "event"),
        "ours":    os.path.join(out_dir, "separate", "ours"),
    }
    for d in separate_dirs.values():
        os.makedirs(d, exist_ok=True)
# -------------------------------------------

def run_split(split_name: str, loader, dataset):
    split_vis_dir = os.path.join(vis_dir, split_name)
    os.makedirs(split_vis_dir, exist_ok=True)

    rows = []
    idx_global = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {split_name.upper()}"):
            thermal = batch["thermal"].to(device)   # [B,1,H,W]
            events  = batch["events"].to(device)    # [B,T,H,W]
            gt_rgb  = batch["rgb_gt"].to(device)    # [B,3,H,W]
            fnames  = batch["filename"]             # list of strings

            # Mode handling
            thermal_in = thermal if args.mode in ["full","thermal-only"] else None
            events_in  = events  if args.mode in ["full","event-only"]   else None

            # Pred
            out_lab  = model(thermal_in, events_in)
            pred_rgb = lab_to_rgb(out_lab).clamp(0,1)

            # Metrics
            ps_list, ss_list, lp_list = compute_metrics_batch(pred_rgb, gt_rgb, lpips_fn)
            de_list = compute_deltaE_batch(pred_rgb, gt_rgb)

            # ---- COLLAGE EACH N SAMPLES ----
            if args.save_every > 0:
                for i in range(pred_rgb.size(0)):
                    if (idx_global % args.save_every) == 0:
                        gt_vis      = gt_rgb[i:i+1]
                        th_vis      = thermal_to_vis(thermal[i:i+1]) if thermal is not None else None
                        ev_vis      = events_to_vis(events[i:i+1])   if events  is not None else None
                        pred_vis    = pred_rgb[i:i+1]
                        panel = make_panel(gt_vis, th_vis, ev_vis, pred_vis)
                        out_name = f"{split_name}_{idx_global:06d}.png"
                        save_image(panel, os.path.join(split_vis_dir, out_name))
                    idx_global += 1

            # ---- SAVE SEPARATE ----
            if args.save_separate:
                B = pred_rgb.size(0)
                for i in range(B):
                    stem = Path(fnames[i]).stem
                    suffix = f"{split_name}_{stem}.png"
                    
                    # NOTE: Uncomment lines below if you want to save inputs/GT too
                    # save_image(gt_rgb[i:i+1], os.path.join(separate_dirs["visible"], suffix))
                    # save_image(thermal_to_vis(thermal[i:i+1]), os.path.join(separate_dirs["thermal"], suffix))
                    # save_image(events_to_vis(events[i:i+1]), os.path.join(separate_dirs["event"], suffix))
                    
                    # Always save prediction
                    save_image(
                        pred_rgb[i:i+1].detach().cpu(),
                        os.path.join(separate_dirs["ours"], suffix)
                    )

            # rows per-metric
            for i in range(len(fnames)):
                rows.append({
                    "filename": fnames[i],
                    "video_id": batch["video_id"][i],
                    "psnr": ps_list[i],
                    "ssim": ss_list[i],
                    "lpips": lp_list[i],
                    "deltaE00": de_list[i]
                })

    df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, f"{split_name}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ Saved metrics: {csv_path}")

    summary = save_summary_csv(df, split_name, out_dir)

    def fmt(metric, decimals=2):
        if metric not in summary.metric.values: return "N/A"
        mean = summary.loc[summary.metric==metric, "mean"].values[0]
        std  = summary.loc[summary.metric==metric, "std"].values[0]
        if metric in ("ssim", "lpips"):
            return f"{mean:.4f} ± {std:.4f}" if not np.isnan(std) else f"{mean:.4f}"
        return f"{mean:.2f} ± {std:.2f}" if not np.isnan(std) else f"{mean:.2f}"

    print(f"\n{split_name.upper()} AVERAGE:")
    print(f"PSNR : {fmt('psnr')}")
    print(f"SSIM : {fmt('ssim')}")
    print(f"LPIPS: {fmt('lpips')}")
    print(f"ΔE00 : {fmt('deltaE00')}")

# Run VAL & TEST
run_split("val",  val_loader,  val_dataset)
run_split("test", test_loader, test_dataset)
print("\n✓ Evaluation completed.")