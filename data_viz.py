# viz_mask0_lidar_rgboverlay.py
import os, sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib

# --- project imports (adjust to your setup) ---
REPO_ROOT = r"C:\Users\Lukas\richard\xmem_e2e\XMem"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from nuscenes.nuscenes import NuScenes
from torch.utils.data import DataLoader
from traj.datamodules import NuScenesSeqLoader, collate_varK

# ----------------- config -----------------
INDEX_PATH = "train_agents_index.pkl"
NUSCENES_DATAROOT = r"e:\nuscenes"
RESIZE = True
RESIZE_WH = (960, 540)     # (W, H)
BATCH_SIZE = 1
NUM_SAMPLES = 10
SHOW = True
SAVE_DIR = r"C:\Users\Lukas\richard\xmem_e2e\viz_out_triptych_gain_thick"
os.makedirs(SAVE_DIR, exist_ok=True)
# ------------------------------------------

# ============================ helpers ============================
def to_np_img(t: torch.Tensor) -> np.ndarray:
    t = t.clamp(0,1).detach().cpu().numpy()
    return (t.transpose(1,2,0) * 255.0).astype(np.uint8)

def norm01(x: np.ndarray, eps=1e-6) -> np.ndarray:
    x = x.astype(np.float32)
    m = np.nanmin(x); M = np.nanmax(x)
    if not np.isfinite(m) or not np.isfinite(M) or M <= m + eps:
        return np.zeros_like(x, dtype=np.float32)
    return (x - m) / (M - m + eps)

def masks_to_color_overlay(base_rgb_u8: np.ndarray, masks_torch: torch.Tensor | None, alpha=0.4) -> np.ndarray:
    H, W, _ = base_rgb_u8.shape
    base = base_rgb_u8.astype(np.float32) / 255.0
    if masks_torch is None or not hasattr(masks_torch, "numel") or masks_torch.numel() == 0:
        return base_rgb_u8
    K = masks_torch.shape[0]
    cmap = matplotlib.colormaps["tab10"].resampled(max(10, K))
    color = np.zeros((H, W, 3), dtype=np.float32)
    m_np = masks_torch.detach().cpu().numpy()
    for k in range(K):
        mk = norm01(m_np[k])
        col = np.array(cmap(k % cmap.N)[:3], dtype=np.float32)
        color += mk[..., None] * col
    out = (1 - alpha) * base + alpha * np.clip(color, 0, 1)
    return (out.clip(0,1) * 255).astype(np.uint8)

def _extract_mask0_t0(masksL):
    """
    Returns HxW float32 mask for the first instance (mask[0]) at t=0.
    Accepts either a Tensor [K,H,W] or [T,K,H,W], or a list/tuple wrapping that tensor.
    """
    if masksL is None:
        return None
    m = masksL[0] if isinstance(masksL, (list, tuple)) and len(masksL) > 0 else masksL
    if not torch.is_tensor(m) or m.numel() == 0:
        return None
    if m.dim() == 4:  # [T,K,H,W]
        m = m[0]      # t=0 -> [K,H,W]
    if m.dim() != 3 or m.shape[0] < 1:
        return None
    mask0 = m[0]      # first instance -> [H,W]
    mask0 = mask0.detach().cpu().float().numpy()
    if mask0.max() > 1.0:
        mask0 = (mask0 > 0.5).astype(np.float32)
    return mask0

# -------- LiDAR mapper with GAIN + THICKENING --------
def lidar_to_rgb_with_gain(channel,
                           occ,
                           cmap_name="turbo",
                           invert=True,
                           q_low=1.0, q_high=99.0,
                           gamma=0.5,
                           gain=2.6,
                           thicken_px=4,
                           thicken_iters=2):
    """
    channel: HxW float
    occ:     HxW {0,1}
    returns: HxW x 3 uint8 RGB
    """
    try:
        import cv2
    except Exception:
        cv2 = None

    x = channel.astype(np.float32)
    x = np.where(occ > 0, x, np.nan)

    if invert:
        x = 1.0 - x

    if np.isfinite(x).any():
        lo = np.nanpercentile(x, q_low)
        hi = np.nanpercentile(x, q_high)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
            lo, hi = np.nanmin(x), np.nanmax(x)
        x = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    else:
        x = np.zeros_like(x, dtype=np.float32)

    x = np.power(x, gamma)
    x = np.clip(x * gain, 0.0, 1.0)

    if cv2 is not None and thicken_px > 0 and thicken_iters > 0:
        k = 2 * thicken_px + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        occ_thick = (occ > 0).astype(np.uint8)
        for _ in range(thicken_iters):
            occ_thick = cv2.dilate(occ_thick, kernel, iterations=1)
        x = np.where(occ_thick > 0, x, 0.0)

    cmap = matplotlib.colormaps.get_cmap(cmap_name)
    rgba = cmap(np.nan_to_num(x, nan=0.0))
    rgb = (rgba[..., :3] * 255.0).astype(np.uint8)
    return rgb
# ---------------------------------------------------------------------------

def visualize_triptych_mask0(batch,
                             nusc,
                             lidar_channel="depth",
                             cmap="turbo",
                             invert=True,
                             q_low=0.5, q_high=99.7,
                             gamma=0.45,
                             gain=2.8,
                             thicken_px=4,
                             thicken_iters=2,
                             title_prefix=""):
    frames     = batch["frames"]
    lidar_maps = batch["lidar_maps"]
    masksL     = batch.get("init_masks", None)
    target_id = batch.get("meta", None)[0]["target"]["agent_id"]
    instance_tok = nusc.get('instance', target_id)
    category = nusc.get('category', instance_tok['category_token'])
    print("CATEGORY:", category)

    b, t = 0, 0
    rgb = to_np_img(frames[b, t])

    # LiDAR prep
    lmap = lidar_maps[b, t].detach().cpu().numpy()
    depth_n, height_n, inten_n, count_n, occ = lmap

    if lidar_channel == "depth":
        chan = depth_n; inv = True if invert is None else invert
    elif lidar_channel == "height":
        chan = height_n; inv = False if invert is None else invert
    elif lidar_channel == "intensity":
        chan = np.clip(inten_n, 0.0, 1.0); inv = False if invert is None else invert
    elif lidar_channel == "count":
        chan = np.clip(count_n, 0.0, 1.0); inv = False if invert is None else invert
    else:
        raise ValueError("lidar_channel must be one of {'depth','height','intensity','count'}")

    lidar_rgb = lidar_to_rgb_with_gain(
        chan, occ,
        cmap_name=cmap,
        invert=inv,
        q_low=q_low, q_high=q_high,
        gamma=gamma,
        gain=gain,
        thicken_px=thicken_px,
        thicken_iters=thicken_iters
    )

    # mask[0] @ t=0
    mask0 = _extract_mask0_t0(masksL)  # HxW float32 in [0,1]
    if mask0 is None:
        return None

    # RGB + mask[0] overlay (only the first instance)
    mask0_torch = torch.from_numpy(mask0[None, ...])  # [1,H,W]
    rgb_overlay = masks_to_color_overlay(rgb, mask0_torch, alpha=0.6)

    # ---- Plot: [mask0, LiDAR, RGB+overlay] ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), squeeze=False)

    axes[0, 0].imshow(mask0, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    axes[0, 0].set_title("mask[0] @ t=0")

    axes[0, 1].imshow(lidar_rgb)
    axes[0, 1].set_title(f"LiDAR ({lidar_channel})")

    axes[0, 2].imshow(rgb_overlay)
    axes[0, 2].set_title("RGB + mask[0] overlay")

    for ax in axes[0]:
        ax.axis("off")

    cov = float(occ.sum()) / (occ.shape[0] * occ.shape[1])
    fig.suptitle(f"{title_prefix}  LiDAR cov={cov:.2%}  |  channel={lidar_channel}", y=0.98)
    plt.tight_layout()
    return fig

def main():
    print("Loading NuScenes + dataset...")
    nusc = NuScenes(version="v1.0-trainval", dataroot=NUSCENES_DATAROOT, verbose=False)

    ds = NuScenesSeqLoader(
        index_path=INDEX_PATH,
        resize=RESIZE,
        resize_wh=RESIZE_WH,
        nusc=nusc,
        classes_prefix=("vehicle.car", "vehicle.truck", "vehicle.bus"),
        max_K=4,
    )
    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        pin_memory=True, collate_fn=collate_varK
    )

    it = iter(loader)
    for i in range(NUM_SAMPLES):
        try:
            batch = next(it)
        except StopIteration:
            break

        fig = visualize_triptych_mask0(

            batch,
            nusc,
            lidar_channel="depth",
            cmap="turbo",
            invert=True,
            q_low=0.5, q_high=99.7,
            gamma=0.45,
            gain=3,
            thicken_px=6,
            thicken_iters=3,
            title_prefix=f"Seq #{i}"
        
        )
        if fig is None:
            print(f"Seq #{i}: no mask[0] available")
            continue

        out_path = os.path.join(SAVE_DIR, f"seq_{i:03d}_mask0_lidar_rgboverlay.png")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=150)
        if SHOW:
            plt.show()
        plt.close(fig)
        print(f"Saved {out_path}")

if __name__ == "__main__":
    main()