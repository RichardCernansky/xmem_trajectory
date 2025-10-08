import os                                   # filesystem helpers
import numpy as np                          # arrays / math
import torch                                # tensors
import matplotlib.pyplot as plt             # plotting

import math

class TrajVisualizer:
    CAM_ORDER_DEFAULT = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")

    def __init__(self, save_dir=None, dpi=120, draw_seams=True, cam_order=None):
        self.save_dir = save_dir
        self.dpi = dpi
        self.draw_seams = draw_seams
        self.cam_order = cam_order or self.CAM_ORDER_DEFAULT
        if save_dir: os.makedirs(save_dir, exist_ok=True)

    @staticmethod
    def _get_meta_viz_tiled(batch, b):
        meta_all = batch.get("meta")
        meta_b = meta_all[b] if isinstance(meta_all, list) else meta_all
        vt = meta_b.get("viz_tiled", None)
        if vt is None: raise KeyError("meta['viz_tiled'] missing; ensure loader populates it.")
        for k in ("rgb_crops", "K_scaled", "T_cam_from_ego", "H", "cw"):
            if k not in vt: raise KeyError(f"meta['viz_tiled'] missing key: {k}")
        if "overlap_lf" not in meta_b or "overlap_fr" not in meta_b:
            raise KeyError("meta missing 'overlap_lf'/'overlap_fr' for trimmed pano.")
        return meta_b, vt

    @staticmethod
    def _to_img(crop):
        if isinstance(crop, torch.Tensor):
            crop = crop.detach().cpu()
            if crop.ndim == 3 and crop.shape[0] in (1, 3):  # CHW -> HWC
                crop = crop.permute(1, 2, 0)
            crop = crop.numpy()
        im = np.asarray(crop).astype(np.float32)
        if im.max() <= 5.0 and im.min() >= -5.0:  # likely normalized
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            im = im * std + mean
        if im.max() <= 1.0 + 1e-6:
            im = im * 255.0
        return np.clip(im, 0, 255).astype(np.uint8)

    @staticmethod
    def _to_np(arr):
        return arr.detach().cpu().numpy().astype(np.float32) if isinstance(arr, torch.Tensor) else np.asarray(arr, dtype=np.float32)

    def build_trimmed_pano(self, rgb_crops, cw, H, ov_lf, ov_fr):
        imL, imF, imR = [self._to_img(im) for im in rgb_crops]
        F_keep = imF[:, ov_lf:, :]
        R_keep = imR[:, ov_fr:, :]
        pano = np.concatenate([imL, F_keep, R_keep], axis=1)
        if pano.dtype != np.uint8: pano = np.clip(pano, 0, 255).astype(np.uint8)
        x_off = {
            "CAM_FRONT_LEFT": 0,
            "CAM_FRONT":      cw - ov_lf,
            "CAM_FRONT_RIGHT":(cw - ov_lf) + (cw - ov_fr)
        }
        min_u = {
            "CAM_FRONT_LEFT": 0.0,
            "CAM_FRONT":      float(ov_lf),
            "CAM_FRONT_RIGHT":float(ov_fr)
        }
        seams = [x_off["CAM_FRONT"], x_off["CAM_FRONT_RIGHT"]]
        return pano, x_off, min_u, seams

    @staticmethod
    def project_xy_to_trimmed_pano(traj_xy_m, K_scaled, T_cam_from_ego, H, cw, cam_order, x_off, min_u):
        T = traj_xy_m.shape[0]
        uv = np.full((T, 2), np.nan, dtype=np.float32)
        for i in range(T):
            X_ego = np.array([traj_xy_m[i, 0], traj_xy_m[i, 1], 0.0, 1.0], dtype=np.float32)
            for cam in cam_order:
                Tc = T_cam_from_ego[cam]
                Xc = (Tc @ X_ego.reshape(4, 1)).squeeze()
                Z = float(Xc[2])
                if Z <= 1e-6: continue
                K = K_scaled[cam]
                u = (K[0, 0] * (Xc[0] / Z)) + K[0, 2]
                v = (K[1, 1] * (Xc[1] / Z)) + K[1, 2]
                if (min_u[cam] <= u < cw) and (0.0 <= v < H):
                    uv[i] = [u + x_off[cam], v]
                    break
        return uv

    # ---------- NEW: batch mosaic ----------
    def render_batch(self, batch, epoch, pred_abs_k, mode_probs=None, title="Trimmed pano | batch", 
                     highlight="prob", draw_gt=True, ncols=4, fname=None, legend=False):
        """
        Draw all B samples in a grid, preserving native pano resolution per tile.
        """
        B = pred_abs_k.shape[0]
        # Probe size from item 0 to size the canvas (assume fixed crop size within a batch)
        meta0, vt0 = self._get_meta_viz_tiled(batch, 0)
        H  = int(vt0["H"]); cw = int(vt0["cw"])
        ov_lf0 = int(meta0["overlap_lf"]); ov_fr0 = int(meta0["overlap_fr"])
        pano_w = cw + (cw - ov_lf0) + (cw - ov_fr0)  # LEFT + kept FRONT + kept RIGHT

        ncols = max(1, int(ncols))
        nrows = int(math.ceil(B / ncols))

        figsize = ( (pano_w * ncols) / self.dpi, (H * nrows) / self.dpi )
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=self.dpi)
        axes = np.atleast_1d(axes).ravel()

        for b in range(B):
            ax = axes[b]
            meta_b, vt = self._get_meta_viz_tiled(batch, b)
            rgb_crops = vt["rgb_crops"]
            K_scaled = vt["K_scaled"]
            T_cam_from_ego = vt["T_cam_from_ego"]
            H_b = int(vt["H"]); cw_b = int(vt["cw"])
            ov_lf = int(meta_b["overlap_lf"]); ov_fr = int(meta_b["overlap_fr"])
            pano, x_off, min_u, seams = self.build_trimmed_pano(rgb_crops, cw_b, H_b, ov_lf, ov_fr)

            scene = meta_b.get("scene_name", f"b{b}")
            ax.imshow(pano); ax.set_axis_off()
            ax.set_title(scene, fontsize=10)

            if self.draw_seams:
                for sx in seams: ax.vlines(sx, 0, H_b - 1, linestyles="--", linewidth=1)
                x1, x2 = seams
                ax.text(max(2, x1 // 2), 12, "LEFT",  fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1))
                ax.text(x1 + max(2, (x2 - x1) // 2), 12, "FRONT", fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1))
                ax.text(x2 + max(2, (pano.shape[1] - x2) // 2), 12, "RIGHT", fontsize=7, color="white",
                        bbox=dict(facecolor="black", alpha=0.4, pad=1))

            gt_np = None
            if draw_gt and ("traj" in batch) and (batch["traj"] is not None):
                gt_np = self._to_np(batch["traj"][b])

            best_idx = None
            if highlight == "prob" and (mode_probs is not None):
                best_idx = int(torch.argmax(mode_probs[b]).item())
            elif highlight == "fde" and gt_np is not None:
                gt_t = torch.as_tensor(gt_np, device=pred_abs_k.device)
                d = torch.linalg.norm(pred_abs_k[b] - gt_t.unsqueeze(0), dim=-1)
                best_idx = int(torch.argmin(d[:, -1]).item())

            if gt_np is not None:
                uv_gt = self.project_xy_to_trimmed_pano(gt_np, K_scaled, T_cam_from_ego, H_b, cw_b, self.cam_order, x_off, min_u)
                ax.plot(uv_gt[:, 0], uv_gt[:, 1], linewidth=2.5, color="black", alpha=0.9, label="GT")
                vis_gt = ~np.isnan(uv_gt).any(axis=1)
                if np.any(vis_gt):
                    first_gt = np.argmax(vis_gt)
                    last_gt  = len(vis_gt) - 1 - np.argmax(vis_gt[::-1])
                    ax.scatter(uv_gt[first_gt, 0], uv_gt[first_gt, 1], s=30, c="black", edgecolors="white", zorder=5)
                    ax.scatter(uv_gt[last_gt, 0],  uv_gt[last_gt, 1],  s=45, c="black", edgecolors="white", zorder=5)

            Km = pred_abs_k.shape[1]
            cmap = plt.cm.get_cmap("tab20", max(Km, 3))
            probs_b = mode_probs[b].detach().cpu().numpy() if mode_probs is not None else None

            for k in range(Km):
                traj_np = self._to_np(pred_abs_k[b, k])
                uv_k = self.project_xy_to_trimmed_pano(traj_np, K_scaled, T_cam_from_ego, H_b, cw_b, self.cam_order, x_off, min_u)
                is_best = (best_idx is not None and k == best_idx)
                lw = 2.4 if is_best else 1.8
                z  = 3 if is_best else 2
                ax.plot(uv_k[:, 0], uv_k[:, 1], linewidth=lw, color=cmap(k), alpha=0.95, zorder=z)
                vis_k = ~np.isnan(uv_k).any(axis=1)
                if np.any(vis_k):
                    first_k = np.argmax(vis_k)
                    last_k  = len(vis_k) - 1 - np.argmax(vis_k[::-1])
                    ax.scatter(uv_k[first_k, 0], uv_k[first_k, 1], s=20, color=cmap(k), edgecolors="black", zorder=z+1)
                    ax.scatter(uv_k[last_k, 0],  uv_k[last_k, 1],  s=30, color=cmap(k), edgecolors="white", zorder=z+1)

            if legend:
                ax.legend(fontsize=7, framealpha=0.85, loc="lower right")

        # turn off extra axes
        for j in range(B, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(title, fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        if self.save_dir or fname:
            if fname is None: fname = "batch_trimmed_all_modes.png"
            out = fname if self.save_dir is None else os.path.join(self.save_dir, fname)
            fig.savefig(out)
            plt.close(fig)
            return out
        return fig
