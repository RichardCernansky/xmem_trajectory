import os                                   # filesystem helpers
import numpy as np                          # arrays / math
import torch                                # tensors
import matplotlib.pyplot as plt             # plotting


class TrajVisualizer:
    CAM_ORDER_DEFAULT = ("CAM_FRONT", "CAM_FRONT_LEFT", "CAM_FRONT_RIGHT")  # projection camera order

    def __init__(self, save_dir=None, dpi=120, draw_seams=True, cam_order=None):  # ctor
        self.save_dir = save_dir                     # optional directory to save images
        self.dpi = dpi                               # figure DPI
        self.draw_seams = draw_seams                 # draw dashed seam guides and labels
        self.cam_order = cam_order or self.CAM_ORDER_DEFAULT  # order to try cameras
        if save_dir: os.makedirs(save_dir, exist_ok=True)      # ensure output directory exists

    @staticmethod
    def _get_meta_viz_tiled(batch, b):               # fetch viz bundle + meta for sample b
        meta_all = batch.get("meta")                 # meta can be list or dict
        meta_b = meta_all[b] if isinstance(meta_all, list) else meta_all  # select current meta
        vt = meta_b.get("viz_tiled", None)           # tiled bundle with crops/K/T/H/cw
        if vt is None: raise KeyError("meta['viz_tiled'] missing; ensure loader populates it.")  # guard
        # verify required fields exist
        for k in ("rgb_crops", "K_scaled", "T_cam_from_ego", "H", "cw"):
            if k not in vt: raise KeyError(f"meta['viz_tiled'] missing key: {k}")
        # also need overlaps from meta (for trimming)
        if "overlap_lf" not in meta_b or "overlap_fr" not in meta_b:
            raise KeyError("meta missing 'overlap_lf'/'overlap_fr' for trimmed pano.")
        return meta_b, vt

    @staticmethod
    def _to_np(arr):                                 # tensor/array → numpy float32
        return arr.detach().cpu().numpy().astype(np.float32) if isinstance(arr, torch.Tensor) else np.asarray(arr, dtype=np.float32)

    @staticmethod
    def build_trimmed_pano(rgb_crops, cw, H, ov_lf, ov_fr):  # build background pano with trimming
        imL, imF, imR = [np.asarray(im) for im in rgb_crops]     # unpack L/F/R crops (H, cw, 3)
        F_keep = imF[:, ov_lf:, :]                                # trim FRONT: drop left ov_lf px
        R_keep = imR[:, ov_fr:, :]                                # trim RIGHT: drop left ov_fr px
        pano = np.concatenate([imL, F_keep, R_keep], axis=1)      # final pano: [LEFT | trimmed FRONT | trimmed RIGHT]
        if pano.dtype != np.uint8: pano = np.clip(pano, 0, 255).astype(np.uint8)  # ensure uint8 for display
        # x-offsets where each camera’s kept region starts in the pano
        x_off = {
            "CAM_FRONT_LEFT": 0,                                   # LEFT starts at 0
            "CAM_FRONT":      cw - ov_lf,                          # FRONT starts after trimming
            "CAM_FRONT_RIGHT":(cw - ov_lf) + (cw - ov_fr)          # RIGHT starts after both trims
        }
        # minimum u in the crop that is still kept (for visibility test)
        min_u = {
            "CAM_FRONT_LEFT": 0.0,                                 # keep all LEFT pixels
            "CAM_FRONT":      float(ov_lf),                        # keep FRONT u >= ov_lf
            "CAM_FRONT_RIGHT":float(ov_fr)                         # keep RIGHT u >= ov_fr
        }
        # seam lines (for visualization): L|F and F|R boundaries in pano coords
        seams = [x_off["CAM_FRONT"], x_off["CAM_FRONT_RIGHT"]]     # [cw - ov_lf, (cw - ov_lf)+(cw - ov_fr)]
        return pano, x_off, min_u, seams

    @staticmethod
    def project_xy_to_trimmed_pano(traj_xy_m, K_scaled, T_cam_from_ego, H, cw, cam_order, x_off, min_u):  # ego→pano pixels
        T = traj_xy_m.shape[0]                                      # number of points
        uv = np.full((T, 2), np.nan, dtype=np.float32)              # initialize as invisible
        for i in range(T):                                          # per-time-step
            X_ego = np.array([traj_xy_m[i, 0], traj_xy_m[i, 1], 0.0, 1.0], dtype=np.float32)  # homogeneous ego point
            for cam in cam_order:                                   # try cameras in priority order
                Tc = T_cam_from_ego[cam]                            # 4x4 cam←ego
                Xc = (Tc @ X_ego.reshape(4, 1)).squeeze()           # → camera coords [X,Y,Z,1]
                Z = float(Xc[2])                                    # depth along +Z
                if Z <= 1e-6: continue                              # behind camera
                K = K_scaled[cam]                                   # 3x3 intrinsics for (H, cw)
                u = (K[0, 0] * (Xc[0] / Z)) + K[0, 2]               # fx' * X/Z + cx'
                v = (K[1, 1] * (Xc[1] / Z)) + K[1, 2]               # fy' * Y/Z + cy'
                if (min_u[cam] <= u < cw) and (0.0 <= v < H):       # inside the kept band & image
                    uv[i] = [u + x_off[cam], v]                     # shift to pano coordinates
                    break                                            # accept first visible camera
        return uv                                                    # (T,2), NaNs where invisible

    def render(self, batch, pred_abs_k, mode_probs=None, sample_idx=0, title="Trimmed pano | all modes",
               highlight="prob", draw_gt=True, fname=None, legend_loc="lower right"):  # main draw
        b = int(sample_idx)                                         # choose item in batch
        meta_b, vt = self._get_meta_viz_tiled(batch, b)             # get meta + viz bundle
        rgb_crops = vt["rgb_crops"]                                 # [L, F, R] crops
        K_scaled = vt["K_scaled"]                                   # per-cam intrinsics (H, cw)
        T_cam_from_ego = vt["T_cam_from_ego"]                       # per-cam extrinsics at frame time
        H = int(vt["H"]); cw = int(vt["cw"])                        # crop size
        ov_lf = int(meta_b["overlap_lf"])                           # L–F overlap in pixels
        ov_fr = int(meta_b["overlap_fr"])                           # F–R overlap in pixels

        pano, x_off, min_u, seams = self.build_trimmed_pano(rgb_crops, cw, H, ov_lf, ov_fr)  # background + mapping
        scene = meta_b.get("scene_name", f"b{b}")                   # title suffix

        gt_np = None                                                # default: no GT
        if draw_gt and ("traj" in batch) and (batch["traj"] is not None):  # show GT if present
            gt_np = self._to_np(batch["traj"][b])                   # (T,2) numpy array

        best_idx = None                                             # highlighted mode index
        if highlight == "prob" and (mode_probs is not None):        # highlight most probable
            best_idx = int(torch.argmax(mode_probs[b]).item())      # argmax over K
        elif highlight == "fde" and gt_np is not None:              # highlight best FDE vs GT
            gt_t = torch.as_tensor(gt_np, device=pred_abs_k.device) # GT tensor for distance calc
            d = torch.linalg.norm(pred_abs_k[b] - gt_t.unsqueeze(0), dim=-1)  # (K,T) distances
            best_idx = int(torch.argmin(d[:, -1]).item())           # smallest final-step error

        fig = plt.figure(figsize=(12, 5), dpi=self.dpi)             # create figure
        ax = fig.add_subplot(111)                                   # single axes
        ax.imshow(pano)                                             # show trimmed pano
        ax.set_axis_off()                                           # hide axes
        ax.set_title(f"{title} • {scene}")                          # title

        if self.draw_seams:                                         # draw seam guides and labels
            for sx in seams: ax.vlines(sx, 0, H - 1, linestyles="--", linewidth=1)  # seam lines
            x1, x2 = seams                                          # L|F and F|R seam x-positions
            ax.text(max(2, x1 // 2), 12, "LEFT",  fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.4, pad=2))     # LEFT label
            ax.text(x1 + max(2, (x2 - x1) // 2), 12, "FRONT", fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.4, pad=2))  # FRONT label
            ax.text(x2 + max(2, (pano.shape[1] - x2) // 2), 12, "RIGHT", fontsize=8, color="white", bbox=dict(facecolor="black", alpha=0.4, pad=2))  # RIGHT label

        if gt_np is not None:                                       # draw GT polyline + endpoints
            uv_gt = self.project_xy_to_trimmed_pano(gt_np, K_scaled, T_cam_from_ego, H, cw, self.cam_order, x_off, min_u)  # project GT
            ax.plot(uv_gt[:, 0], uv_gt[:, 1], linewidth=3.0, color="black", alpha=0.9, label="GT")  # GT path
            vis_gt = ~np.isnan(uv_gt).any(axis=1)                   # visible points mask
            if np.any(vis_gt):                                      # endpoints if any visible
                first_gt = np.argmax(vis_gt)                        # first visible idx
                last_gt  = len(vis_gt) - 1 - np.argmax(vis_gt[::-1])  # last visible idx
                ax.scatter(uv_gt[first_gt, 0], uv_gt[first_gt, 1], s=40, c="black", edgecolors="white", zorder=5)  # start
                ax.scatter(uv_gt[last_gt, 0],  uv_gt[last_gt, 1],  s=60, c="black", edgecolors="white", zorder=5)  # end

        Km = pred_abs_k.shape[1]                                    # number of modes
        cmap = plt.cm.get_cmap("tab20", max(Km, 3))                 # palette
        probs_b = mode_probs[b].detach().cpu().numpy() if mode_probs is not None else None  # per-mode probs or None

        for k in range(Km):                                         # draw each mode
            traj_np = self._to_np(pred_abs_k[b, k])                 # (T,2) numpy
            uv_k = self.project_xy_to_trimmed_pano(traj_np, K_scaled, T_cam_from_ego, H, cw, self.cam_order, x_off, min_u)  # project
            label = f"mode {k}" + (f" (p={probs_b[k]:.2f})" if probs_b is not None else "")  # legend text
            is_best = (best_idx is not None and k == best_idx)      # highlight flag
            lw = 2.8 if is_best else 2.0                            # thicker line for highlighted
            z  = 3 if is_best else 2                                # bring to front if highlighted
            ax.plot(uv_k[:, 0], uv_k[:, 1], linewidth=lw, color=cmap(k), alpha=0.95, label=label, zorder=z)  # draw polyline
            vis_k = ~np.isnan(uv_k).any(axis=1)                     # visible points
            if np.any(vis_k):                                       # draw endpoints
                first_k = np.argmax(vis_k)                          # first visible
                last_k  = len(vis_k) - 1 - np.argmax(vis_k[::-1])   # last visible
                ax.scatter(uv_k[first_k, 0], uv_k[first_k, 1], s=24, color=cmap(k), edgecolors="black", zorder=z + 1)  # start marker
                ax.scatter(uv_k[last_k, 0],  uv_k[last_k, 1],  s=36, color=cmap(k), edgecolors="white", zorder=z + 1)  # end marker

        ax.legend(loc=legend_loc, fontsize=8, framealpha=0.85) if legend_loc else None  # legend if requested
        fig.tight_layout()                                          # compact layout

        if self.save_dir or fname:                                  # save-to-disk
            if fname is None: fname = f"{scene}_trimmed_all_modes_b{b}.png"  # default filename
            out = fname if self.save_dir is None else os.path.join(self.save_dir, fname)  # resolve output path
            fig.savefig(out)                                        # write image
            plt.close(fig)                                          # free figure memory
            return out                                              # return saved path
        return fig                                                  # otherwise return Figure
