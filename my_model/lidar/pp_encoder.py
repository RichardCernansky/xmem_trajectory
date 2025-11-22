"""
PointPillars fork from SECOND.
Code written by Alex Lang and Oscar Beijbom, 2018.
Licensed under MIT License [see LICENSE].
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from mmcv.cnn import build_norm_layer
from torch import nn
from torch.nn import functional as F
from mmengine import Registry
from mmcv.ops import Voxelization
import numpy as np


__all__ = ["PillarFeatureNet", "PointPillarsScatter", "PointPillarsEncoder"]

LIDAR_ENCODERS = Registry('lidar_encoder')

def build_lidar_encoder(cfg: Dict[str, Any]):
    return LIDAR_ENCODERS.build(cfg)


def get_paddings_indicator(actual_num: torch.Tensor, max_num: int, axis: int = 0) -> torch.Tensor:
    """Create boolean mask by actually number of a padded tensor.
    Args:
        actual_num ([type]): [description]
        max_num ([type]): [description]
    Returns:
        [type]: [description]
    """

    actual_num = torch.unsqueeze(actual_num, axis + 1)
    # tiled_actual_num: [N, M, 1]
    max_num_shape: list[int] = [1] * len(actual_num.shape)
    max_num_shape[axis + 1] = -1
    max_num = torch.arange(max_num, dtype=torch.int, device=actual_num.device).view(
        max_num_shape
    )

    # tiled_actual_num: [[3,3,3,3,3], [4,4,4,4,4], [2,2,2,2,2]]
    # tiled_max_num: [[0,1,2,3,4], [0,1,2,3,4], [0,1,2,3,4]]
    paddings_indicator = actual_num.int() > max_num
    # paddings_indicator shape: [batch_size, max_num]
    return paddings_indicator


class PFNLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, norm_cfg: Optional[Dict[str, Any]] = None, last_layer: bool = False):
        """
        Pillar Feature Net Layer.
        The Pillar Feature Net could be composed of a series of these layers, but the PointPillars paper results only
        used a single PFNLayer. This layer performs a similar role as second.pytorch.voxelnet.VFELayer.
        :param in_channels: <int>. Number of input channels.
        :param out_channels: <int>. Number of output channels.
        :param last_layer: <bool>. If last_layer, there is no concatenation of features.
        """

        super().__init__()
        self.name = "PFNLayer"
        self.last_vfe = last_layer
        if not self.last_vfe:
            out_channels = out_channels // 2
        self.units = out_channels

        if norm_cfg is None:
            norm_cfg = dict(type="BN1d", eps=1e-3, momentum=0.01)
        self.norm_cfg = norm_cfg

        self.linear = nn.Linear(in_channels, self.units, bias=False)
        self.norm = build_norm_layer(self.norm_cfg, self.units)[1]

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:

        x = self.linear(inputs)
        torch.backends.cudnn.enabled = False
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        torch.backends.cudnn.enabled = True
        x = F.relu(x)

        x_max = torch.max(x, dim=1, keepdim=True)[0]

        if self.last_vfe:
            return x_max
        else:
            x_repeat = x_max.repeat(1, inputs.shape[1], 1)
            x_concatenated = torch.cat([x, x_repeat], dim=2)
            return x_concatenated


@LIDAR_ENCODERS.register_module()
class PillarFeatureNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 5,
        feat_channels: Tuple[int, ...] = (64,),
        with_distance: bool = False,
        voxel_size: Tuple[float, float, float] = (0.2, 0.2, 4),
        point_cloud_range: Tuple[float, float, float, float, float, float] = (0, -40, -3, 70.4, 40, 1),
        norm_cfg: Optional[Dict[str, Any]] = None,
    ):
        """
        Pillar Feature Net.
        The network prepares the pillar features and performs forward pass through PFNLayers. This net performs a
        similar role to SECOND's second.pytorch.voxelnet.VoxelFeatureExtractor.
        :param num_input_features: <int>. Number of input features, either x, y, z or x, y, z, r.
        :param num_filters: (<int>: N). Number of features in each of the N PFNLayers.
        :param with_distance: <bool>. Whether to include Euclidean distance to points.
        :param voxel_size: (<float>: 3). Size of voxels, only utilize x and y size.
        :param pc_range: (<float>: 6). Point cloud range, only utilize x and y min.
        """

        super().__init__()
        self.name = "PillarFeatureNet"
        assert len(feat_channels) > 0

        self.in_channels = in_channels
        # TODO set back to 5
        in_channels += 5
        if with_distance:
            in_channels += 1
        self._with_distance = with_distance

        # Create PillarFeatureNet layers
        feat_channels = [in_channels] + list(feat_channels)
        pfn_layers = []
        for i in range(len(feat_channels) - 1):
            in_filters = feat_channels[i]
            out_filters = feat_channels[i + 1]
            if i < len(feat_channels) - 2:
                last_layer = False
            else:
                last_layer = True
            pfn_layers.append(
                PFNLayer(
                    in_filters, out_filters, norm_cfg=norm_cfg, last_layer=last_layer
                )
            )
        self.pfn_layers = nn.ModuleList(pfn_layers)

        # Need pillar (voxel) size and x/y offset in order to calculate pillar offset
        self.vx = voxel_size[0]
        self.vy = voxel_size[1]
        self.x_offset = self.vx / 2 + point_cloud_range[0]
        self.y_offset = self.vy / 2 + point_cloud_range[1]

    def forward(self, features: torch.Tensor, num_voxels: torch.Tensor, coors: torch.Tensor) -> torch.Tensor:
        device = features.device

        dtype = features.dtype

        # Find distance of x, y, and z from cluster center
        # features = features[:, :, :self.num_input]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        f_cluster = features[:, :, :3] - points_mean

        # Find distance of x, y, and z from pillar center
        # f_center = features[:, :, :2]
        # modified according to xyz coords
        f_center = torch.zeros_like(features[:, :, :2])
        f_center[:, :, 0] = features[:, :, 0] - (coors[:, 3].to(dtype).unsqueeze(1) * self.vx + self.x_offset)
        f_center[:, :, 1] = features[:, :, 1] - (coors[:, 2].to(dtype).unsqueeze(1) * self.vy + self.y_offset)

        # Combine together feature decorations
        features_ls = [features, f_cluster, f_center]
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features_ls.append(points_dist)
        features = torch.cat(features_ls, dim=-1)

        # The feature decorations were calculated without regard to whether pillar was empty. Need to ensure that
        # empty pillars remain set to zeros.
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        features *= mask

        # Forward pass through PFNLayers
        for pfn in self.pfn_layers:
            features = pfn(features)

        return features.squeeze()


@LIDAR_ENCODERS.register_module()
class PointPillarsScatter(nn.Module):
    def __init__(self, in_channels: int = 64, output_shape: Tuple[int, int] = (512, 512), **kwargs):
        """
        Point Pillar's Scatter.
        Converts learned features from dense tensor to sparse pseudo image. This replaces SECOND's
        second.pytorch.voxelnet.SparseMiddleExtractor.
        :param output_shape: ([int]: 4). Required output shape of features.
        :param num_input_features: <int>. Number of input features.
        """

        super().__init__()
        self.in_channels = in_channels
        self.output_shape = output_shape
        self.ny = output_shape[0]
        self.nx = output_shape[1]

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, output_shape={tuple(self.output_shape)}"
        )

    def forward(self, voxel_features: torch.Tensor, coords: torch.Tensor, batch_size: int) -> torch.Tensor:
        # batch_canvas will be the final output.
        batch_canvas = []
        for batch_itt in range(batch_size):
            # Create the canvas for this sample
            canvas = torch.zeros(
                self.in_channels,
                self.nx * self.ny,
                dtype=voxel_features.dtype,
                device=voxel_features.device,
            )

            # Only include non-empty pillars
            batch_mask = coords[:, 0] == batch_itt

            this_coords = coords[batch_mask, :]
            # modified -> xyz coords
            # indices = this_coords[:, 1] * self.ny + this_coords[:, 2]
            indices = this_coords[:, 2] * self.nx + this_coords[:, 3]
            indices = indices.type(torch.long)
            voxels = voxel_features[batch_mask, :]
            voxels = voxels.t()

            # Now scatter the blob back to the canvas.
            canvas[:, indices] = voxels

            # Append to a list for later stacking.
            batch_canvas.append(canvas)

        # Stack to 3-dim tensor (batch-size, nchannels, nrows*ncols)
        batch_canvas = torch.stack(batch_canvas, 0)

        # Undo the column stacking to final 4-dim tensor
        # batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.nx, self.ny)
        batch_canvas = batch_canvas.view(batch_size, self.in_channels, self.ny, self.nx)
        return batch_canvas


class PointPillarsHead(nn.Module):
    def __init__(self, in_channel: int, num_anchors: int, num_classes: int) -> None:
        super().__init__()

        self.conv_cls = nn.Conv2d(in_channel, num_anchors * num_classes, 1)
        self.conv_reg = nn.Conv2d(in_channel, num_anchors * 7, 1)
        self.conv_dir_cls = nn.Conv2d(in_channel, num_anchors * 2, 1)

        # consistent with mmdet3d bias init
        conv_layer_id = 0
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                if conv_layer_id == 0:
                    prior_prob = 0.01
                    bias_init = float(-np.log((1 - prior_prob) / prior_prob))
                    nn.init.constant_(m.bias, bias_init)
                else:
                    nn.init.constant_(m.bias, 0.0)
                conv_layer_id += 1

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (bs, C, H, W)

        Returns:
            bbox_cls_pred: (bs, A*num_classes, H, W)
            bbox_pred: (bs, A*7, H, W)
            bbox_dir_cls_pred: (bs, A*2, H, W)
        """
        bbox_cls_pred = self.conv_cls(x)
        bbox_pred = self.conv_reg(x)
        bbox_dir_cls_pred = self.conv_dir_cls(x)
        return bbox_cls_pred, bbox_pred, bbox_dir_cls_pred


@LIDAR_ENCODERS.register_module()
class PointPillarsEncoder(nn.Module):
    def __init__(
        self,
        pts_voxel_encoder: Dict[str, Any],
        pts_middle_encoder: Dict[str, Any],
        pts_backbone: Optional[Dict[str, Any]] = None,
        pts_neck: Optional[Dict[str, Any]] = None,
        pp_head: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__()
        # derive effective voxel size to match scatter output shape
        pc_range: List[float] = list(pts_voxel_encoder.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]))
        base_voxel_size: List[float] = list(pts_voxel_encoder.get('voxel_size', [0.2, 0.2, 4.0]))
        out_shape: List[int] = list(pts_middle_encoder.get('output_shape', [256, 256]))

        span_x: float = pc_range[3] - pc_range[0]
        span_y: float = pc_range[4] - pc_range[1]
        grid_x: int = int(round(span_x / base_voxel_size[0]))
        grid_y: int = int(round(span_y / base_voxel_size[1]))
        # scale voxel size to meet desired output shape if divisible
        scale_x: int = max(1, grid_x // int(out_shape[0]))
        scale_y: int = max(1, grid_y // int(out_shape[1]))
        eff_voxel_size: List[float] = [base_voxel_size[0] * scale_x, base_voxel_size[1] * scale_y, base_voxel_size[2]]

        self.voxelize_module = Voxelization(
            max_num_points=kwargs.get('max_num_points', 10),
            point_cloud_range=pc_range,
            voxel_size=eff_voxel_size,
            max_voxels=kwargs.get('max_voxels', [90000, 120000])
        )
        self.voxelize_reduce: bool = False

        # build PFN and scatter with effective voxel size to keep geometric consistency
        pts_voxel_encoder_cfg = dict(pts_voxel_encoder)
        pts_voxel_encoder_cfg['voxel_size'] = eff_voxel_size
        pts_voxel_encoder_cfg['point_cloud_range'] = pc_range
        self.pts_voxel_encoder = build_lidar_encoder(pts_voxel_encoder_cfg)
        self.pts_middle_encoder = build_lidar_encoder(pts_middle_encoder)

        # optional backbone and neck (PointPillars-style)
        self.pts_backbone = build_lidar_encoder(pts_backbone) if pts_backbone is not None else None
        self.pts_neck = build_lidar_encoder(pts_neck) if pts_neck is not None else None

        # optional PointPillars detection head
        self._pp_head_cfg: Optional[Dict[str, Any]] = dict(pp_head) if pp_head is not None else None
        self._pp_head: Optional[PointPillarsHead] = None
        self._pp_preds: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = None

    @torch.no_grad()
    def voxelize(self, points: Union[List[torch.Tensor], torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feats: list[torch.Tensor] = []
        coords: list[torch.Tensor] = []
        sizes: list[torch.Tensor] = []

        # normalize input to list[Tensor[N, C]] per batch
        if isinstance(points, torch.Tensor):
            if points.dim() == 4:
                # [B, T, N, C] -> take last frame by default
                points_list: List[torch.Tensor] = [points[b, -1] for b in range(points.size(0))]
            elif points.dim() == 3:
                # [B, N, C]
                points_list = [points[b] for b in range(points.size(0))]
            else:
                # [N, C]
                points_list = [points]
        else:
            points_list = points

        for k, res in enumerate(points_list):
            ret = self.voxelize_module(res)
            if len(ret) == 3:
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats_out = torch.cat(feats, dim=0) if len(feats) > 0 else torch.empty(0)
        coords_out = torch.cat(coords, dim=0) if len(coords) > 0 else torch.empty(0)
        sizes_out = torch.cat(sizes, dim=0) if len(sizes) > 0 else torch.empty(0)

        if sizes and self.voxelize_reduce:
            feats_out = feats_out.sum(dim=1, keepdim=False) / sizes_out.type_as(feats_out).view(-1, 1)
            feats_out = feats_out.contiguous()

        return feats_out, coords_out, sizes_out

    def forward(self, points: Union[List[torch.Tensor], torch.Tensor]) -> torch.Tensor:
        batch_size: int
        if isinstance(points, torch.Tensor):
            if points.dim() == 4:
                batch_size = points.size(0)
            elif points.dim() == 3:
                batch_size = points.size(0)
            else:
                batch_size = 1
        else:
            batch_size = len(points)

        feats, coords, sizes = self.voxelize(points)
        x = self.pts_voxel_encoder(feats, sizes, coords)
        x = self.pts_middle_encoder(x, coords, batch_size)

        if self.pts_backbone is not None:
            xs = self.pts_backbone(x)
            if self.pts_neck is not None:
                x = self.pts_neck(xs)
            else:
                # if no neck, use the last stage
                x = xs[-1] if isinstance(xs, list) else xs

        # lazily build and run PointPillars head if configured
        if self._pp_head_cfg is not None:
            if self._pp_head is None:
                in_channel: int = int(x.size(1))
                num_anchors: int = int(self._pp_head_cfg.get('n_anchors', self._pp_head_cfg.get('num_anchors', 2)))
                num_classes: int = int(self._pp_head_cfg.get('n_classes', self._pp_head_cfg.get('num_classes', 1)))
                self._pp_head = PointPillarsHead(in_channel=in_channel, num_anchors=num_anchors, num_classes=num_classes).to(x.device)
            cls_pred, box_pred, dir_pred = self._pp_head(x)
            self._pp_preds = (cls_pred, box_pred, dir_pred)
        return x

    def get_pointpillars_head_outputs(self) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Return latest PointPillars head outputs if head is configured and forward was run."""
        return self._pp_preds


@LIDAR_ENCODERS.register_module()
class PPBackbone(nn.Module):
    def __init__(self, in_channels: int = 64, out_channels: Tuple[int, int, int] = (64, 128, 256), layer_nums: Tuple[int, int, int] = (3, 5, 5), layer_strides: Tuple[int, int, int] = (2, 2, 2)):
        super().__init__()
        assert len(out_channels) == len(layer_nums) == len(layer_strides)
        self.blocks = nn.ModuleList()
        c_in = in_channels
        for i in range(len(layer_strides)):
            stride = layer_strides[i]
            c_out = out_channels[i]
            layers = [
                nn.Conv2d(c_in, c_out, kernel_size=3, stride=stride, padding=1, bias=False),
                build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                nn.ReLU(inplace=True),
            ]
            for _ in range(layer_nums[i]):
                layers += [
                    nn.Conv2d(c_out, c_out, kernel_size=3, stride=1, padding=1, bias=False),
                    build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                    nn.ReLU(inplace=True),
                ]
            self.blocks.append(nn.Sequential(*layers))
            c_in = c_out

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats: List[torch.Tensor] = []
        for blk in self.blocks:
            x = blk(x)
            feats.append(x)
        return feats


@LIDAR_ENCODERS.register_module()
class PPNeck(nn.Module):
    def __init__(self, in_channels: Tuple[int, int, int] = (64, 128, 256), upsample_strides: Tuple[int, int, int] = (1, 2, 4), out_channels: Tuple[int, int, int] = (64, 64, 64)):
        super().__init__()
        assert len(in_channels) == len(upsample_strides) == len(out_channels)
        deblocks: List[nn.Module] = []
        for c_in, stride, c_out in zip(in_channels, upsample_strides, out_channels):
            deblock = nn.Sequential(
                nn.ConvTranspose2d(c_in, c_out, kernel_size=stride, stride=stride, bias=False),
                build_norm_layer(dict(type="BN2d", eps=1e-3, momentum=0.01), c_out)[1],
                nn.ReLU(inplace=True),
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)
        self.out_channels = sum(out_channels)
        # optional projection to a target dim if specified at runtime
        self.proj: Optional[nn.Conv2d] = None

    def forward(self, xs: List[torch.Tensor], out_channels_proj: Optional[int] = None) -> torch.Tensor:
        assert len(xs) == len(self.deblocks)
        ups: List[torch.Tensor] = []
        for x, deblock in zip(xs, self.deblocks):
            ups.append(deblock(x))
        x = torch.cat(ups, dim=1)
        if out_channels_proj is not None:
            if self.proj is None or self.proj.out_channels != out_channels_proj:
                self.proj = nn.Conv2d(self.out_channels, out_channels_proj, kernel_size=1, bias=False)
            x = self.proj(x)
        return x


