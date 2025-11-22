
from nuscenes.nuscenes import NuScenes
from datamodule.mask_utils import bev_box_mask_from_ann
from typing import List, Dict, Optional, Tuple

class MockLoader:

    """Minimal loader interface for mask generation during filtering."""
    def __init__(self, nusc, bev_config):
        self.nusc = nusc
        self.bev_x_min = float(bev_config["x_min"])
        self.bev_x_max = float(bev_config["x_max"])
        self.bev_y_min = float(bev_config["y_min"])
        self.bev_y_max = float(bev_config["y_max"])
        self.H_bev = int(bev_config["H_bev"])
        self.W_bev = int(bev_config["W_bev"])
        self.res_x = float(bev_config["res_x"])
        self.res_y = float(bev_config["res_y"])


def _get_mask_pixel_count(
    nusc: NuScenes,
    sample_token: str,
    inst_tok: str,
    lidar_sd_token: str,
    mock_loader: MockLoader
) -> int:
    """
    Generate actual mask and count pixels.
    
    Returns:
        num_pixels (int): Number of pixels in the mask (0 if not visible)
    """
    # Get annotation
    s = nusc.get("sample", sample_token)
    ann = None
    for ann_tok in s["anns"]:
        a = nusc.get("sample_annotation", ann_tok)
        if a["instance_token"] == inst_tok:
            ann = a
            break
    
    if ann is None:
        return 0
    
    # Generate mask using your existing function
    mask = bev_box_mask_from_ann(mock_loader, ann, lidar_sd_token)
    
    # Count non-zero pixels
    num_pixels = int(mask.sum())
    
    return num_pixels


def _check_sequence_mask_visibility(
    nusc: NuScenes,
    obs_tokens: List[str],
    lidar_keyframe_tokens: List[str],
    inst_tok: str,
    config: dict
) -> Tuple[bool, Dict[str, any]]:
    """
    Check sequence visibility using actual mask pixel counts.
    
    Returns:
        (is_valid, stats)
    """
    mask_req = config["filtering"]["mask_requirements"]
    bev_config = config["filtering"]["bev_config"]
    
    # Create mock loader for mask generation
    mock_loader = MockLoader(nusc, bev_config)
    
    invisible_count = 0
    max_consecutive = 0
    current_consecutive = 0
    visibility_flags = []
    pixel_counts = []
    
    min_pixels_any = mask_req.get("min_pixels_any_frame", 200)
    
    for t, (sample_tok, lidar_tok) in enumerate(zip(obs_tokens, lidar_keyframe_tokens)):
        num_pixels = _get_mask_pixel_count(
            nusc, sample_tok, inst_tok, lidar_tok, mock_loader
        )
        
        pixel_counts.append(num_pixels)
        print(num_pixels) 
        # Consider visible if mask has enough pixels
        is_visible = num_pixels >= min_pixels_any
        visibility_flags.append(is_visible)
        
        if not is_visible:
            invisible_count += 1
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 0
    
    # Check criteria
    is_valid = True
    rejection_reason = None
    
    # First frame must be visible with enough pixels
    if mask_req.get("require_first_visible", True):
        min_first = mask_req.get("min_pixels_first_frame", 500)
        if pixel_counts[0] < min_first:
            is_valid = False
            rejection_reason = f"first_frame_too_small ({pixel_counts[0]} < {min_first} pixels)"
    
    # Check total invisible frames
    if is_valid and invisible_count > mask_req.get("max_invisible_frames", 3):
        is_valid = False
        rejection_reason = f"too_many_invisible ({invisible_count}/{len(obs_tokens)})"
    
    # Check consecutive invisible frames
    if is_valid and max_consecutive > mask_req.get("max_consecutive_invisible", 2):
        is_valid = False
        rejection_reason = f"consecutive_invisible ({max_consecutive})"
    
    stats = {
        "invisible_count": invisible_count,
        "max_consecutive_invisible": max_consecutive,
        "visibility_flags": visibility_flags,
        "pixel_counts": pixel_counts,
        "first_frame_pixels": pixel_counts[0],
        "avg_pixels": sum(pixel_counts) / max(sum(visibility_flags), 1),
        "visibility_ratio": sum(visibility_flags) / len(visibility_flags),
        "rejection_reason": rejection_reason
    }
    
    return is_valid, stats