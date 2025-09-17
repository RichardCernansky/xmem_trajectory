import os, pickle, argparse, random
from nuscenes.nuscenes import NuScenes
from agent_index import build_agent_sequence_index


PERCENTAGE = 100 # in %

def _downsample(rows, frac: float, seed: int):
    if not rows:
        return rows
    if frac >= 1.0:
        return rows
    n_keep = max(1, int(len(rows) * frac))
    rng = random.Random(seed)
    # sample indices, then sort to preserve original order
    keep_idx = sorted(rng.sample(range(len(rows)), n_keep))
    return [rows[i] for i in keep_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default=r"e:\nuscenes")
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--camera", type=str, default="CAM_FRONT")
    ap.add_argument("--t_in", type=int, default=3)
    ap.add_argument("--t_out", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_future", type=int, default=30)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)
    ap.add_argument("--out_prefix", type=str, default="agents_index")
    # NEW: downsampling controls
    ap.add_argument("--fraction", type=float, default=PERCENTAGE/100, help="Fraction of rows to keep per split.")
    ap.add_argument("--seed", type=int, default=42, help="Random seed for reproducible sampling.")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    common = dict(
        cam=args.camera, t_in=args.t_in, t_out=args.t_out, stride=args.stride,
        min_future=args.min_future, min_speed_mps=args.min_speed_mps, dataroot=args.dataroot
    )

    # Build full indexes
    train_rows = build_agent_sequence_index(nusc, splits="train", **common)
    val_rows = build_agent_sequence_index(nusc, splits="val", **common)

    # Downsample for testing
    train_rows_small = _downsample(train_rows, args.fraction, args.seed)
    val_rows_small = _downsample(val_rows, args.fraction, args.seed)

    # Save
    with open(f"train_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(train_rows_small, f)

    with open(f"val_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(val_rows_small, f)

    print(f"Train: kept {len(train_rows_small)}/{len(train_rows)} rows "
          f"({len(train_rows_small)/max(1,len(train_rows)):.1%})")
    print(f"Val:   kept {len(val_rows_small)}/{len(val_rows)} rows "
          f"({len(val_rows_small)/max(1,len(val_rows)):.1%})")

if __name__ == "__main__":
    main()
