import os, pickle, argparse, random
from nuscenes.nuscenes import NuScenes
from agent_index import build_agent_sequence_index

def _downsample(rows, frac: float, seed: int):
    if not rows or frac >= 1.0:
        return rows
    n_keep = max(1, int(len(rows) * frac))
    rng = random.Random(seed)
    keep_idx = sorted(rng.sample(range(len(rows)), n_keep))
    return [rows[i] for i in keep_idx]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default=r"e:\nuscenes")
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--cameras", nargs="+", default=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"])
    ap.add_argument("--t_in", type=int, default=8)
    ap.add_argument("--t_out", type=int, default=10)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_future", type=int, default=10)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)
    ap.add_argument("--out_prefix", type=str, default="agents_index")
    ap.add_argument("--fraction", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    common = dict(
        cameras=args.cameras, t_in=args.t_in, t_out=args.t_out,
        stride=args.stride, min_future=args.min_future,
        min_speed_mps=args.min_speed_mps, dataroot=args.dataroot,
    )

    train_rows = build_agent_sequence_index(nusc, splits="train", **common)
    val_rows   = build_agent_sequence_index(nusc, splits="val",   **common)

    train_rows = _downsample(train_rows, args.fraction, args.seed)
    val_rows   = _downsample(val_rows, args.fraction, args.seed)

    with open(f"train_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(train_rows, f)
    with open(f"val_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(val_rows, f)

    print(f"Train: {len(train_rows)} rows")
    print(f"Val:   {len(val_rows)} rows")

if __name__ == "__main__":
    main()
