# build_index_5to1.py
import os, pickle, argparse, random
from data.configs.filenames import DATAROOT, TRAIN_INDEX, VAL_INDEX
from nuscenes.nuscenes import NuScenes
from index_nuscenes.agent_index import build_agent_sequence_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default=DATAROOT)
    ap.add_argument("--version",  type=str, default="v1.0-trainval")
    ap.add_argument("--cameras",  nargs="+",
                    default=["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"])
    ap.add_argument("--t_in",     type=int,   default=4)
    ap.add_argument("--t_out",    type=int,   default=12)
    ap.add_argument("--stride",   type=int,   default=1)
    ap.add_argument("--min_future",     type=int,   default=12)
    ap.add_argument("--min_speed_mps",  type=float, default=0.0)
    ap.add_argument("--n_total",  type=int,   default=2000, help="how many rows to keep in total (train+val)")
    ap.add_argument("--seed",     type=int,   default=42)
    ap.add_argument("--out_prefix", type=str, default="agents_index")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    # Build a single pool from ALL scenes (train+val) — ONE CALL ONLY.
    rows = build_agent_sequence_index(
        nusc,
        splits=None,                      # <- all scenes in this version
        cameras=args.cameras,
        t_in=args.t_in,
        t_out=args.t_out,
        stride=args.stride,
        min_future=args.min_future,
        min_speed_mps=args.min_speed_mps,
        dataroot=args.dataroot,
        throttle_max_rows=args.n_total            # IMPORTANT: disable the default 200-row throttle
    )

    if len(rows) == 0:
        raise RuntimeError("No rows built. Check your filters (min_future, cameras, etc.).")

    # Shuffle so we don't “start from the beginning” each time
    rng = random.Random(args.seed)
    rng.shuffle(rows)

    # Keep N_TOTAL and split 5:1 (≈83.33% train / 16.67% val)
    n_total  = args.n_total
    rows     = rows[:n_total]
    n_train  = (5 * n_total) // 6
    n_val    = n_total - n_train
    train_rows = rows[:n_train]
    val_rows   = rows[n_train:n_train + n_val]

    # Save
    with open(TRAIN_INDEX, "wb") as f:
        pickle.dump(train_rows, f)
    with open(VAL_INDEX, "wb") as f:
        pickle.dump(val_rows, f)

    print(f"Built pool: {len(rows)} rows  ->  train: {len(train_rows)}  |  val: {len(val_rows)}")
    print(f"Saved to: {TRAIN_INDEX}  /  {VAL_INDEX}")

if __name__ == "__main__":
    main()
