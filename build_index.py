import os, pickle, argparse
from nuscenes.nuscenes import NuScenes
from agent_index import build_agent_sequence_index

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataroot", type=str, default=r"e:\nuscenes")
    ap.add_argument("--version", type=str, default="v1.0-trainval")
    ap.add_argument("--camera", type=str, default="CAM_FRONT")
    ap.add_argument("--t_in", type=int, default=8)
    ap.add_argument("--t_out", type=int, default=30)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--min_future", type=int, default=30)
    ap.add_argument("--min_speed_mps", type=float, default=0.0)
    ap.add_argument("--out_prefix", type=str, default="agents_index")
    args = ap.parse_args()

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=True)

    common = dict(
        cam=args.camera, t_in=args.t_in, t_out=args.t_out, stride=args.stride,
        min_future=args.min_future, min_speed_mps=args.min_speed_mps, dataroot=args.dataroot
    )

    train_rows = build_agent_sequence_index(nusc, splits="train", **common)
    with open(f"train_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(train_rows, f)

    val_rows = build_agent_sequence_index(nusc, splits="val", **common)
    with open(f"val_{args.out_prefix}.pkl", "wb") as f:
        pickle.dump(val_rows, f)

if __name__ == "__main__":
    main()
