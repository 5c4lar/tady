import numpy as np
import pathlib


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--prune_dir", type=str, default="data/prune")
    parser.add_argument("--gt_dir", type=str, default="data/gt_npz")
    args = parser.parse_args()
    gt_dir = pathlib.Path(args.gt_dir)
    gt_path = gt_dir / args.dataset / (args.input + ".npz")
    prune_dir = pathlib.Path(args.prune_dir)
    prune_path = prune_dir / args.dataset / args.input / "gt.npz"

    data = np.load(prune_path)
    
    print("exclusive", data["exclusive"].astype(np.uint64))
    print("dangling", data["dangling"].astype(np.uint64))
    print("coexist", data["coexist"].astype(np.uint64))
    
if __name__ == "__main__":
    main()