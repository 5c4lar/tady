import numpy as np
import pathlib
from tady.prune import precision, recall, f1, fp, fn

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Path to the gt npz")
    parser.add_argument("--pred", type=str, help="Path to the pred npz")
    args = parser.parse_args()
    gt = np.load(args.gt)
    mask = gt["mask"]
    label = gt["labels"]
    pred_data = np.load(args.pred)
    pred = pred_data["pred"]
    base_addr = pred_data["base_addr"].astype(np.uint64)
    gt_list = np.where(label & mask)[0].astype(np.uint64) + base_addr
    print([hex(addr) for addr in gt_list.tolist()])
    pred_list = np.where(pred & mask)[0].astype(np.uint64) + base_addr
    print([hex(addr) for addr in pred_list.tolist()])
    p = precision(pred, label, mask)
    r = recall(pred, label, mask)
    f1_score = f1(p, r)
    fp_addr = fp(pred, label, mask, base_addr)
    fn_addr = fn(pred, label, mask, base_addr)
    fp_addr = [hex(addr) for addr in fp_addr]
    fn_addr = [hex(addr) for addr in fn_addr]
    print(f"Precision: {p}")
    print(f"Recall: {r}")
    print(f"F1 score: {f1_score}")
    print(f"False positives: {fp_addr}")
    print(f"False negatives: {fn_addr}")

if __name__ == "__main__":
    main()