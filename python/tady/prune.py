from tady.utils.loader import load_text
import numpy as np
import pathlib
from tady import cpp
from typing import Optional


def precision(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Computes precision for binary classification."""
    true_positives = np.sum(np.logical_and(pred == 1, target == 1), where=mask)
    predicted_positives = np.sum(pred == 1, where=mask)
    return true_positives / (predicted_positives + epsilon)

def recall(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Computes recall for binary classification."""
    true_positives = np.sum(np.logical_and(pred == 1, target == 1), where=mask)
    actual_positives = np.sum(target == 1, where=mask)
    return true_positives / (actual_positives + epsilon)

def f1(p, r, epsilon: float = 1e-7) -> np.ndarray:
    return 2 * p * r / (p + r + epsilon)

def fp(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, base_addr: np.ndarray) -> np.ndarray:
    false_positives = np.where(np.logical_and(pred == 1, target == 0) & mask)[0] + base_addr
    return false_positives.astype(np.uint64)

def fn(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, base_addr: np.ndarray) -> np.ndarray:
    false_negatives = np.where(np.logical_and(pred == 0, target == 1) & mask)[0] + base_addr
    return false_negatives.astype(np.uint64)

def get_pdt(successors: np.ndarray, cf: np.ndarray, pdt_path: Optional[pathlib.Path] = None):

    if pdt_path is not None and pdt_path.exists():
        try:
            pdt_cache = np.load(pdt_path)
            wccs = pdt_cache["wccs"]
            dom_tree = pdt_cache["dom_tree"]
            pdt = cpp.PostDominatorTree(successors, cf, wccs, dom_tree)
        except:
            pdt = cpp.PostDominatorTree(successors, cf)
            wccs = pdt.get_wccs()
            dom_tree = pdt.get_ipdom()
            np.savez(pdt_path, wccs=wccs, dom_tree=dom_tree)
    else:
        pdt = cpp.PostDominatorTree(successors, cf)
    return pdt

def prune(gt_path: pathlib.Path, pred_path: pathlib.Path, pdt_path: Optional[pathlib.Path] = None):
    data = np.load(gt_path)
    disassembler = cpp.Disassembler()
    text_array = data["text_array"]
    use_64_bit = data["use_64_bit"].item()
    base_addr = data["base_addr"].item()
    mask = data["mask"]
    instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)
    cf = flow_kind > 1
    pdt = get_pdt(successors, cf, pdt_path)
    pred_data = np.load(pred_path)
    if "logits" in pred_data:
        score = pred_data["logits"]
    else:
        pred = pred_data["pred"]
        score = np.where(pred, 1, -1)
    pruned = pdt.prune(score)
    pruned_pred = np.zeros(len(text_array), dtype=np.bool_)
    pruned_pred[pruned] = 1
    pruned_pred = np.where(mask, pruned_pred, False)
    p = precision(pruned_pred, data["labels"], mask)
    r = recall(pruned_pred, data["labels"], mask)
    f1_score = f1(p, r)
    errors = pdt.get_errors(score)
    fps = fp(pruned_pred, data["labels"], mask, base_addr)
    fns = fn(pruned_pred, data["labels"], mask, base_addr)
    result = {
        "precision": p,
        "recall": r,
        "f1": f1_score,
        "fps": fps,
        "fns": fns,
        "coexist": errors["coexist"] + base_addr,
        "dangling": errors["dangling"] + base_addr,
        "exclusive": errors["exclusive"] + base_addr,
        "pruned": pruned_pred,
    }
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", type=str, help="Path to the gt npz")
    parser.add_argument("--pred", type=str, help="Path to the pred npz")
    parser.add_argument("--pdt", type=str, help="Path to the pdt npz")
    args = parser.parse_args()
    result = prune(args.gt, args.pred, args.pdt)
    print(result["precision"])
    print(result["recall"])
    print(result["f1"])
    print([f"{x:x}" for x in result["fps"]])
    print([f"{x:x}" for x in result["fns"]])
    # print(result["coexist"])
    # print(result["dangling"])
    # print(result["exclusive"])
    # print(result["pruned"])

if __name__ == "__main__":
    main()
