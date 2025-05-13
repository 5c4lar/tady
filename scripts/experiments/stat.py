import json
import pathlib
import multiprocessing
import hydra
from tqdm import tqdm
from omegaconf import DictConfig
import numpy as np

def precision(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Computes precision for binary classification."""
    true_positives = np.sum(np.logical_and(pred == 1, target == 1))
    predicted_positives = np.sum(pred == 1)
    return true_positives / (predicted_positives + epsilon)

def recall(pred: np.ndarray, target: np.ndarray, epsilon: float = 1e-7) -> np.ndarray:
    """Computes recall for binary classification."""
    true_positives = np.sum(np.logical_and(pred == 1, target == 1))
    actual_positives = np.sum(target == 1)
    return true_positives / (actual_positives + epsilon)

def ins_to_mask(start, end, instructions):
    # Ensure the start is less than the end for indexing
    # This standard Python min/max should be fine if start/end are scalars
    # If they are JAX arrays and this function is JITted, this might need adjustment
    # or be handled outside. For now, assuming they are scalars as per typical usage.
    _start, _end = min(start, end), max(start, end)
    
    # Initialize the mask with zeros
    # The size of the mask is determined by the original, potentially swapped, start and end
    mask_size = end - start
    mask = np.zeros(mask_size, dtype=np.bool_)

    # Create a tensor of points and mask the points inside the region
    points_tensor = np.array(instructions)
    
    # Filter points that fall within the [_start, _end) range
    # and normalize them relative to the original 'start' to match mask indices
    valid_points = points_tensor[np.logical_and(points_tensor >= _start, points_tensor < _end)]
    normalized_points = valid_points - start # Normalize to the original start

    # Use boolean indexing to set points inside the range to True
    # Only update if normalized_points is not empty
    if normalized_points.size > 0:
        mask[normalized_points] = True
    
    return mask

def process_file(args):
    gt_root_path, gt_file, pred_file, label = args
    # print(gt_file, pred_file)
    gt = json.load(open(gt_file, "r"))
    rel_path = gt_file.relative_to(gt_root_path)
    try:
        addrs = json.load(open(pred_file, "r"))
        if type(addrs) == dict:
            addrs = addrs[label + "s"]
        if type(addrs[0]) == list:
            addrs = sum(addrs, [])
    except:
        print(f"Failed to process {pred_file}")
        return str(rel_path), None, None, None
    start, end = gt['start'], gt['end']
    total = end - start
    labels = ins_to_mask(gt["start"], gt["end"], gt[label + "s"])
    pred = ins_to_mask(gt["start"], gt["end"], addrs)

    p = precision(pred, labels)
    r = recall(pred, labels)
    true_insts = np.sum(labels)
    correct_insts = np.sum(np.logical_and(pred, labels))
    print(f"{rel_path}: {p}, {r}, {total}, {true_insts}, {correct_insts}")
    return str(rel_path), p, r, total

@hydra.main(version_base=None, config_path="conf", config_name="stat")
def main(args: DictConfig):

    gt_root_path = pathlib.Path(args.gt)
    model_id = "_".join([str(i) for i in args.tags])
    output_path = pathlib.Path(args.output) / (model_id + ".json")
    # if output_path.exists():
    #     return
    input_root_path = pathlib.Path(args.input) / model_id
    print(gt_root_path, input_root_path)
    if gt_root_path.is_file() and input_root_path.is_file():
        process_file((gt_root_path, gt_root_path, input_root_path))
        return
    files = gt_root_path.rglob("*")
    # files = [i for i in gt_root_path.rglob("*") if i.is_file() if 'SPEC' in str(i) and (not 'OfastLTO' in str(i)) and (not 'OsLTO' in str(i))]
    
    tasks = []
    for file in files:
        if file.is_file():
            rel_path = file.relative_to(gt_root_path)
            pred_file = input_root_path / rel_path
            if not pred_file.exists():
                # print(pred_file.with_suffix(""))
                if pred_file.with_suffix("").is_dir():
                    pred_file = pred_file.with_suffix("") / "result.json"
                else:
                    continue
            tasks.append((gt_root_path, file, pred_file, args.label))

    results = []
    with multiprocessing.Pool(args.process) as pool, tqdm(total = len(tasks)) as pbar:
        for result in pool.imap_unordered(process_file, tasks):
            pbar.update()
            pbar.refresh()
            results.append(result)

     # Collect results into a dictionary
    results_dict = {path: {"precision": precision, "recall": recall, "total": total} for path, precision, recall, total in results if total is not None}
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results_dict, f)
        
if __name__ == "__main__":
    main()
