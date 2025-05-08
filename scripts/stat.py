import argparse
import json
import torchmetrics
import torch
import pathlib
import multiprocessing
import hydra
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf

def ins_to_mask(start, end, instructions):
    # Initialize the mask with zeros
    mask = torch.zeros(end - start, dtype=torch.bool)
    
    # Ensure the start is less than the end for indexing
    start, end = min(start, end), max(start, end)
    
    # Mark the region between start and end as 1 using slicing
    mask[start:end] = 1
    
    # Create a tensor of points and mask the points inside the region
    points_tensor = torch.tensor(instructions)
    
    points_tensor = points_tensor[torch.logical_and(points_tensor >= start, points_tensor < end)] - start
    
    # Use boolean indexing to set points inside the range to 1
    mask[points_tensor] = True
    
    return mask

def process_file(args):
    gt_root_path, gt_file, pred_file, label = args
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

    precision = torchmetrics.functional.precision(pred, labels, 'binary')
    recall = torchmetrics.functional.recall(pred, labels, 'binary')
    print(f"{rel_path}: {precision.item()}, {recall.item()}, {total}")
    return str(rel_path), precision.item(), recall.item(), total

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
