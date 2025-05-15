import json

import pathlib
import numpy as np
import hydra
from omegaconf import DictConfig
from tady import cpp
from tady.utils.loader import load_text
from multiprocessing import Pool
from tqdm import tqdm

def process_file(arg):
    args, file, rel_path, model_id, result_path = arg
    prune_path = pathlib.Path(args.prune_dir) / model_id / (str(rel_path) + ".json")
    errors_path = pathlib.Path(args.errors_dir) / model_id / (str(rel_path) + ".json")
    if prune_path.exists() and errors_path.exists() and not args.overwrite:
        return
    disassembler = cpp.Disassembler()
    text_array, use_64_bit, base_addr = load_text(file)
    instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)
    score = np.load(result_path)
    cf = flow_kind > 1
    pdt = cpp.PostDominatorTree(successors, cf)
    pruned = pdt.prune(score) + base_addr
    errors = pdt.get_errors(score)
    prune_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prune_path, "w") as f:
        json.dump({"instructions": pruned.tolist()}, f)
    with open(errors_path, "w") as f:
        json.dump({key: (value + base_addr).tolist() for key, value in errors.items()}, f)
      
def gt_to_scores(gt_path, start, end):
    with open(gt_path, "r") as f:
        gt = json.load(f)
    scores = np.ones(end - start) * -1
    points = np.array(gt["instructions"], dtype=np.int32)
    valid_points = points[(points >= start) & (points < end)]
    scores[valid_points - start] = 1
    return scores

def process_gt(arg):
    args, file, rel_path, result_path = arg
    prune_path = pathlib.Path(args.prune_dir) / "gt" / (str(rel_path) + ".json")
    errors_path = pathlib.Path(args.errors_dir) / "gt" / (str(rel_path) + ".json")
    if prune_path.exists() and errors_path.exists() and not args.overwrite:
        return
    disassembler = cpp.Disassembler()
    text_array, use_64_bit, base_addr = load_text(file)
    start, end = base_addr, base_addr + len(text_array)
    instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)
    score = gt_to_scores(result_path, start, end)
    cf = flow_kind > 1
    pdt = cpp.PostDominatorTree(successors, cf)
    pruned = pdt.prune(score) + base_addr
    errors = pdt.get_errors(score)
    prune_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.parent.mkdir(parents=True, exist_ok=True)
    with open(prune_path, "w") as f:
        json.dump({"instructions": pruned.tolist()}, f)
    with open(errors_path, "w") as f:
        json.dump({key: (value + base_addr).tolist() for key, value in errors.items()}, f)
        
@hydra.main(version_base=None, config_path="conf", config_name="prune")
def main(args: DictConfig):
    bin_dir = pathlib.Path(args.bin_dir)
    files = bin_dir.rglob("*")
    
    if args.prune_gt:
        tasks_gt = []
        gt_dir = pathlib.Path(args.gt_dir)
        for file in files:
            if file.is_file():
                rel_path = file.relative_to(bin_dir)
                result_path = gt_dir / (str(rel_path) + ".json")
                if not result_path.exists():
                    # print(f"Result not found for {rel_path}")
                    continue
                tasks_gt.append((args, file, rel_path, result_path))
        with Pool(args.process) as pool, tqdm(total = len(tasks_gt)) as pbar:
            for result in pool.imap_unordered(process_gt, tasks_gt):
                pbar.update()
                pbar.refresh()
    else:
        model_id = "_".join([str(i) for i in args.tags])
        input_path = pathlib.Path(args.input_dir) / model_id
        tasks = []
        for file in files:
            if file.is_file():
                rel_path = file.relative_to(bin_dir)
                result_path = input_path / rel_path / "score.npy"
                if not result_path.exists():
                    print(f"Result not found for {rel_path}")
                    continue
                tasks.append((args, file, rel_path, model_id, result_path))
        with Pool(args.process) as pool, tqdm(total = len(tasks)) as pbar:
            for result in pool.imap_unordered(process_file, tasks):
                pbar.update()
                pbar.refresh()

if __name__ == "__main__":
    main()