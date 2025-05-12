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
    disassembler = cpp.Disassembler("x86_64")
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
    
    
@hydra.main(version_base=None, config_path="conf", config_name="prune")
def main(args: DictConfig):
    bin_dir = pathlib.Path(args.bin_dir)
    model_id = "_".join([str(i) for i in args.tags])
    input_path = pathlib.Path(args.input_dir) / model_id
    files = bin_dir.rglob("*")
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