from tady import cpp
from tady.utils.loader import load_text
from multiprocessing import Pool
import pathlib
from tqdm import tqdm
import json
import numpy as np
disassembler = cpp.Disassembler()

def process_file(arg):
    args, file_path = arg
    base_dir = pathlib.Path(args.dir)
    rel_path = file_path.relative_to(base_dir)
    gt_base_dir = pathlib.Path(args.gt)
    gt_path = gt_base_dir / (str(rel_path) + ".json")
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}")
        return file_path, (0, 0, 0)
    with open(gt_path, "r") as f:
        gt = json.load(f)
    
    text_array, use_64_bit, base_addr = load_text(file_path)
    instructions = np.array(gt["instructions"]) - base_addr
    gt_mask = np.zeros(len(text_array), dtype=np.bool)
    valid_mask = (instructions >= 0) & (instructions < len(text_array))
    if sum(valid_mask) == 0:
        return file_path, (0, 0, 0)
    try:
        gt_mask[instructions[valid_mask]] = True
    except Exception as e:
        print(instructions[valid_mask])
        raise e
    instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)
    flow_kind = flow_kind[gt_mask]
    control_flow = control_flow[gt_mask]
    addrs = (np.arange(len(text_array), dtype=np.int32) + base_addr)[gt_mask]
    indirect = addrs[flow_kind == 10]
    # icalls = addrs[((flow_kind == 10) & (control_flow[:, 0] == -1))]
    # ijumps = addrs[((flow_kind == 10) & (control_flow[:, 0] == -1))]
    # indirect_calls = len(icalls)
    # indirect_jumps = len(ijumps)
    # ic, ij, total = int(indirect_calls), int(indirect_jumps), int(sum(gt_mask))
    # rate = float(ic + ij) / total
    # print(f"{file_path}: {rate}")
    # if indirect_calls > 0 or indirect_jumps > 0:
        # print(f"Indirect calls: {indirect_calls}")
        # print(f"Indirect jumps: {indirect_jumps}")
        # print(f"File: {file_path}")
    return file_path, (len(indirect), len(gt_mask), len(indirect) / len(gt_mask))
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, required=True)
    parser.add_argument("--gt", type=str, required=True)
    parser.add_argument("--output", type=str, required=False)
    parser.add_argument("--process", type=int, default=16)
    args = parser.parse_args()
    base_dir = pathlib.Path(args.dir)
    files = [f for f in base_dir.rglob("*") if f.is_file()]
    results = {}
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for file_path,result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            results[str(file_path)] = result
            pbar.update()
            pbar.refresh()
    # Sort by the number of indirect
    results = sorted(results.items(), key=lambda x: x[1][0], reverse=True)
    with open(args.output, "w") as f:
        json.dump(results, f)
    # calculate the max number of indirect
if __name__ == "__main__":
    main()