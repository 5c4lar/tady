import pathlib
import argparse
import json
import numpy as np
from tady import cpp
import hydra
from omegaconf import DictConfig

def get_pdt(file: pathlib.Path, successors: np.ndarray, cf: np.ndarray):
    pdt_path = file.parent / "pdt.npz"
    pdt_data = np.load(pdt_path)
    return cpp.PostDominatorTree(successors, cf, pdt_data["wccs"], pdt_data["dom_tree"])

def error_stat(results: dict):
    # Calculate file level error rate and errors per bytes for each category
    error_by_category = results["error_by_category"]
    all_errors = results["all_errors"]
    error_files = results["error_files"]
    total_files = results["total_files"]
    error_rate = error_files / total_files
    error_rate_by_category = {k: v / total_files for k, v in error_by_category.items()}
    all_sizes = results["all_sizes"]
    error_rate_per_bytes = {}
    total_size = 0
    num_errors = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
    for file, errors in all_errors.items():
        size = all_sizes[file]
        total_size += size
        for category, count in errors.items():
            num_errors[category] += count
    error_rate_per_bytes = {k: v / total_size for k, v in num_errors.items()}
    return error_rate_per_bytes, error_rate_by_category, error_rate
    
@hydra.main(version_base=None, config_path="conf", config_name="error_stat")
def main(args: DictConfig):
    disassembler = cpp.Disassembler()
    root_dir = pathlib.Path("data/prune") / args.dataset.name
    output_dir = pathlib.Path("data/error") / args.dataset.name
    gt_dir = pathlib.Path("data/gt_npz") / args.dataset.name
    for model in args.models:
        print(f"Processing {model} for {args.dataset.name}")
        log_file = output_dir / f"{model}_error_stat.log"
        json_path = output_dir / f"{model}_error_stat.json"
        if json_path.exists() and not args.overwrite:
            print(f"Error stat file {json_path} already exists, skipping...")
            with open(json_path, "r") as f:
                results = json.load(f)
                error_rate_per_bytes, error_rate_by_category, error_rate = error_stat(results)
                print(f"Error rate per bytes: {error_rate_per_bytes}")
                print(f"Error rate by category: {error_rate_by_category}")
                print(f"Error rate: {error_rate}")
                continue
        output_dir.mkdir(parents=True, exist_ok=True)
        f = log_file.open("w")
        files = list(root_dir.rglob(f"{model}*.npz"))  # Convert generator to list
        if len(files) == 0:
            print(f"No files found for {model} in {root_dir}")
            continue
        errors = [(file, np.load(file)) for file in files]
        total = len(errors)
        error_files = 0
        error_by_category = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
        all_errors = {}
        all_sizes = {}
        for file in files:
            rel_path = file.relative_to(root_dir)
            pdt_path = file.parent / "pdt.npz"
            num_errors_per_file = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
            gt_path = gt_dir / (str(rel_path.parent) + ".npz")
            gt = np.load(gt_path)
            text_array = gt["text_array"]
            use_64_bit = gt["use_64_bit"]
            size = len(text_array)
            all_sizes[str(rel_path.parent)] = size
            base_addr = int(gt["base_addr"].astype(np.uint64))
            instr_len, flow_kind, _, successors = disassembler.superset_disasm(text_array, use_64_bit)

            error = np.load(file)
            has_error = False
            print(f"Processing {file}")
            if len(error["dangling"]) > 0:
                has_error = True
                num_errors_per_file["dangling"] += len(error["dangling"])
                error_by_category["dangling"] += 1
                f.write(f"DANGLING ERROR: {rel_path.parent}: {len(error['dangling'])}\n")
            if len(error["exclusive"]) > 0:
                has_error = True
                num_errors_per_file["exclusive"] += len(error["exclusive"])
                error_by_category["exclusive"] += 1
                f.write(f"EXCLUSIVE ERROR: {rel_path.parent}: {len(error['exclusive'])}\n")
            if len(error["coexist"]) > 0:
                pdt = get_pdt(file, successors, flow_kind > 1)
                absolute_ipdom = pdt.get_absolute_ipdom()
                currents = error["coexist"].astype(np.uint64)
                ipdoms = absolute_ipdom[currents - base_addr]
                has_coexist = False
                error_nop = False
                for current, ipdom in zip(currents, ipdoms):
                    if ipdom == -1:
                        continue
                    inst_len = instr_len[ipdom]
                    current_inst_len = instr_len[current - base_addr]
                    current_inst = disassembler.inst_str(text_array[current - base_addr:current - base_addr + current_inst_len], use_64_bit.item(), int(current)).lstrip()
                    addr = int(base_addr) + int(ipdom)
                    inst_str = disassembler.inst_str(text_array[ipdom:ipdom+inst_len], use_64_bit.item(), addr).lstrip()
                    if inst_str.startswith("nop"): # or inst_str.startswith("lea"):
                        error_nop = True
                        num_errors_per_file["nop"] += 1
                        f.write(f"NOP ERROR: {rel_path.parent}: 0x{current:x}[{current_inst_len}]-0x{addr:x}[{inst_len}] {current_inst} {inst_str}\n")
                    elif current_inst.startswith("lock"):
                        pass
                    else:
                        has_coexist = True
                        num_errors_per_file["coexist"] += 1
                        f.write(f"COE ERROR: {rel_path.parent}: 0x{current:x}[{current_inst_len}]-0x{addr:x}[{inst_len}] {current_inst} {inst_str}\n")
                if error_nop:
                    error_by_category["nop"] += 1
                    has_error = True
                if has_coexist:
                    has_error = True
                    error_by_category["coexist"] += 1
            if has_error:
                error_files += 1
            all_errors[str(rel_path.parent)] = num_errors_per_file
            
        for category, count in error_by_category.items():
            f.write(f"{category}: {count}, {count / total:.4f}\n")
        f.write(f"Total files: {total}\n")
        f.write(f"Error files: {error_files}\n")
        f.write(f"Error rate: {error_files / total:.4f}\n")
        results = {
            "error_by_category": error_by_category,
            "all_errors": all_errors,
            "error_files": error_files,
            "total_files": total,
            "error_rate": error_files / total,
            "all_sizes": all_sizes
        }
        with open(json_path, "w") as f:
            json.dump(results, f, indent=4)
        error_rate_per_bytes, error_rate_by_category, error_rate = error_stat(results)
        print(f"Error rate per bytes: {error_rate_per_bytes}")
        print(f"Error rate by category: {error_rate_by_category}")
        print(f"Error rate: {error_rate}")
        f.close()

if __name__ == "__main__":
    main()