import pathlib
import argparse
import json
import numpy as np
from tady import cpp
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
from multiprocessing import Pool
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
    # error_rate = error_files / total_files
    error_rate_by_category = {k: v / total_files for k, v in error_by_category.items()}
    all_sizes = results["all_sizes"]
    error_rate_per_bytes = {}
    total_size = 0
    num_errors = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0, "total": 0}
    has_errors = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0, "total": 0}
    for file, errors in all_errors.items():
        size = all_sizes[file]
        total_size += size
        # try:
        #     errors["coexist"] += errors["nop"]
        #     errors.pop("nop")
        # except:
        #     pass
        has_error = False
        for category, count in errors.items():
            num_errors[category] += count
            has_errors[category] += count > 0
            num_errors["total"] += count
            has_error = has_error or count > 0
        has_errors["total"] += has_error
    error_rate_per_bytes = {k: v / total_size for k, v in num_errors.items()}
    error_rate_by_category = {k: v / total_files for k, v in has_errors.items()}
    return error_rate_per_bytes, error_rate_by_category, num_errors, len(all_errors)
  
def result_summary(results: dict):
    error_rate_per_bytes, error_rate_by_category, num_errors, total_files = error_stat(results)
    result = {
        "file_level_error_rate": error_rate_by_category,
        "byte_level_error_rate": error_rate_per_bytes,
        "error_rate_per_1MB": {k: v * 1024 * 1024 for k, v in error_rate_per_bytes.items()},
        "error_num": num_errors,
        "total_files": total_files
    }
    return result
    
def process_file(arg):
    args, file, dataset = arg
    root_dir = pathlib.Path("data/prune") / dataset
    output_dir = pathlib.Path("data/error") / dataset
    gt_dir = pathlib.Path("data/gt_npz") / dataset
    disassembler = cpp.Disassembler()
    rel_path = file.relative_to(root_dir)
    pdt_path = file.parent / "pdt.npz"
    num_errors_per_file = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
    gt_path = gt_dir / (str(rel_path.parent) + ".npz")
    gt = np.load(gt_path)
    text_array = gt["text_array"]
    use_64_bit = gt["use_64_bit"]
    size = len(text_array)
    base_addr = int(gt["base_addr"].astype(np.uint64))
    instr_len, flow_kind, _, successors = disassembler.superset_disasm(text_array, use_64_bit)
    error_by_category = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
    error = np.load(file)
    has_error = False
    error_report = []
    print(f"Processing {file}")
    if len(error["dangling"]) > 0:
        has_error = True
        num_errors_per_file["dangling"] += len(error["dangling"])
        error_by_category["dangling"] += 1
        error_report.append(f"DANGLING ERROR: {rel_path.parent}: {len(error['dangling'])}")
    if len(error["exclusive"]) > 0:
        has_error = True
        num_errors_per_file["exclusive"] += len(error["exclusive"])
        error_by_category["exclusive"] += 1
        error_report.append(f"EXCLUSIVE ERROR: {rel_path.parent}: {len(error['exclusive'])}")
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
                error_report.append(f"NOP ERROR: {rel_path.parent}: 0x{current:x}[{current_inst_len}]-0x{addr:x}[{inst_len}] {current_inst} {inst_str}")
            elif current_inst.startswith("lock"):
                pass
            else:
                has_coexist = True
                num_errors_per_file["coexist"] += 1
                error_report.append(f"COE ERROR: {rel_path.parent}: 0x{current:x}[{current_inst_len}]-0x{addr:x}[{inst_len}] {current_inst} {inst_str}")
        if error_nop:
            error_by_category["nop"] += 1
            has_error = True
        if has_coexist:
            has_error = True
            error_by_category["coexist"] += 1

    return str(rel_path.parent), size, num_errors_per_file, error_by_category, error_report, has_error

@hydra.main(version_base=None, config_path="conf", config_name="error_stat")
def main(args: DictConfig):
    all_stats = {}
    for dataset in args.datasets:
        root_dir = pathlib.Path("data/prune") / dataset
        output_dir = pathlib.Path("data/error") / dataset
        gt_dir = pathlib.Path("data/gt_npz") / dataset
        for model in args.models:
            print(f"Processing {model} for {dataset}")
            log_file = output_dir / f"{model}_error_stat.log"
            json_path = output_dir / f"{model}_error_stat.json"
            if json_path.exists() and not args.overwrite:
                print(f"Error stat file {json_path} already exists, skipping...")
                with open(json_path, "r") as f:
                    results = json.load(f)
                    print(f"Total files: {results['total_files']}")
                    # results["error_by_category"]["coexist"] += results["error_by_category"]["nop"]
                    # results["error_by_category"].pop("nop")
                    error_rate_per_bytes, error_rate_by_category, error_rate, total_files = error_stat(results)
                    print(f"Error count by category: {results['error_by_category']}")
                    print(f"Errors total: {sum(results['error_by_category'].values())}")
                    print(f"Error rate per bytes: {error_rate_per_bytes}")
                    print(f"Error rate by category: {error_rate_by_category}")
                    print(f"Error rate: {error_rate}")
                    all_stats[f"{dataset}_{model}"] = result_summary(results)
                    continue
            output_dir.mkdir(parents=True, exist_ok=True)
            f = log_file.open("w")
            files = list(root_dir.rglob(f"{model}*.npz"))  # Convert generator to list
            if len(files) == 0:
                print(f"No files found for {model} in {root_dir}")
                continue
            if args.num_samples and args.num_samples < len(files):
                import random
                random.seed(0)
                files = random.sample(files, args.num_samples)
            # errors = [(file, np.load(file)) for file in files]
            total = len(files)
            error_files = 0
            error_by_category = {"coexist": 0, "dangling": 0, "exclusive": 0, "nop": 0}
            all_errors = {}
            all_sizes = {}
            with Pool(args.process) as pool, tqdm(total = len(files)) as pbar:
                for result in pool.imap_unordered(process_file, [(args, file, dataset) for file in files]):
                    path, size, num_errors_per_file, error_by_category, error_report, has_error = result
                    f.write("\n".join(error_report) + "\n")
                    all_errors[path] = num_errors_per_file
                    all_sizes[path] = size
                    for category, count in num_errors_per_file.items():
                        error_by_category[category] += count
                    error_files += has_error
                    pbar.update()
                    pbar.refresh()
            for category, count in error_by_category.items():
                f.write(f"{category}: {count}, {count / total:.4f}\n")
            f.write(f"Total files: {total}\n")
            f.write(f"Error files: {error_files}\n")
            f.write(f"Error rate: {error_files / total:.4f}\n")
            print(f"Total files: {total}")
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
            error_rate_per_bytes, error_rate_by_category, error_rate, total_files = error_stat(results)
            print(f"Error rate per bytes: {error_rate_per_bytes}")
            print(f"Error rate by category: {error_rate_by_category}")
            print(f"Error rate: {error_rate}")
            f.close()
            all_stats[f"{dataset}_{model}"] = result_summary(results)
    with open(pathlib.Path("data/error") / "error_stat.json", "w") as f:
        json.dump(all_stats, f, indent=4)

if __name__ == "__main__":
    main()