from collections import defaultdict
import json

import pathlib
import numpy as np
import hydra
from omegaconf import DictConfig
from tady import cpp
from tady.utils.loader import load_text
from multiprocessing import Pool
from tqdm import tqdm

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
    return false_positives

def fn(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, base_addr: np.ndarray) -> np.ndarray:
    false_negatives = np.where(np.logical_and(pred == 0, target == 1) & mask)[0] + base_addr
    return false_negatives

def get_pdt(output_prefix: pathlib.Path, successors: np.ndarray, cf: np.ndarray):
    pdt_path = output_prefix / "pdt.npz"
    pdt_path.parent.mkdir(parents=True, exist_ok=True)
    if pdt_path.exists():
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
        wccs = pdt.get_wccs()
        dom_tree = pdt.get_ipdom()
        np.savez(pdt_path, wccs=wccs, dom_tree=dom_tree)
    return pdt

def process_file(arg):
    args, file, rel_path = arg
    output_prefix = pathlib.Path(args.output_dir) / (rel_path.with_suffix(""))
    data = np.load(file)
    disassembler = cpp.Disassembler()
    text_array = data["text_array"]
    use_64_bit = data["use_64_bit"].item()
    base_addr = data["base_addr"].item()
    pdt = None
    for model in args.models:
        output_path = output_prefix / (model + ".npz")
        if output_path.exists():
            continue
        
        if model == "gt":
            score = np.ones(len(text_array), dtype=np.float32) * -1
            score[data["labels"]] = 1
        else:
            eval_path = pathlib.Path(args.eval_dir) / model / rel_path
            if not eval_path.exists():
                # print(f"Requirement {str(eval_path)} does not exist")
                continue
            try:
                eval_data = np.load(eval_path)
                score = eval_data["scores"]
            except Exception as e:
                print(f"Error loading {eval_path}: {e}")
                eval_path.unlink()
                continue
        if pdt is None:
            instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)   
            cf = flow_kind > 1
            pdt = get_pdt(output_prefix, successors, cf)
        pruned = pdt.prune(score)
        pruned_pred = np.zeros(len(text_array), dtype=np.bool_)
        pruned_pred[pruned] = 1
        p = precision(pruned_pred, data["labels"], data["mask"])
        r = recall(pruned_pred, data["labels"], data["mask"])
        f1_score = f1(p, r)
        errors = pdt.get_errors(score)
        result = {
            "precision": p,
            "recall": r,
            "f1": f1_score,
            "coexist": errors["coexist"] + data["base_addr"],
            "dangling": errors["dangling"] + data["base_addr"],
            "exclusive": errors["exclusive"] + data["base_addr"],
            "pruned": pruned_pred,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        

def parse_rw_opts(task):
    parts = str(task).split("/")
    proj = parts[0]
    compiler = parts[1]
    opt = parts[2]
    arch = "x86" if '32' in compiler else "x64"
    return (compiler, opt, arch)

def parse_x86_sok_opts(task):
    # Parse the task string to extract dataset, compiler, and optimization level
    # Example: "linux/utils/binutils/gcc_m32_Os/elfedit.json", "linux/utils/binutils/gcc_Os/elfedit.json"
    parts = task.split("/")
    if 'openssl' in parts[1]:
        opts = parts[-1]
        options = opts.split("_")
        opt = options[-1].split(".")[0]
        compiler = "gcc"
        if len(options) == 2:
            arch = 'x64'
        elif len(options) == 3:
            arch = 'x32'
        return (compiler, opt, arch)
    opts = parts[-2]
    options = opts.split("_")
    if len(options) == 2:
        compiler = options[0]
        opt = options[1]
        arch = "x64"
    elif len(options) == 3:
        compiler = options[0]
        opt = options[2]
        arch = "x32"
    return (compiler, opt, arch)

def parse_quarks_opts(task):
    # Parse the task string to extract obfuscator, obfuscation type, and optimization level
    # Example: freetype/obfuscated/ollvm/opaque/100/freetype_ollvm_clang14_x64_opaque_100_5_O0.exe.json
    # Or sources: freetype/sources/freetype_clang14_x64_O0.exe
    parts = str(task).split("/")
    if len(parts) < 4:
        obfuscator = "None"
        obfuscation_type = "None"
    else:
        obfuscator = parts[2]
        obfuscation_type = parts[3]
    obfuscation_level = parts[-1].split(".")[0].split("_")[-1]
    return (obfuscator, obfuscation_type, obfuscation_level)

def average_result(args):
    gt_dir = pathlib.Path(args.gt_dir)
    output_dir = pathlib.Path(args.output_dir)
    pruned = {}
    for model in args.models:
        total_precision = defaultdict(float)
        total_recall = defaultdict(float)
        total_inst = defaultdict(int)
        average_precision = defaultdict(float)
        average_recall = defaultdict(float)
        model_tasks = list(output_dir.rglob(f"{model}.npz"))
        if args.num_samples and args.num_samples < len(model_tasks):
            import random
            random.seed(0)
            model_tasks = random.sample(model_tasks, args.num_samples)
        for task in model_tasks:
            data = np.load(task)
            rel_path = task.relative_to(output_dir).parent
            gt_path = gt_dir / (str(rel_path) + ".npz")
            gt_data = np.load(gt_path)
            parts = task.relative_to(output_dir).with_suffix("").as_posix().split("/")
            task_name = task.relative_to(output_dir).parent.as_posix()
            # if args.test_dataset == "x86_dataset":
            #     opt = parse_x86_sok_opts(task_name)
            # elif args.test_dataset == "quarks":
            #     opt = parse_quarks_opts(task_name)
            # elif args.test_dataset == "llvm-test-suite-gtirb":
            #     opt = str(parts[:2]) if "SPEC" not in parts else str([parts[0], parts[3] if not ('2006' in parts[3] or '2017' in parts[3]) else '2017' if '2017' in parts[3] else '2006'])
            # elif args.test_dataset == "rw":
            #     opt = parse_rw_opts(task_name)
            # else:
            #     opt = "all"
            opt = "all"
            total = np.sum(gt_data["mask"])
            total_precision[opt] += data["precision"] * total
            total_recall[opt] += data["recall"] * total
            total_inst[opt] += total
        print(f"Model: {model}, dataset: {args.test_dataset}")
        for opt in total_inst:
            average_precision[opt] = total_precision[opt] / total_inst[opt]
            average_recall[opt] = total_recall[opt] / total_inst[opt]
            F1 = 2 * average_precision[opt] * average_recall[opt] / (average_precision[opt] + average_recall[opt]) if (average_precision[opt] + average_recall[opt]) > 0 else 0
            print(f"{opt} F1: {F1:.4f}, Average Precision: {average_precision[opt]:.4f}, Average Recall: {average_recall[opt]:.4f}")
            pruned[f"{args.test_dataset}/{model}/{opt}"] = {
                "precision": average_precision[opt],
                "recall": average_recall[opt],
                "f1": F1,
            }
    return pruned

def average_before_prune(args):
    gt_dir = pathlib.Path(args.gt_dir)
    output_dir = pathlib.Path(args.eval_dir)
    before = {}
    for model in args.models:
        model_dir = output_dir / model
        model_tasks = list(model_dir.rglob("*.npz"))
        if args.num_samples and args.num_samples < len(model_tasks):
            import random
            random.seed(0)
            model_tasks = random.sample(model_tasks, args.num_samples)
        total_precision = defaultdict(float)
        total_recall = defaultdict(float)
        total_inst = defaultdict(int)
        average_precision = defaultdict(float)
        average_recall = defaultdict(float)
        for task in model_tasks:
            data = np.load(task)
            rel_path = task.relative_to(model_dir).parent
            parts = task.relative_to(model_dir).with_suffix("").as_posix().split("/")
            task_name = task.relative_to(model_dir).as_posix()
            gt_path = gt_dir / task_name
            gt_data = np.load(gt_path)
            # if args.test_dataset == "x86_dataset":
            #     opt = parse_x86_sok_opts(task_name)
            # elif args.test_dataset == "quarks":
            #     opt = parse_quarks_opts(task_name)
            # elif args.test_dataset == "llvm-test-suite-gtirb":
            #     opt = str(parts[:2]) if "SPEC" not in parts else str([parts[0], parts[3] if not ('2006' in parts[3] or '2017' in parts[3]) else '2017' if '2017' in parts[3] else '2006'])
            # elif args.test_dataset == "rw":
            #     opt = parse_rw_opts(task_name)
            # else:
            #     opt = "all"
            opt = "all"
            total = gt_data["mask"].sum()
            total_precision[opt] += data["precision"] * total
            total_recall[opt] += data["recall"] * total
            total_inst[opt] += total
        print(f"Model: {model}, dataset: {args.test_dataset}")
        for opt in total_inst:
            # print(f"Total: {total_inst[opt]}, Precision: {total_precision[opt]}, Recall: {total_recall[opt]}")
            average_precision[opt] = total_precision[opt] / total_inst[opt]
            average_recall[opt] = total_recall[opt] / total_inst[opt]
            F1 = 2 * average_precision[opt] * average_recall[opt] / (average_precision[opt] + average_recall[opt]) if (average_precision[opt] + average_recall[opt]) > 0 else 0
            print(f"{opt} F1: {F1:.4f}, Average Precision: {average_precision[opt]:.4f}, Average Recall: {average_recall[opt]:.4f}")
            before[f"{args.test_dataset}/{model}/{opt}"] = {
                "precision": average_precision[opt],
                "recall": average_recall[opt],
                "f1": F1,
            }
    return before

@hydra.main(version_base=None, config_path="conf", config_name="prune")
def main(args: DictConfig):
    gt_dir = pathlib.Path(args.gt_dir)
    files = [i for i in gt_dir.rglob("*.npz") if i.is_file()]
    if args.num_samples and args.num_samples < len(files):
        import random
        random.seed(0)
        files = random.sample(files, args.num_samples)
    tasks_gt = []
    for file in files:
        rel_path = file.relative_to(gt_dir)
        tasks_gt.append((args, file, rel_path))
    with Pool(args.process) as pool, tqdm(total = len(tasks_gt)) as pbar:
        for result in pool.imap_unordered(process_file, tasks_gt):
            pbar.update()
            pbar.refresh()
    pruned = average_result(args)
    before = average_before_prune(args)
    result = {
        "pruned": pruned,
        "before": before,
    }
    with open(pathlib.Path(args.output_dir) / "prune_result.json", "w") as f:
        json.dump(result, f)
if __name__ == "__main__":
    main()