import json
import argparse
from collections import defaultdict
import hydra
import pathlib
from omegaconf import DictConfig, OmegaConf

def parse_rw_opts(task):
    parts = task.split("/")
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
    parts = task.split("/")
    if len(parts) < 4:
        obfuscator = "None"
        obfuscation_type = "None"
    else:
        obfuscator = parts[2]
        obfuscation_type = parts[3]
    obfuscation_level = parts[-1].split(".")[0].split("_")[-1]
    return (obfuscator, obfuscation_type, obfuscation_level)

@hydra.main(version_base=None, config_path="conf", config_name="average")
def main(args: DictConfig):
    model_id = "_".join([str(i) for i in args.tags]) if args.model_id is None else args.model_id 
    print(model_id)
    if args.prune:
        input_path = (pathlib.Path(args.input) / (model_id + "_pruned.json"))
    else:
        input_path = (pathlib.Path(args.input) / (model_id + ".json"))
    result = json.load(open(input_path, "r"))
    total_precision = defaultdict(float)
    total_recall = defaultdict(float)
    total_inst = defaultdict(int)
    average_precision = defaultdict(float)
    average_recall = defaultdict(float)
    for task, values in result.items():
        if "m32" in task:
            continue
        # parts = task.split("/")
        # opt = str(parts[:2]) if "SPEC" not in parts else str([parts[0], parts[3]])
        parts = task.split("/")
        if args.test_dataset == "x86_dataset":
            opt = parse_x86_sok_opts(task)
        elif args.test_dataset == "quarks":
            opt = parse_quarks_opts(task)
        elif args.test_dataset == "llvm-test-suite-gtirb":
            opt = str(parts[:2]) if "SPEC" not in parts else str([parts[0], parts[3] if not ('2006' in parts[3] or '2017' in parts[3]) else '2017' if '2017' in parts[3] else '2006'])
        elif args.test_dataset == "rw":
            opt = parse_rw_opts(task)
        else:
            opt = "all"
        total_precision[opt] += values["precision"] * values["total"]
        total_recall[opt] += values["recall"] * values["total"]
        total_inst[opt] += values["total"]
    for opt in total_inst:
        average_precision[opt] = total_precision[opt] / total_inst[opt]
        average_recall[opt] = total_recall[opt] / total_inst[opt]
        F1 = 2 * average_precision[opt] * average_recall[opt] / (average_precision[opt] + average_recall[opt]) if (average_precision[opt] + average_recall[opt]) > 0 else 0
        print(f"{opt} F1: {F1:.4f}, Average Precision: {average_precision[opt]:.4f}, Average Recall: {average_recall[opt]:.4f}")
        # print(f"{opt} Average Precision: {average_precision[opt]:.4f}, Average Recall: {average_recall[opt]:.4f}")

if __name__ == "__main__":
    main()