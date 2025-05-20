from collections import defaultdict
import json
import pathlib
from multiprocessing.pool import ThreadPool as Pool
import grpc
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tady.utils.loader import len_to_overlappings, chunk_data
import tensorflow as tf
from transformers import PreTrainedTokenizerFast
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc
from concurrent.futures import ProcessPoolExecutor
import subprocess

import numpy as np
from typing import List
from tady import cpp

import os
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

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


def disassemble_batch(byte_chunks, use_64_bit):
    disassembler = cpp.Disassembler()
    instr_lens = []
    control_flows = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(
            chunks, use_64)
        instr_lens.append(instr_len)
        control_flows.append(control_flow)
    return np.array(instr_lens), np.array(control_flows)

def tokenize_batch(byte_chunks, use_64_bit, tokenizer):
    tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer, clean_up_tokenization_spaces=False)
    disassembler = cpp.Disassembler()
    instr_lens = []
    connections = []
    input_ids = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(chunks, use_64)
        overlappings = len_to_overlappings(instr_len)
        asms = disassembler.disasm_to_str(chunks, use_64, 0)
        res = tokenizer(asms, max_length=16, padding="max_length", truncation=True)
        input_ids.append(np.array(res["input_ids"], dtype=np.int32))
        instr_lens.append(np.array(res["attention_mask"]).sum(axis=-1, dtype=np.uint8))
        connections.append(np.concatenate([control_flow, overlappings], axis=-1, dtype=np.int32))
    return np.array(input_ids), np.array(instr_lens), np.array(connections)

def send_request(stub, model, byte_chunks, use_64_bit, disassembler, tokenizer=None):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = "serving_default"
    if disassembler == "jax":
        request.inputs["byte_sequence"].CopyFrom(
            tf.make_tensor_proto(byte_chunks)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
    elif disassembler == "cpp":
        request.inputs["byte_sequence"].CopyFrom(
            tf.make_tensor_proto(byte_chunks)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
        instr_lens, control_flows = disassemble_batch(byte_chunks, use_64_bit)
        request.inputs["instr_len"].CopyFrom(
            tf.make_tensor_proto(instr_lens)
        )
        request.inputs["control_flow"].CopyFrom(
            tf.make_tensor_proto(control_flows)
        )
    elif disassembler == "token":
        with ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tokenize_batch, byte_chunks, use_64_bit, tokenizer)
            input_ids, instr_lens, connections = future.result()
        request.inputs["input_ids"].CopyFrom(
            tf.make_tensor_proto(input_ids)
        )
        request.inputs["instr_len"].CopyFrom(
            tf.make_tensor_proto(instr_lens)
        )
        request.inputs["connections"].CopyFrom(
            tf.make_tensor_proto(connections)
        )
        request.inputs["use_64_bit"].CopyFrom(
            tf.make_tensor_proto(use_64_bit)
        )
    result = stub.Predict(request, 100)  # 10 secs timeout
    # Transform the result to a numpy array
    # Extract the first output tensor from the outputs map
    # Convert the result to a numpy array
    result = result.outputs['output_0']
    result = tf.make_ndarray(result)
    result = np.array(result)
    return result


def batchify(byte_chunks: np.ndarray, masks: np.ndarray, batch_size: int):
    # Batchify the byte chunks and masks
    batched_byte_chunks = []
    batched_masks = []
    for i in range(0, len(byte_chunks), batch_size):
        batch_byte_chunks = byte_chunks[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        if len(batch_byte_chunks) < batch_size:
            # Pad the batch with zeros
            pad_size = batch_size - len(batch_byte_chunks)
            batch_byte_chunks = np.pad(batch_byte_chunks, ((0, pad_size), (0, 0)), mode='constant', constant_values=0x90)
            batch_masks = np.pad(batch_masks, ((0, pad_size), (0, 0)), mode='constant', constant_values=False)
        batched_byte_chunks.append(np.array(batch_byte_chunks, dtype=np.uint8))
        batched_masks.append(np.array(batch_masks, dtype=np.bool))
    return batched_byte_chunks, batched_masks

def eval_tady(arg):
    args, file, output, model, stub = arg
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(output) / rel_path
        if output_path.exists():
            try:
                data = np.load(output_path)
            except Exception as e:
                print(f"Error loading {output_path}: {e}")
                raise e
            p, r, t = data["precision"].item(), data["recall"].item(), data["total"].item()
            return str(rel_path.with_suffix("")), {"precision": p, "recall": r, "total": t}
        # print(f"Disassembling {rel_path}")
        data = np.load(file)
        byte_chunks, masks = chunk_data(data["text_array"], args.model.seq_len, args.model.window_size)
        batched_byte_chunks, batched_masks = batchify(
            byte_chunks, masks, args.batch_size)
        logits = []
        for sequence, mask in zip(batched_byte_chunks, batched_masks):
            is_64_bit = np.array([data["use_64_bit"]] * len(sequence), dtype=np.bool)
            result = send_request(stub, model, sequence, is_64_bit, args.model.disassembler, args.tokenizer)
            logits.append(result[mask])
        logits = np.concatenate(logits, axis=0).flatten()
        pred = logits > args.threshold
        p = precision(pred, data["labels"], data["mask"])
        r = recall(pred, data["labels"], data["mask"])
        t = int(sum(data["mask"]))
        result = {
            "scores": logits,
            "pred": pred,
            "base_addr": np.array(data["base_addr"], dtype=np.uint64),
            "precision": np.array(p, dtype=np.float32),
            "recall": np.array(r, dtype=np.float32),
            "total": np.array(t, dtype=np.int32),
            "f1": f1(p, r),
            "fp": fp(pred, data["labels"], data["mask"], data["base_addr"]),
            "fn": fn(pred, data["labels"], data["mask"], data["base_addr"]),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        return str(rel_path.with_suffix("")), {"precision": p, "recall": r, "total": t}

def average_result(args, result):
    total_precision = defaultdict(float)
    total_recall = defaultdict(float)
    total_inst = defaultdict(int)
    average_precision = defaultdict(float)
    average_recall = defaultdict(float)
    for task, values in result.items():
        parts = task.split("/")
        # if args.test_dataset == "x86_dataset":
        #     opt = parse_x86_sok_opts(task)
        # elif args.test_dataset == "quarks":
        #     opt = parse_quarks_opts(task)
        # elif args.test_dataset == "llvm-test-suite-gtirb":
        #     opt = str(parts[:2]) if "SPEC" not in parts else str([parts[0], parts[3] if not ('2006' in parts[3] or '2017' in parts[3]) else '2017' if '2017' in parts[3] else '2006'])
        # elif args.test_dataset == "rw":
        #     opt = parse_rw_opts(task)
        # else:
        #     opt = "all"
        opt = "all"
        total_precision[opt] += values["precision"] * values["total"]
        total_recall[opt] += values["recall"] * values["total"]
        total_inst[opt] += values["total"]
    for opt in total_inst:
        average_precision[opt] = total_precision[opt] / total_inst[opt]
        average_recall[opt] = total_recall[opt] / total_inst[opt]
        F1 = 2 * average_precision[opt] * average_recall[opt] / (average_precision[opt] + average_recall[opt]) if (average_precision[opt] + average_recall[opt]) > 0 else 0
        print(f"{opt} F1: {F1:.4f}, Average Precision: {average_precision[opt]:.4f}, Average Recall: {average_recall[opt]:.4f}")

def process_tady(args, files, model):
    output = pathlib.Path(args.output) / model
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    results = {}
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for result in pool.imap_unordered(eval_tady, [(args, file, output, model, stub) for file in files]):
            if result is not None:
                rel_path, value = result
                results[rel_path] = value
            pbar.update()
            pbar.refresh()
    return results

def disassemble_ida(args, task):
    bin_dir = pathlib.Path(args.bin_dir)
    bin_path = bin_dir / task.with_suffix("")
    pred_dir = pathlib.Path(args.pred) / args.model_id 
    pred_path = pred_dir / task.with_suffix(".npz")
    if pred_path.exists():
        return np.load(pred_path)
    subprocess.run(["python", "-m", args.ida.script, "--dir", str(bin_dir), "--file", str(bin_path), "--output", str(pred_dir)])
    return np.load(pred_path)

def disassemble_ghidra(args, task):
    bin_dir = pathlib.Path(args.bin_dir)
    bin_path = bin_dir / task.with_suffix("")
    pred_dir = pathlib.Path(args.pred) / args.model_id 
    pred_path = pred_dir / task.with_suffix(".npz")
    if pred_path.exists():
        return np.load(pred_path)
    env = os.environ.copy()
    env["GHIDRA_INSTALL_DIR"] = args.ghidra.ghidra_root
    ret = subprocess.run(
        ["python", "-m", args.ghidra.script, "--dir", str(bin_dir), "--file", str(bin_path), "--output", str(pred_dir)],
        env=env,
        capture_output=True
    )
    if ret.returncode != 0:
        print(f"Error disassembling {task}: {ret.stderr}")
        return None
    return np.load(pred_path)

def disassemble_deepdi(args, task):
    bin_dir = pathlib.Path("/work/data/bin") / args.test_dataset
    bin_path = bin_dir / task.with_suffix("")
    np_path = pathlib.Path("/work/data/pred_strip") / args.test_dataset / args.model_id
    np_path_host = pathlib.Path(args.pred) / args.model_id / task.with_suffix(".npz")
    if np_path_host.exists():
        return np.load(np_path_host)
    cmd = [
        "docker",
        "exec",
        "-it",
        args.deepdi.container,
        "/bin/bash",
        "-c",
        f"PYTHONPATH=. python3 /work/scripts/baselines/DeepDi/DeepDiLief.py --gpu --dir {bin_dir} --file '{bin_path}' --output '{np_path}' --key {args.deepdi.key}"
    ]
    subprocess.run(cmd, capture_output=True)
    assert np_path_host.exists()
    return np.load(np_path_host)

def disassemble_deepdi_batch(args, tasks):
    bin_dir = pathlib.Path("/work/data/bin") / args.test_dataset
    bin_paths = [bin_dir / task.with_suffix("") for task in tasks]
    np_path = pathlib.Path("/work/data/pred_strip") / args.test_dataset / args.model_id
    np_paths = [np_path / task.with_suffix(".npz") for task in bin_paths]
    np_paths_host = [pathlib.Path(args.pred) / args.model_id / task for task in tasks]
    tasks_path = pathlib.Path("data/deepdi_tasks.json")
    with open(tasks_path, "w") as f:
        json.dump([str(task) for task in bin_paths], f)
    if all(np_path_host.exists() for np_path_host in np_paths_host):
        return
    cmd = [
        "docker",
        "exec",
        "-it",
        args.deepdi.container,
        "/bin/bash",
        "-c",
        f"PYTHONPATH=. python3 /work/scripts/baselines/DeepDi/DeepDiLief.py --gpu --dir {bin_dir} --output '{np_path}' --key {args.deepdi.key} --process {args.process} --task /work/data/deepdi_tasks.json"
    ]
    
    ret = subprocess.run(cmd, capture_output=True)
    if ret.returncode != 0:
        print(f"Error disassembling {tasks}: {ret.stderr}")
        exit(1)
        return None
    
def disassemble_ddisasm(args, task):
    bin_dir = pathlib.Path("/work/data/bin_strip") / args.test_dataset
    bin_path = bin_dir / task.with_suffix("")
    json_dir = pathlib.Path("/work/data/pred_strip") / args.test_dataset / args.model_id
    json_path_host = pathlib.Path(args.pred) / args.model_id / task.with_suffix(".json")
    if json_path_host.exists():
        with open(json_path_host, "r") as f:
            instructions = json.load(f)["instructions"]
            return instructions
    cmd = [
        "docker",
        "exec",
        "-it",
        args.ddisasm.container,
        "python3",
        "/work/scripts/baselines/ddisasm/batch_run.py",
        "--dir",
        str(bin_dir),
        "--file",
        str(bin_path),
        "--output",
        str(json_dir)
    ]
    subprocess.run(cmd, capture_output=True, timeout=60)  # 1 minute timeout
    assert json_path_host.exists()
    with open(json_path_host, "r") as f:
        instructions = json.load(f)["instructions"]
    return instructions

def disassemble_xda(args, task):
    bin_dir = pathlib.Path(args.bin_dir)
    bin_path = bin_dir / task.with_suffix("")
    pred_dir = pathlib.Path(args.pred) / args.model_id 
    pred_path = pred_dir / task.with_suffix(".npz")
    if pred_path.exists():
        return np.load(pred_path)
    subprocess.run([
        "bash",
        "-c",
        f"conda run -n {args.xda.conda_env} python {args.xda.script} --gpu --file {bin_path} --dir {bin_dir} --output {pred_dir} --model_path {args.xda.model_path} --dict_path {args.xda.dict_path}",    
    ],env={"PATH": os.environ["PATH"]},capture_output=True)
    return np.load(pred_path)

def eval_pred(arg):
    args, file, output, model = arg
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        bin_path = pathlib.Path(args.bin_dir) / rel_path.with_suffix("")
        output_path = pathlib.Path(output) / rel_path
        if output_path.exists():
            data = np.load(output_path)
            p, r, t = data["precision"].item(), data["recall"].item(), data["total"].item()
            return str(rel_path.with_suffix("")), {"precision": p, "recall": r, "total": t}
        data = np.load(file)
        match model:
            case "ida":
                pred = disassemble_ida(args, rel_path)
            case "ghidra":
                try:
                    pred = disassemble_ghidra(args, rel_path)
                    if pred is None:
                        return None
                except Exception as e:
                    print(f"Error disassembling {rel_path}: {e}")
                    return None
            case "deepdi":
                try:
                    pred = disassemble_deepdi(args, rel_path)
                    logits = pred["logits"]
                    logits = -np.log((1 / (min(logits + 1e-8, 1))) - 1)
                    logits = -np.log((1 / (min(logits + 1e-8, 1))) - 1)
                    pred = {"pred": pred["pred"], "logits": logits, "base_addr": pred["base_addr"]}
                except Exception as e:
                    print(f"Error disassembling {rel_path}: {e}")
                    return None
            case "xda":
                try:
                    pred = disassemble_xda(args, rel_path)
                except:
                    print(f"Error")
                    return None
            case "ddisasm":
                try:
                    instructions = disassemble_ddisasm(args, rel_path)
                except Exception as e:
                    print(f"Error disassembling {rel_path}: {e}")
                    return None
                points = np.array(instructions, dtype=np.uint64) - data["base_addr"]
                points = points[(points > 0) & (points < len(data["text_array"]))]
                pred = np.zeros(len(data["text_array"]), dtype=np.bool)
                pred[points] = True
                pred = {"pred": pred, "base_addr": data["base_addr"]}
            case _:
                raise ValueError(f"Model {model} not supported")
        p = precision(pred["pred"], data["labels"], data["mask"])
        r = recall(pred["pred"], data["labels"], data["mask"])
        t = int(sum(data["mask"]))
        result = {
            "pred": pred["pred"],
            "base_addr": np.array(pred["base_addr"], dtype=np.uint64),
            "precision": np.array(p, dtype=np.float32),
            "recall": np.array(r, dtype=np.float32),
            "total": np.array(t, dtype=np.int32),
            "f1": f1(p, r),
            "fp": fp(pred["pred"], data["labels"], data["mask"], data["base_addr"]),
            "fn": fn(pred["pred"], data["labels"], data["mask"], data["base_addr"]),
        }
        if "logits" in pred:
            result["scores"] = pred["logits"]
        else:
            result["scores"] = np.ones(len(data["text_array"]), dtype=np.float32) * -1
            result["scores"][pred["pred"]] = 1
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        return str(rel_path.with_suffix("")), {"precision": p, "recall": r, "total": t}

def process_pred(args, files, model):
    output = pathlib.Path(args.output) / model
    results = {}
    if model == "deepdi":
        rel_paths = [file.relative_to(args.dir) for file in files]
        print("Batch disassembling deepdi")
        disassemble_deepdi_batch(args, rel_paths)
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for result in pool.imap_unordered(eval_pred, [(args, file, output, model) for file in files]):
            if result is not None:
                rel_path, value = result
                results[rel_path] = value
            pbar.update()
            pbar.refresh()
    return results

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(args: DictConfig):
    root_dir = pathlib.Path(args.dir)
    model_id = "_".join([str(i) for i in args.tags])
    model = args.model_id if args.model_id else model_id
    print(f"Testing {model} on {root_dir}")
    files = [i for i in root_dir.rglob("*.npz") if i.is_file()]
    if args.num_samples and args.num_samples < len(files):
        import random
        random.seed(0)
        files = random.sample(files, args.num_samples)
    if args.model_id is None:
        results = process_tady(args, files, model)
    else:
        results = process_pred(args, files, model)
    stats_path = pathlib.Path(args.stats) / (model + ".json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(results, f)
    average_result(args, results)


if __name__ == "__main__":
    main()
