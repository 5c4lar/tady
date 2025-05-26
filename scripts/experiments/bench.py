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
import time

import numpy as np
from typing import List
from tady import cpp

import os
import hashlib
os.environ["GRPC_ENABLE_FORK_SUPPORT"] = "false"

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

def time_disassemble_ida(args, bin_path):
    """Time IDA disassembly and return runtime in seconds"""
    pred_dir = pathlib.Path(args.pred) / "ida_runtime" 
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        ret = subprocess.run(
            ["python", "-m", args.ida.script, "--dir", str(bin_path.parent), "--file", str(bin_path), "--output", str(pred_dir)], 
            capture_output=True, 
            timeout=120  # 5 minute timeout
        )
        if ret.returncode != 0:
            return None, f"IDA failed: {ret.stderr.decode()}"
        end_time = time.time()
        return end_time - start_time, None
    except subprocess.TimeoutExpired:
        return None, "IDA timeout"
    except Exception as e:
        return None, f"IDA error: {str(e)}"

def time_disassemble_ghidra(args, bin_path):
    """Time Ghidra disassembly and return runtime in seconds"""
    pred_dir = pathlib.Path(args.pred) / "ghidra_runtime"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        env = os.environ.copy()
        env["GHIDRA_INSTALL_DIR"] = args.ghidra.ghidra_root
        ret = subprocess.run(
            ["python", "-m", args.ghidra.script, "--dir", str(bin_path.parent), "--file", str(bin_path), "--output", str(pred_dir)],
            env=env,
            capture_output=True,
            timeout=120  # 5 minute timeout
        )
        if ret.returncode != 0:
            return None, f"Ghidra failed: {ret.stderr.decode()}"
        end_time = time.time()
        return end_time - start_time, None
    except subprocess.TimeoutExpired:
        return None, "Ghidra timeout"
    except Exception as e:
        return None, f"Ghidra error: {str(e)}"

def time_disassemble_deepdi(args, bin_path):
    """Time DeepDi disassembly and return runtime in seconds"""
    bin_dir_docker = pathlib.Path("/work/data/bin") / args.test_dataset
    np_path = pathlib.Path("/work/data/pred_strip") / args.test_dataset / "deepdi_runtime"
    
    start_time = time.time()
    try:
        cmd = [
            "docker",
            "exec",
            "-it",
            args.deepdi.container,
            "/bin/bash",
            "-c",
            f"PYTHONPATH=. python3 /work/scripts/baselines/DeepDi/DeepDiLief.py --gpu --dir {bin_dir_docker} --file '{bin_path.name}' --output '{np_path}' --key {args.deepdi.key}"
        ]
        ret = subprocess.run(cmd, capture_output=True, timeout=120)
        if ret.returncode != 0:
            return None, f"DeepDi failed: {ret.stderr.decode()}"
        end_time = time.time()
        return end_time - start_time, None
    except subprocess.TimeoutExpired:
        return None, "DeepDi timeout"
    except Exception as e:
        return None, f"DeepDi error: {str(e)}"

def time_disassemble_ddisasm(args, bin_path):
    """Time ddisasm disassembly and return runtime in seconds"""
    bin_dir_docker = pathlib.Path("/work/data/bin_strip") / args.test_dataset
    json_dir = pathlib.Path("/work/data/pred_strip") / args.test_dataset / "ddisasm_runtime"
    print("Running ddisasm on", bin_path.name)
    bin_path_docker = pathlib.Path("/work/") / bin_path
    start_time = time.time()
    try:
        cmd = [
            # "docker",
            # "exec",
            # "-it",
            # args.ddisasm.container,
            # "timeout",
            # "--kill-after=5s",
            # "60s",  # 5 minute timeout
            # "python3",
            # "/work/scripts/baselines/ddisasm/batch_run.py",
            # "--dir",
            # str(bin_dir_docker),
            # "--file",
            # str(bin_path),
            # "--output",
            # str(json_dir)
            "docker",
            "exec",
            "-it",
            args.ddisasm.container,
            "timeout",
            "--kill-after=5s",
            "60s",  # 5 minute timeout
            "ddisasm",
            str(bin_path_docker),
        ]
        print(" ".join(cmd))
        ret = subprocess.run(cmd, capture_output=True, timeout=310)  # Slightly longer than docker timeout
        if ret.returncode != 0:
            return None, f"ddisasm failed: {ret.stderr.decode()}"
        end_time = time.time()
        return end_time - start_time, None
    except subprocess.TimeoutExpired:
        return None, "ddisasm timeout"
    except Exception as e:
        return None, f"ddisasm error: {str(e)}"

def time_disassemble_xda(args, bin_path):
    """Time XDA disassembly and return runtime in seconds"""
    pred_dir = pathlib.Path(args.pred) / "xda_runtime"
    pred_dir.mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    try:
        ret = subprocess.run([
            "bash",
            "-c",
            f"conda run -n {args.xda.conda_env} python {args.xda.script} --gpu --file {bin_path} --dir {bin_path.parent} --output {pred_dir} --model_path {args.xda.model_path} --dict_path {args.xda.dict_path}",    
        ], env={"PATH": os.environ["PATH"]}, capture_output=True, timeout=120)
        if ret.returncode != 0:
            return None, f"XDA failed: {ret.stderr.decode()}"
        end_time = time.time()
        return end_time - start_time, None
    except subprocess.TimeoutExpired:
        return None, "XDA timeout"
    except Exception as e:
        return None, f"XDA error: {str(e)}"

def time_disassemble_tady(args, bin_path, model, stub):
    """Time TADY disassembly and return runtime in seconds"""
    # Load the corresponding .npz file for this binary
    task_name = str(bin_path).replace(f"data/bin_strip/{args.test_dataset}/", "")
    data_path = pathlib.Path(args.dir) / (task_name + ".npz")
    print("Running TADY on", bin_path.name)
    if not data_path.exists():
        return None, f"Data file not found: {data_path}"
    
    start_time = time.time()
    try:
        data = np.load(data_path)
        byte_chunks, masks = chunk_data(data["text_array"], args.model.seq_len, args.model.sliding_window)
        batched_byte_chunks, batched_masks = batchify(byte_chunks, masks, args.batch_size)
        
        for sequence, mask in zip(batched_byte_chunks, batched_masks):
            is_64_bit = np.array([data["use_64_bit"]] * len(sequence), dtype=np.bool)
            result = send_request(stub, model, sequence, is_64_bit, args.model.disassembler, getattr(args, 'tokenizer', None))
        
        end_time = time.time()
        return end_time - start_time, None
    except Exception as e:
        return None, f"TADY error: {str(e)}"

def benchmark_single_disassembler(args, binaries, disasm_name, disasm_func, model=None, stub=None, cache_data=None, cache_file=None):
    """Benchmark a single disassembler on all binaries"""
    print(f"\n=== Running {disasm_name} on all binaries ===")
    
    if cache_data is None:
        cache_data = {}
    
    results = []
    binaries_to_process = []
    cached_success_count = 0
    cached_error_count = 0
    
    # Check which binaries need processing vs can use cache
    for bin_path in binaries:
        cache_key = get_cache_key(disasm_name, bin_path.name)
        cached_result = cache_data.get(cache_key)
        
        if is_cached_result_valid(cached_result, bin_path):
            # Use cached result (success or error)
            result = {
                "binary": str(bin_path.name),
                "size_bytes": bin_path.stat().st_size if bin_path.exists() else 0,
                "disassembler": disasm_name,
                "runtime": cached_result.get("runtime"),
                "error": cached_result.get("error"),
                "cached": True
            }
            results.append(result)
            
            if cached_result.get("runtime") is not None:
                cached_success_count += 1
            else:
                cached_error_count += 1
        else:
            # Need to process this binary
            binaries_to_process.append(bin_path)
    
    if cached_success_count > 0 or cached_error_count > 0:
        print(f"  Using {cached_success_count + cached_error_count} cached results ({cached_success_count} success, {cached_error_count} errors)")
    
    if not binaries_to_process:
        print(f"  All results cached, skipping processing")
        return results
    
    print(f"  Processing {len(binaries_to_process)} new binaries")
    
    # Process remaining binaries
    with Pool(args.process) as pool, tqdm(total=len(binaries_to_process), desc=f"{disasm_name}") as pbar:
        if disasm_name == 'tady':
            task_args = [(args, bin_path, disasm_name, disasm_func, model, stub) for bin_path in binaries_to_process]
        else:
            task_args = [(args, bin_path, disasm_name, disasm_func, None, None) for bin_path in binaries_to_process]
        
        for result in pool.imap_unordered(run_single_disassembler_on_binary, task_args):
            results.append(result)
            
            # Cache all results (both success and error) immediately
            if cache_file:
                bin_path = pathlib.Path(result["binary"])
                # Find the actual binary path from binaries_to_process
                for bp in binaries_to_process:
                    if bp.name == result["binary"]:
                        bin_path = bp
                        break
                
                cache_key = get_cache_key(disasm_name, result["binary"])
                cache_data[cache_key] = {
                    "runtime": result["runtime"],
                    "error": result["error"],
                    "binary_hash": get_binary_hash(bin_path),
                    "timestamp": time.time(),
                    "size_bytes": result["size_bytes"]
                }
                save_cache(cache_file, cache_data)
            
            pbar.update()
            pbar.refresh()
    
    return results

def run_single_disassembler_on_binary(arg):
    """Run a single disassembler on a single binary"""
    args, bin_path, disasm_name, disasm_func, model, stub = arg
    
    result = {
        "binary": str(bin_path.name),
        "size_bytes": bin_path.stat().st_size if bin_path.exists() else 0,
        "disassembler": disasm_name,
        "runtime": None,
        "error": None
    }
    
    try:
        if disasm_name == 'tady':
            runtime, error = disasm_func(args, bin_path, model, stub)
        else:
            runtime, error = disasm_func(args, bin_path)
        
        if runtime is not None:
            result["runtime"] = runtime
        else:
            result["error"] = error
            
    except Exception as e:
        result["error"] = f"Unexpected error: {str(e)}"
    
    return result

def read_binary_list(file_path):
    """Read list of binaries from file"""
    binaries = []
    with open(file_path, 'r') as f:
        data = json.load(f)
        for item in data:
            binaries.append(pathlib.Path(item))
    return binaries

def process_runtime_benchmark(args, binaries, model=None):
    """Process runtime benchmarking for all binaries, one disassembler at a time"""
    all_results = []
    
    # Setup cache
    output_dir = pathlib.Path(args.output)
    cache_file = output_dir / "benchmark_cache.json"
    cache_data = load_cache(cache_file)
    
    print(f"Loaded cache with {len(cache_data)} entries")
    
    # Setup TADY if model is provided
    stub = None
    if model:
        channel = grpc.insecure_channel(f"{args.host}:{args.port}")
        stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    # Define all disassemblers to test
    disassemblers = []
    
    # if hasattr(args, 'ida') and args.ida:
    #     disassemblers.append(('ida', time_disassemble_ida))
    # if hasattr(args, 'ghidra') and args.ghidra:
    #     disassemblers.append(('ghidra', time_disassemble_ghidra))
    if hasattr(args, 'deepdi') and args.deepdi:
        disassemblers.append(('deepdi', time_disassemble_deepdi))
    # if hasattr(args, 'ddisasm') and args.ddisasm:
    #     disassemblers.append(('ddisasm', time_disassemble_ddisasm))
    # if hasattr(args, 'xda') and args.xda:
    #     disassemblers.append(('xda', time_disassemble_xda))
    if model and stub:
        print("Testing TADY")
        disassemblers.append(('tady', time_disassemble_tady))
    
    # Run each disassembler on all binaries sequentially
    for disasm_name, disasm_func in disassemblers:
        disasm_results = benchmark_single_disassembler(args, binaries, disasm_name, disasm_func, model, stub, cache_data, cache_file)
        all_results.extend(disasm_results)
        
        # Print intermediate summary for this disassembler
        success_count = sum(1 for r in disasm_results if r["runtime"] is not None)
        cached_count = sum(1 for r in disasm_results if r.get("cached", False))
        cached_success_count = sum(1 for r in disasm_results if r.get("cached", False) and r["runtime"] is not None)
        cached_error_count = sum(1 for r in disasm_results if r.get("cached", False) and r["runtime"] is None)
        total_count = len(disasm_results)
        avg_time = np.mean([r["runtime"] for r in disasm_results if r["runtime"] is not None]) if success_count > 0 else 0
        
        print(f"{disasm_name} completed: {success_count}/{total_count} successful ({cached_success_count} cached success, {cached_error_count} cached errors), avg time: {avg_time:.2f}s")
    
    return all_results

def convert_results_to_original_format(all_results):
    """Convert the new result format back to the original format for compatibility"""
    # Group results by binary
    binary_results = defaultdict(lambda: {"runtimes": {}, "errors": {}})
    
    for result in all_results:
        binary_name = result["binary"]
        disasm_name = result["disassembler"]
        
        if "binary" not in binary_results[binary_name]:
            binary_results[binary_name]["binary"] = binary_name
            binary_results[binary_name]["size_bytes"] = result["size_bytes"]
        
        if result["runtime"] is not None:
            binary_results[binary_name]["runtimes"][disasm_name] = result["runtime"]
        else:
            binary_results[binary_name]["errors"][disasm_name] = result["error"]
    
    return list(binary_results.values())

def summarize_results(results):
    """Summarize runtime results"""
    # Handle both old and new result formats
    if results and "disassembler" in results[0]:
        # New format - convert to old format for summary
        results = convert_results_to_original_format(results)
    
    disassembler_stats = defaultdict(list)
    error_counts = defaultdict(int)
    
    for result in results:
        for disasm, runtime in result["runtimes"].items():
            disassembler_stats[disasm].append(runtime)
        for disasm, error in result["errors"].items():
            error_counts[disasm] += 1
    
    print("\n=== Runtime Summary ===")
    for disasm in sorted(disassembler_stats.keys()):
        runtimes = disassembler_stats[disasm]
        if runtimes:
            avg_time = np.mean(runtimes)
            median_time = np.median(runtimes)
            min_time = np.min(runtimes)
            max_time = np.max(runtimes)
            success_rate = len(runtimes) / (len(runtimes) + error_counts[disasm]) * 100
            
            print(f"{disasm}:")
            print(f"  Success rate: {success_rate:.1f}% ({len(runtimes)}/{len(runtimes) + error_counts[disasm]})")
            print(f"  Average time: {avg_time:.2f}s")
            print(f"  Median time: {median_time:.2f}s")
            print(f"  Min time: {min_time:.2f}s")
            print(f"  Max time: {max_time:.2f}s")
        else:
            print(f"{disasm}: No successful runs")

def get_binary_hash(bin_path):
    """Get a hash of the binary file for cache validation"""
    try:
        with open(bin_path, 'rb') as f:
            # Read first and last 1KB to create a quick hash
            first_chunk = f.read(1024)
            f.seek(-1024, 2)  # Seek to 1KB from end
            last_chunk = f.read(1024)
            content = first_chunk + last_chunk + str(bin_path.stat().st_size).encode()
            return hashlib.md5(content).hexdigest()
    except Exception:
        return None

def load_cache(cache_file):
    """Load existing benchmark cache"""
    if not cache_file.exists():
        return {}
    
    try:
        with open(cache_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Warning: Could not load cache file {cache_file}: {e}")
        return {}

def save_cache(cache_file, cache_data):
    """Save benchmark cache"""
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_file, 'w') as f:
            json.dump(cache_data, f, indent=2, default=str)
    except Exception as e:
        print(f"Warning: Could not save cache file {cache_file}: {e}")

def is_cached_result_valid(cache_entry, bin_path):
    """Check if cached result is still valid (for both success and error cases)"""
    if not cache_entry:
        return False
    
    # Check if binary still exists and hash matches
    if not bin_path.exists():
        return False
    
    cached_hash = cache_entry.get('binary_hash')
    current_hash = get_binary_hash(bin_path)
    
    return cached_hash == current_hash and cached_hash is not None

def get_cache_key(disasm_name, binary_name):
    """Get cache key for a disassembler-binary combination"""
    return f"{disasm_name}_{binary_name}"

@hydra.main(version_base=None, config_path="conf", config_name="eval")
def main(args: DictConfig):
    if not hasattr(args, 'binary_list') or not args.binary_list:
        print("Error: Please provide a binary_list file containing list of binaries to benchmark")
        return
    
    # Read list of binaries
    binaries = read_binary_list(args.binary_list)
    print(f"Found {len(binaries)} binaries to benchmark")
    
    # Handle cache clearing if requested
    output_dir = pathlib.Path(args.output)
    cache_file = output_dir / "benchmark_cache.json"
    
    if hasattr(args, 'clear_cache') and args.clear_cache:
        if cache_file.exists():
            cache_file.unlink()
            print("Cache cleared")
        else:
            print("No cache file to clear")
    
    # Setup model if running TADY
    model = None
    if hasattr(args, 'model_id') and args.model_id:
        model = args.model_id
        print(f"Will also benchmark TADY model: {model}")
    elif hasattr(args, 'tags') and args.tags:
        model = "_".join([str(i) for i in args.tags])
        print(f"Will also benchmark TADY model: {model}")
    
    # Run benchmarks (returns new format with per-disassembler results)
    all_results = process_runtime_benchmark(args, binaries, model)
    
    # Convert results to original format for compatibility
    results = convert_results_to_original_format(all_results)
    
    # Calculate cache statistics
    total_runs = len(all_results)
    cached_runs = sum(1 for r in all_results if r.get("cached", False))
    cached_success = sum(1 for r in all_results if r.get("cached", False) and r["runtime"] is not None)
    cached_errors = sum(1 for r in all_results if r.get("cached", False) and r["runtime"] is None)
    cache_hit_rate = (cached_runs / total_runs * 100) if total_runs > 0 else 0
    
    # Save detailed results in both formats
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save in original format for compatibility
    with open(output_dir / "runtime_benchmark.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save in new format with per-disassembler timing
    with open(output_dir / "runtime_benchmark_detailed.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    summarize_results(results)
    
    # Print cache statistics
    print(f"\n=== Cache Statistics ===")
    print(f"Total benchmark runs: {total_runs}")
    print(f"Cache hits: {cached_runs} ({cache_hit_rate:.1f}%)")
    print(f"  - Cached successes: {cached_success}")
    print(f"  - Cached errors: {cached_errors}")
    print(f"New runs: {total_runs - cached_runs}")
    
    print(f"\nResults saved to:")
    print(f"  Original format: {output_dir / 'runtime_benchmark.json'}")
    print(f"  Detailed format: {output_dir / 'runtime_benchmark_detailed.json'}")
    print(f"  Cache file: {output_dir / 'benchmark_cache.json'}")
    
    if cached_runs > 0:
        print(f"\nNext run will use cached results (including skipped errors).")
        print(f"Use 'clear_cache=true' to force fresh runs of all binaries.")


if __name__ == "__main__":
    main()
