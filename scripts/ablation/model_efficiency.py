import json
import pathlib
import time
from multiprocessing.pool import ThreadPool as Pool
import grpc
import hydra
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm
from tady.utils.loader import preprocess_binary
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2, prediction_service_pb2_grpc

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any
from tady import cpp

disassembler = cpp.Disassembler()
def disassemble_batch(byte_chunks, use_64_bit):
    instr_lens = []
    control_flows = []
    for chunks, use_64 in zip(byte_chunks, use_64_bit):
        instr_len, _, control_flow, _ = disassembler.superset_disasm(
            chunks, use_64)
        instr_lens.append(instr_len)
        control_flows.append(control_flow)
    return np.array(instr_lens), np.array(control_flows)


def send_request(stub, model, byte_chunks, use_64_bit, disassembler, instr_lens=None, control_flows=None):
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model
    request.model_spec.signature_name = "serving_default"
    request.inputs["byte_sequence"].CopyFrom(
        tf.make_tensor_proto(byte_chunks)
    )
    request.inputs["use_64_bit"].CopyFrom(
        tf.make_tensor_proto(use_64_bit)
    )
    if disassembler == "cpp":
        
        request.inputs["instr_len"].CopyFrom(
            tf.make_tensor_proto(instr_lens)
        )
        request.inputs["control_flow"].CopyFrom(
            tf.make_tensor_proto(control_flows)
        )
    result = stub.Predict(request, 100)  # 10 secs timeout
    result = result.outputs['output_0']
    result = tf.make_ndarray(result)
    result = np.array(result)
    return result


def batchify(byte_chunks: np.ndarray, masks: np.ndarray, instr_lens: np.ndarray, control_flows: np.ndarray, batch_size: int):
    # Batchify the byte chunks and masks
    batched_byte_chunks = []
    batched_masks = []
    batched_instr_lens = []
    batched_control_flows = []
    for i in range(0, len(byte_chunks), batch_size):
        batch_byte_chunks = byte_chunks[i:i + batch_size]
        batch_masks = masks[i:i + batch_size]
        batch_instr_lens = instr_lens[i:i + batch_size]
        batch_control_flows = control_flows[i:i + batch_size]
        if len(batch_byte_chunks) < batch_size:
            # Pad the batch with zeros
            pad_size = batch_size - len(batch_byte_chunks)
            batch_byte_chunks = np.pad(batch_byte_chunks, ((0, pad_size), (0, 0)), mode='constant', constant_values=0x90)
            batch_masks = np.pad(batch_masks, ((0, pad_size), (0, 0)), mode='constant', constant_values=False)
            batch_instr_lens = np.pad(batch_instr_lens, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            batch_control_flows = np.pad(batch_control_flows, ((0, pad_size), (0, 0), (0, 0)), mode='constant', constant_values=0)
        batched_byte_chunks.append(np.array(batch_byte_chunks, dtype=np.uint8))
        batched_masks.append(np.array(batch_masks, dtype=np.bool))
        batched_instr_lens.append(np.array(batch_instr_lens, dtype=np.uint8))
        batched_control_flows.append(np.array(batch_control_flows, dtype=np.int32))
    return batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows

def process_file(args, file, model, stub, is_warmup=False, run_number=0):
    if not file.is_file():
        return None
        
    timings = {
        "file": str(file),
        "preprocess": 0,
        "inference": 0,
        "total": 0,
        "run_number": run_number
    }
    
    start_time = time.time()
    
    # Preprocess binary
    preprocess_start = time.time()
    byte_chunks, masks, use_64_bit, base_addr = preprocess_binary(file)
    batched_instr_lens = []
    batched_control_flows = []
    for sequence, mask in zip(byte_chunks, masks):
        is_64_bit = np.array([use_64_bit] * len(sequence), dtype=np.bool)
        if args.disassembler == "cpp":
            instr_lens, _, control_flows, _ = disassembler.superset_disasm(
                sequence, use_64_bit)
            batched_instr_lens.append(instr_lens)
            batched_control_flows.append(control_flows)
        else:
            batched_instr_lens.append(np.zeros((len(sequence),), dtype=np.uint8))
            batched_control_flows.append(np.zeros((len(sequence), 4), dtype=np.int32))
    batched_instr_lens = np.array(batched_instr_lens, dtype=np.uint8)
    batched_control_flows = np.array(batched_control_flows, dtype=np.int32)
    preprocess_end = time.time()
    timings["preprocess"] = preprocess_end - preprocess_start
    
    # Batchify
    batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows = batchify(
        byte_chunks, masks, batched_instr_lens, batched_control_flows, args.batch_size)
    
    # Inference
    inference_start = time.time()
    total_bytes = 0
    logits = []
    for sequence, mask, instr_lens, control_flows in zip(batched_byte_chunks, batched_masks, batched_instr_lens, batched_control_flows):
        is_64_bit = np.array([use_64_bit] * len(sequence), dtype=np.bool)
        total_bytes += sequence[mask].size
        result = send_request(stub, model, sequence, is_64_bit, args.disassembler, instr_lens, control_flows)
        logits.append(result[mask])
        
    inference_end = time.time()
    timings["inference"] = inference_end - inference_start
    
    # Total time
    end_time = time.time()
    timings["total"] = end_time - start_time
    timings["total_bytes"] = total_bytes
    timings["bytes_per_second"] = total_bytes / timings["total"] if timings["total"] > 0 else 0
    
    return None if is_warmup else timings


def plot_time_vs_size(benchmark_results, output_path=None):
    """
    Draw a scatter plot showing how processing time scales with processed file size.
    
    Args:
        benchmark_results: List of benchmark result dictionaries
        output_path: Path to save the plot image. If None, the plot will be displayed.
    """
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Extract data from benchmark results
    file_sizes = []
    total_times = []
    preprocess_times = []
    inference_times = []
    
    for result in benchmark_results:
        if "total_bytes" in result and "total" in result:
            file_sizes.append(result["total_bytes"])
            total_times.append(result["total"])
            preprocess_times.append(result["preprocess"])
            inference_times.append(result["inference"])
    
    # Convert to KB for better readability
    file_sizes_mb = [size / 1024.0 / 1024 for size in file_sizes]
    
    # Plot timing components
    ax.scatter(file_sizes_mb, total_times, alpha=0.6, label='Total Time')
    ax.scatter(file_sizes_mb, inference_times, alpha=0.6, label='Inference Time')
    ax.scatter(file_sizes_mb, preprocess_times, alpha=0.6, label='Preprocess Time')
    
    # Add trend lines
    if file_sizes_mb:
        # Fit linear trends to the data
        z_total = np.polyfit(file_sizes_mb, total_times, 1)
        p_total = np.poly1d(z_total)
        z_inference = np.polyfit(file_sizes_mb, inference_times, 1)
        p_inference = np.poly1d(z_inference)
        z_preprocess = np.polyfit(file_sizes_mb, preprocess_times, 1)
        p_preprocess = np.poly1d(z_preprocess)
        x_line = np.linspace(min(file_sizes_mb), max(file_sizes_mb), 100)
        ax.plot(x_line, p_total(x_line), '--', color='blue', 
                label=f'Total Time Trend: {z_total[0]:.2f}x + {z_total[1]:.2f}')
        ax.plot(x_line, p_inference(x_line), '--', color='orange', 
                label=f'Inference Time Trend: {z_inference[0]:.2f}x + {z_inference[1]:.2f}')
        ax.plot(x_line, p_preprocess(x_line), '--', color='green',
                label=f'Preprocess Time Trend: {z_preprocess[0]:.2f}x + {z_preprocess[1]:.2f}')
    
    # Set labels and title
    ax.set_xlabel('Size of code bytes (MB)')
    ax.set_ylabel('Time (seconds)')
    ax.set_title('Processing Time vs. File Size')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Save or show the plot
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark the model on a directory of binaries.")
    parser.add_argument("--samples", type=str, help="Samples stored in json for test.")
    parser.add_argument("--output", type=str, default="benchmark_results.json", help="Path to save the benchmark results.")
    parser.add_argument("--host", type=str, default="localhost", help="Host of the model server.")
    parser.add_argument("--port", type=int, default=8500, help="Port of the model server.")
    parser.add_argument("--model_id", type=str, help="Model ID.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for inference.")
    parser.add_argument("--plot", type=str, default=None, help="Path to save the time vs. size plot. If not specified, no plot will be generated.")
    parser.add_argument("--dir", type=str, help="Base directory for samples.")
    parser.add_argument("--disassembler", type=str, default="cpp", help="Disassembler to use.")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for binary classification.")
    parser.add_argument("--warmup", type=int, default=1, help="Number of warmup runs to perform.")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs to perform and average.")
    parser.add_argument("--noplot", action="store_true", help="Disable plotting even if --plot is provided.")
    
    args = parser.parse_args()
    
    channel = grpc.insecure_channel(f"{args.host}:{args.port}")
    model = args.model_id
    
    with open(args.samples, "r") as f:
        files = json.load(f)
    
    if isinstance(files, dict):
        files = [pathlib.Path(p) for p in files.keys()]
    else:
        files = [pathlib.Path(p) for p in files]
    
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    
    benchmark_results = []
    
    # Perform warmup runs
    if args.warmup > 0:
        print(f"\nPerforming {args.warmup} warmup run(s)...")
        for i in range(args.warmup):
            with tqdm(total=len(files), desc=f"Warmup Run {i+1}/{args.warmup}") as pbar:
                for file in files:
                    # Process each file but discard the results since it's a warmup
                    process_file(args, file, model, stub, is_warmup=True)
                    pbar.update()
    
    # Perform actual benchmark runs
    print(f"\nPerforming {args.runs} benchmark run(s)...")
    start_time = time.time()
    for run_number in range(args.runs):
        run_results = []
        with tqdm(total=len(files), desc=f"Benchmark Run {run_number+1}/{args.runs}") as pbar:
            for file in files:
                result = process_file(args, file, model, stub, is_warmup=False, run_number=run_number)
                if result:
                    run_results.append(result)
                pbar.update()
        benchmark_results.extend(run_results)
    end_time = time.time()
    
    # Calculate aggregate statistics
    total_time = end_time - start_time
    total_bytes = sum(result.get("total_bytes", 0) for result in benchmark_results)
    
    # Compute averages across all runs
    if benchmark_results:
        # Group results by run number
        runs_data = {}
        for r in benchmark_results:
            run_number = r.get("run_number", 0)
            if run_number not in runs_data:
                runs_data[run_number] = []
            runs_data[run_number].append(r)
        
        # Calculate metrics for each run
        run_metrics = []
        for run_number, results in runs_data.items():
            run_avg_preprocess = sum(r["preprocess"] for r in results) / len(results)
            run_avg_inference = sum(r["inference"] for r in results) / len(results)
            run_avg_total = sum(r["total"] for r in results) / len(results)
            run_total_bytes = sum(r.get("total_bytes", 0) for r in results)
            run_avg_bytes = run_total_bytes / len(results)
            
            run_metrics.append({
                "run_number": run_number,
                "preprocess_time": run_avg_preprocess,
                "inference_time": run_avg_inference, 
                "total_time": run_avg_total,
                "bytes_per_file": run_avg_bytes,
                "files_processed": len(results)
            })
        
        # Average metrics across all runs
        avg_preprocess = sum(m["preprocess_time"] for m in run_metrics) / len(run_metrics)
        avg_inference = sum(m["inference_time"] for m in run_metrics) / len(run_metrics)
        avg_total = sum(m["total_time"] for m in run_metrics) / len(run_metrics)
        avg_bytes = sum(m["bytes_per_file"] for m in run_metrics) / len(run_metrics)
        
        # Calculate standard deviations to measure variability between runs
        std_preprocess = np.std([m["preprocess_time"] for m in run_metrics]) if len(run_metrics) > 1 else 0
        std_inference = np.std([m["inference_time"] for m in run_metrics]) if len(run_metrics) > 1 else 0
        std_total = np.std([m["total_time"] for m in run_metrics]) if len(run_metrics) > 1 else 0
    else:
        run_metrics = []
    
    total_files_processed = sum(m["files_processed"] for m in run_metrics) if run_metrics else 0
    
    summary = {
        "total_files": len(files),
        "total_runs": len(run_metrics),
        "processed_files_per_run": len(files),
        "total_time_seconds": total_time,
        "total_bytes": total_bytes,
        "bytes_per_second": total_bytes / total_time if total_time > 0 else 0,
        "files_per_second": total_files_processed / total_time if total_time > 0 else 0,
        "averages": {
            "preprocess_time": avg_preprocess,
            "preprocess_time_std": std_preprocess,
            "inference_time": avg_inference,
            "inference_time_std": std_inference,
            "total_time": avg_total,
            "total_time_std": std_total,
            "bytes_per_file": avg_bytes
        },
        "run_metrics": run_metrics,
        "detailed_results": benchmark_results
    }
    
    print(f"\n==== Benchmark Results ====")
    print(f"Total files: {summary['total_files']}")
    print(f"Total runs: {summary['total_runs']} (after {args.warmup} warmup runs)")
    print(f"Total time: {summary['total_time_seconds']:.2f} seconds")
    print(f"Files per second: {summary['files_per_second']:.2f}")
    print(f"Total bytes processed: {summary['total_bytes']}")
    print(f"Processing speed: {summary['bytes_per_second']:.2f} bytes/sec")
    print(f"\nAverages per file (across all runs):")
    print(f"  Preprocessing time: {avg_preprocess:.4f} ± {std_preprocess:.4f} sec")
    print(f"  Inference time: {avg_inference:.4f} ± {std_inference:.4f} sec")
    print(f"  Total time: {avg_total:.4f} ± {std_total:.4f} sec")
    
    print(f"\nResults by run:")
    for i, run_metric in enumerate(run_metrics):
        print(f"  Run {i+1}:")
        print(f"    Files processed: {run_metric['files_processed']}")
        print(f"    Avg preprocessing time: {run_metric['preprocess_time']:.4f} sec")
        print(f"    Avg inference time: {run_metric['inference_time']:.4f} sec")
        print(f"    Avg total time: {run_metric['total_time']:.4f} sec")
    
    # Generate plot if requested and not disabled
    if args.plot and not args.noplot:
        print(f"\nGenerating time vs. size plot...")
        plot_time_vs_size(benchmark_results, args.plot)
    
    # Save benchmark results
    with open(args.output, 'w') as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
