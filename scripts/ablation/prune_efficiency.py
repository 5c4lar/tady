from tady.utils.loader import load_text
from tady import cpp
import pathlib
import json
import numpy as np
import time
from matplotlib import pyplot as plt
import psutil
import threading
disassembler = cpp.Disassembler("x86_64")


def measure_peak_memory(func, *args, **kwargs):
    process = psutil.Process()
    peak_memory = {"max_rss": 0}
    stop_event = threading.Event()

    def monitor():
        while not stop_event.is_set():
            mem = process.memory_info().rss  # in bytes
            peak_memory["max_rss"] = max(peak_memory["max_rss"], mem)
            time.sleep(0.01)  # polling interval

    monitor_thread = threading.Thread(target=monitor)
    monitor_thread.start()

    try:
        result = func(*args, **kwargs)
    finally:
        stop_event.set()
        monitor_thread.join()

    peak_in_mb = peak_memory["max_rss"] / (1024 ** 2)
    return result, peak_in_mb

def process_sample(sample, score_path):
    """
    Process a single sample to extract the relevant information.
    """
    text_array, use_64_bit, base_addr = load_text(sample)
    disassemble_start = time.time()
    instr_len, flow_kind, control_flow, successors = disassembler.superset_disasm(text_array, use_64_bit)
    disassemble_end = time.time()
    disassemble_time = disassemble_end - disassemble_start
    cf = flow_kind > 1
    pdt_start = time.time()
    pdt = cpp.PostDominatorTree(successors, cf)
    pdt_end = time.time()
    pdt_time = pdt_end - pdt_start
    print(f"Disassembly time: {disassemble_time:.2f}s, PDT construction time: {pdt_time:.2f}s")
    components_size = pdt.get_components_size()
    num_components = len(components_size)
    largest_size = max(components_size)
    print(f"Number of components: {num_components}, Largest component size: {largest_size}")
    score = np.load(score_path)
    prune_start = time.time()
    pdt.prune(score)
    prune_end = time.time()
    prune_time = prune_end - prune_start
    print(f"Pruning time: {prune_time:.2f}s")
    
    error_start = time.time()
    pdt.get_errors(score)
    error_end = time.time()
    error_time = error_end - error_start
    print(f"Error calculation time: {error_time:.2f}s")
    return len(text_array), disassemble_time, pdt_time, prune_time, error_time, num_components, largest_size
    
    

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark the pruning algorithm.")
    parser.add_argument("--samples", type=str, help="Samples stored in json for test.")
    parser.add_argument("--output", type=str, default="prune_time.pdf", help="Path to save the benchmark results.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory for the samples.")
    parser.add_argument("--model_id", type=str, default="default_model", help="Model ID for the samples scores.")
    args = parser.parse_args()
    
    # Load the samples
    samples = json.load(open(args.samples, "r"))
    sizes = []
    disassemble_times = []
    pdt_times = []
    prune_times = []
    error_times = []
    num_components = []
    largest_sizes = []
    memories = []
    for sample in samples:
        ds_name = sample.split("/")[2]
        score_path = sample.replace(f"data/bin/{ds_name}", f"data/results/{ds_name}/{args.model_id}")
        score_path = pathlib.Path(args.base_dir) / score_path  / "score.npy"
        sample_path = pathlib.Path(args.base_dir) / sample
        if not score_path.exists():
            print(f"Score file {score_path} does not exist. Skipping sample {sample}.")
            continue
        (size, disassemble_time, pdt_time, prune_time, error_time, num_component, largest_size), peak_memory = measure_peak_memory(process_sample, sample_path, score_path)
        print(f"Peak memory: {peak_memory:.2f}MB")
        memories.append(peak_memory)
        sizes.append(size / 1024 / 1024)
        disassemble_times.append(disassemble_time)
        pdt_times.append(pdt_time)
        prune_times.append(prune_time)
        error_times.append(error_time)
        num_components.append(num_component)
        largest_sizes.append(largest_size)
    # Plot the results
    plt.figure(figsize=(10, 6))
    
    # Scatter plots
    plt.scatter(sizes, disassemble_times, label="Disassembly Time", marker='o')
    plt.scatter(sizes, pdt_times, label="PDT Construction Time", marker='o')
    plt.scatter(sizes, prune_times, label="Pruning Time", marker='o')
    plt.scatter(sizes, error_times, label="Error Calculation Time", marker='o')
    
    # Fit curves using polynomial regression
    sizes_array = np.array(sizes)
    fit_x = np.linspace(min(sizes), max(sizes), 100)
    
    # Fit and plot each curve
    for data, label, color in zip(
        [disassemble_times, pdt_times, prune_times, error_times],
        ["Disassembly Time", "PDT Construction Time", "Pruning Time", "Error Calculation Time"],
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    ):
        if len(sizes) > 1:  # Only fit if we have enough data points
            z = np.polyfit(sizes, data, 1)
            p = np.poly1d(z)
            plt.plot(fit_x, p(fit_x), color=color, linestyle='--', alpha=0.7, label=f"{label}: {z[0]:.2f}x + {z[1]:.2f}")
            

    plt.xlabel("Size of code bytes (MB)")
    plt.ylabel("Time (seconds)")
    plt.title("Benchmarking Pruning Algorithm")
    plt.legend()
    plt.savefig(f"{args.output}_time.pdf")
    
    # Plot the memory usage
    # Since we cannot enfore the release of memory in python, lets plot the points only when 
    # the largest size is larger than the largest size of the previous point
    # This is a hack to make the plot look better
    
    largest_sizes = np.array(largest_sizes)
    memories = np.array(memories)
    mask = np.zeros_like(largest_sizes, dtype=bool)
    mask[0] = True
    current_largest = largest_sizes[0]
    for i in range(1, len(largest_sizes)):
        if largest_sizes[i] > current_largest:
            mask[i] = True
            current_largest = largest_sizes[i]
    # Plot the points
    largest_sizes = largest_sizes[mask] / 1e6
    memories = memories[mask]
    # Plot the points
    plt.figure(figsize=(10, 6))
    plt.scatter(largest_sizes, memories, label="Peak Memory", marker='o')
    # Fit and plot the curve
    if len(largest_sizes) > 1:  # Only fit if we have enough data points
        z = np.polyfit(largest_sizes, memories, 1)
        p = np.poly1d(z)
        plt.plot(fit_x, p(fit_x), color='tab:blue', linestyle='--', alpha=0.7, label=f"Peak Memory: {z[0]:.2f}x + {z[1]:.2f}")
    plt.xlabel("Largest component size (1e6)")
    plt.ylabel("Peak Memory (MB)")
    plt.title("Peak Memory Usage")
    plt.xlim(0, max(largest_sizes) * 1.1)
    plt.ylim(0, max(memories) * 1.1)
    plt.legend()
    plt.savefig(f"{args.output}_peak_memory.pdf")
        
if __name__ == "__main__":
    main()
