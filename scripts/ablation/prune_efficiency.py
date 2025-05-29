from tady.utils.loader import load_text
from tady import cpp
import pathlib
import json
import numpy as np
import time
from matplotlib import pyplot as plt
import psutil
import threading

# Configure matplotlib for better paper plots
plt.rcParams.update({
    'font.size': 10,          # Base font size
    'axes.labelsize': 10,     # Axis label font size
    'axes.titlesize': 12,     # Title font size
    'xtick.labelsize': 9,     # X-axis tick label size
    'ytick.labelsize': 9,     # Y-axis tick label size
    'legend.fontsize': 9,     # Legend font size
    'figure.titlesize': 12,   # Figure title size
    'lines.linewidth': 1.5,   # Line width
    'lines.markersize': 4,    # Marker size
})

disassembler = cpp.Disassembler()


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
    pdt.prune(score["logits"])
    prune_end = time.time()
    prune_time = prune_end - prune_start
    print(f"Pruning time: {prune_time:.2f}s")
    
    error_start = time.time()
    pdt.get_errors(score["logits"])
    error_end = time.time()
    error_time = error_end - error_start
    print(f"Error calculation time: {error_time:.2f}s")
    return len(text_array), disassemble_time, pdt_time, prune_time, error_time, num_components, largest_size
    
    

def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    else:
        return obj

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Benchmark the pruning algorithm.")
    parser.add_argument("--samples", type=str, help="Samples stored in json for test.")
    parser.add_argument("--output", type=str, default="prune_time.pdf", help="Path to save the benchmark results.")
    parser.add_argument("--base_dir", type=str, default=".", help="Base directory for the samples.")
    parser.add_argument("--model_id", type=str, default="default_model", help="Model ID for the samples scores.")
    args = parser.parse_args()
    
    # Check if data files already exist
    output_data_json = args.output + '_data.json'
    output_data_npz = args.output + '_data.npz'
    
    if pathlib.Path(output_data_npz).exists():
        print(f"Loading existing data from {output_data_npz}")
        # Load existing data
        data = np.load(output_data_npz, allow_pickle=True)
        sizes = data['sizes_mb'].tolist()
        disassemble_times = data['disassemble_times'].tolist()
        pdt_times = data['pdt_times'].tolist()
        prune_times = data['prune_times'].tolist()
        error_times = data['error_times'].tolist()
        num_components = data['num_components'].tolist()
        largest_sizes = data['largest_sizes'].tolist()
        memories = data['peak_memories_mb'].tolist()
        samples = data['samples'].tolist()
        print(f"Loaded data for {len(samples)} samples")
    else:
        print("Processing samples and collecting data...")
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
            score_path = sample.replace(f"data/bin", f"artifacts/scores/")
            score_path = pathlib.Path(args.base_dir) / (score_path + ".npz")
            sample_path = pathlib.Path(args.base_dir) / sample
            if not score_path.exists():
                print(f"Score file {score_path} does not exist. Skipping sample {sample}.")
                continue
            else:
                print(f"Processing sample {sample} with score file {score_path}")
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
        
        # Save the collected data
        data_to_save = {
            'sizes_mb': sizes,
            'disassemble_times': disassemble_times,
            'pdt_times': pdt_times,
            'prune_times': prune_times,
            'error_times': error_times,
            'num_components': num_components,
            'largest_sizes': largest_sizes,
            'peak_memories_mb': memories,
            'samples': samples
        }
        
        # Convert numpy types for JSON serialization
        data_to_save_json = convert_numpy_types(data_to_save)
        
        # Save as JSON
        with open(output_data_json, 'w') as f:
            json.dump(data_to_save_json, f, indent=2)
        print(f"Data saved to {output_data_json}")
    
    print("Generating plots...")
    
    # Plot the results - optimized for double column papers
    fig, ax = plt.subplots(figsize=(5.0, 3.5))  # Larger size for better visibility
    
    # Scatter plots with shorter legend labels
    ax.scatter(sizes, disassemble_times, label="Disassembly", marker='o')
    ax.scatter(sizes, pdt_times, label="PDT Construction", marker='o')
    ax.scatter(sizes, prune_times, label="Pruning", marker='o')
    ax.scatter(sizes, error_times, label="Error Calculation", marker='o')
    
    # Fit curves using polynomial regression
    sizes_array = np.array(sizes)
    fit_x = np.linspace(min(sizes), max(sizes), 100)
    
    # Fit and plot each curve
    for data, label, color in zip(
        [disassemble_times, pdt_times, prune_times, error_times],
        ["Disassembly", "PDT Construction", "Pruning", "Error Calculation"],
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    ):
        if len(sizes) > 1:  # Only fit if we have enough data points
            z = np.polyfit(sizes, data, 1)
            p = np.poly1d(z)
            # Simplified legend without detailed equations for cleaner look
            ax.plot(fit_x, p(fit_x), color=color, linestyle='--', alpha=0.7)
            

    ax.set_xlabel("Size of code bytes (MB)")
    ax.set_ylabel("Time (seconds)")
    ax.set_title("Benchmarking Pruning Algorithm")
    # Position legend inside the plot area with smaller font
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f"{args.output}_time.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
    
    # Plot the memory usage
    # Since we cannot enfore the release of memory in python, lets plot the points only when 
    # the largest size is larger than the largest size of the previous point
    # This is a hack to make the plot look better
    
    largest_sizes = np.array(sizes)
    memories = np.array(memories)
    # Plot the points
    fig, ax = plt.subplots(figsize=(5.0, 3.5))  # Larger size for better visibility
    ax.scatter(largest_sizes, memories, label="Peak Memory", marker='o')
    # Fit and plot the curve
    if len(largest_sizes) > 1:  # Only fit if we have enough data points
        fit_x_memory = np.linspace(min(largest_sizes), max(largest_sizes), 100)
        z = np.polyfit(largest_sizes, memories, 1)
        p = np.poly1d(z)
        # Simplified legend without detailed equation for cleaner look
        ax.plot(fit_x_memory, p(fit_x_memory), color='tab:blue', linestyle='--', alpha=0.7, label="Trend")
    ax.set_xlabel("Size of code bytes (MB)")
    ax.set_ylabel("Peak Memory (MB)")
    ax.set_title("Peak Memory Usage")
    ax.set_xlim(0, max(largest_sizes) * 1.1)
    ax.set_ylim(0, max(memories) * 1.1)
    # Position legend inside the plot area with smaller font
    ax.legend(loc='upper left', fontsize=9, framealpha=0.9)
    plt.tight_layout()
    fig.savefig(f"{args.output}_peak_memory.pdf", bbox_inches='tight', pad_inches=0.1, dpi=300)
        
if __name__ == "__main__":
    main()
