import json
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
from scipy import stats
import pathlib
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# Consistent Matplotlib styling
FIGURE_DPI = 300
BASE_FONT_SIZE = 13
LINE_WIDTH = 1.5
MARKER_SIZE = 4

plt.rcParams.update({
    'font.size': BASE_FONT_SIZE,
    'axes.labelsize': BASE_FONT_SIZE,
    'xtick.labelsize': BASE_FONT_SIZE,
    'ytick.labelsize': BASE_FONT_SIZE,
    'legend.fontsize': BASE_FONT_SIZE,
    # 'figure.titlesize': BASE_FONT_SIZE + 2, # Not used as titles are removed
    'lines.linewidth': LINE_WIDTH,
    'lines.markersize': MARKER_SIZE,
    'font.family': "Times New Roman", # Using Times New Roman
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Define colors and markers for disassemblers (consistent with draw.py)
DISASSEMBLER_COLORS = {
    'ida': '#1f77b4',      # Blue
    'ghidra': '#ff7f0e',   # Orange  
    'xda': '#2ca02c',      # Green
    'deepdi': '#d62728',   # Red
    'ddisasm': '#9467bd',  # Purple
    'tady': '#8c564b'      # Brown
}

DISASSEMBLER_MARKERS = {
    'ida': 'o',
    'ghidra': 's', 
    'xda': '^',
    'deepdi': 'D',
    'ddisasm': 'v',
    'tady': 'X'
}

# --- Data Loading Functions ---

def load_benchmark_cache_data(filename="artifacts/benchmark_cache_figure8.json"):
    """Load benchmark data from benchmark_cache_figure8.json."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    disassemblers = defaultdict(list)
    for key, value in data.items():
        disasm_name = key.split('_')[0]
        if value.get('runtime') is not None and value.get('error') is None:
            disassemblers[disasm_name].append({
                'size_bytes': value['size_bytes'],
                'runtime': value['runtime'],
                'name': key
            })
    return disassemblers

def load_benchmark_results_data(filename="artifacts/benchmark_results_figure9.json"):
    """Load benchmark data from benchmark_results_figure9.json."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data.get("detailed_results", [])

def load_prune_data(filename="artifacts/prune_data_figure10_11.json"):
    """Load pruning benchmark data from prune_data_figure10_11.json."""
    with open(filename, 'r') as f:
        data = json.load(f)
    return data

# --- Plotting Functions ---

def plot_disassembler_runtime_scaling(disassemblers_data, linear_scale=False, output_suffix="", output_prefix=""):
    """Plot disassembler runtime vs. binary size."""
    plt.figure(figsize=(6, 4)) # Adjusted for potentially better fit without title
    
    for disasm_name, entries in disassemblers_data.items():
        if not entries:
            continue
            
        sizes_bytes = np.array([entry['size_bytes'] for entry in entries])
        # Convert sizes from bytes to MB
        sizes_mb = sizes_bytes / (1024.0 * 1024.0)
        runtimes = np.array([entry['runtime'] for entry in entries])
        
        color = DISASSEMBLER_COLORS.get(disasm_name, 'gray')
        marker = DISASSEMBLER_MARKERS.get(disasm_name, 'o')
        
        plt.scatter(sizes_mb, runtimes, 
                   color=color,
                   marker=marker,
                   label=disasm_name.upper(),
                   alpha=1.0,
                   s=30,
                   edgecolors='white',
                   linewidth=0.5)
        
        if len(sizes_mb) > 3:
            sorted_indices = np.argsort(sizes_mb)
            sorted_sizes_mb = sizes_mb[sorted_indices]
            sorted_runtimes = runtimes[sorted_indices]

            if linear_scale:
                try:
                    # Polynomial fit for linear scale
                    z = np.polyfit(sorted_sizes_mb, sorted_runtimes, 1) # Degree 2
                    p = np.poly1d(z)
                    size_range_mb = np.linspace(sorted_sizes_mb.min(), sorted_sizes_mb.max(), 100)
                    trend_runtimes = p(size_range_mb)
                    valid_trend = trend_runtimes >= 0
                    plt.plot(size_range_mb[valid_trend], trend_runtimes[valid_trend],
                            color=color, linestyle='--', alpha=1.0) # No label

                except np.linalg.LinAlgError: # Fallback for singular matrix
                     slope, intercept = np.polyfit(sorted_sizes_mb, sorted_runtimes, 1)
                     size_range_mb = np.linspace(sorted_sizes_mb.min(), sorted_sizes_mb.max(), 100)
                     trend_runtimes = slope * size_range_mb + intercept
                     plt.plot(size_range_mb, trend_runtimes, color=color, linestyle='--', alpha=1.0) # No label

            else: # Log scale
                # Filter out non-positive values before log transformation
                positive_mask = (sorted_sizes_mb > 0) & (sorted_runtimes > 0)
                if np.sum(positive_mask) > 1: # Need at least 2 points for linregress
                    log_sizes_mb = np.log10(sorted_sizes_mb[positive_mask])
                    log_runtimes = np.log10(sorted_runtimes[positive_mask])
                
                    if len(log_sizes_mb) > 1 and len(log_runtimes) > 1 and len(log_sizes_mb) == len(log_runtimes):
                        slope, intercept, r_value, p_value, std_err = stats.linregress(log_sizes_mb, log_runtimes)
                        
                        min_log_size_mb = np.log10(sorted_sizes_mb[positive_mask].min())
                        max_log_size_mb = np.log10(sorted_sizes_mb[positive_mask].max())
                        
                        if min_log_size_mb < max_log_size_mb : # Ensure range is valid
                            size_range_log_mb = np.logspace(min_log_size_mb, max_log_size_mb, 100)
                            trend_runtimes_log = 10**(slope * np.log10(size_range_log_mb) + intercept)
                            plt.plot(size_range_log_mb, trend_runtimes_log,
                                    color=color, linestyle='--', alpha=1.0) # No label

    plt.xlabel('Size of code bytes (MB)')
    plt.ylabel('Runtime (seconds)')
    
    if linear_scale:
        # plt.ticklabel_format(style='scientific', axis='x', scilimits=(0,0))
        pass
    else:
        plt.xscale('log')
        plt.yscale('log')
        
    plt.legend(loc='upper right', frameon=True, fancybox=True, shadow=False, ncol=1)
    plt.tight_layout(pad=0.5) # Add a little padding

    filename_base = f"{output_prefix}figure8_disassembler_runtime_scaling{'_linear' if linear_scale else '_log'}{output_suffix}"
    plt.savefig(f"{filename_base}.pdf", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(f"{filename_base}.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename_base}.pdf and {filename_base}.png")

def plot_model_efficiency(benchmark_results, output_suffix="", output_prefix=""):
    """Plot model processing time vs. file size."""
    plt.figure(figsize=(6, 4))
    
    file_sizes_bytes = []
    total_times = []
    preprocess_times = []
    inference_times = []
    
    for result in benchmark_results:
        if "total_bytes" in result and "total" in result and result["total_bytes"] > 0 and result["total"] > 0:
            file_sizes_bytes.append(result["total_bytes"])
            total_times.append(result["total"])
            preprocess_times.append(result["preprocess"])
            inference_times.append(result["inference"])

    if not file_sizes_bytes:
        print("No valid data for model efficiency plot.")
        plt.close()
        return

    file_sizes_mb = np.array([size / (1024.0 * 1024.0) for size in file_sizes_bytes])
    total_times = np.array(total_times)
    preprocess_times = np.array(preprocess_times)
    inference_times = np.array(inference_times)

    plt.scatter(file_sizes_mb, total_times, alpha=1.0, label='Total Time', marker='o')
    plt.scatter(file_sizes_mb, inference_times, alpha=1.0, label='Inference Time', marker='s')
    plt.scatter(file_sizes_mb, preprocess_times, alpha=1.0, label='Preprocess Time', marker='^')
    
    # Trend lines
    if len(file_sizes_mb) > 1:
        unique_sizes_mb, unique_indices = np.unique(file_sizes_mb, return_index=True) # Polyfit needs unique x
        
        if len(unique_sizes_mb) > 1:
            x_line = np.linspace(min(unique_sizes_mb), max(unique_sizes_mb), 100)
            text_y_start_ax_coord = 0.60 # Start y-position for text in axes coordinates
            text_y_offset_ax_coord = 0.10 # Vertical offset for subsequent lines of text

            for i, (data, color, style, data_label) in enumerate(zip(
                [total_times, inference_times, preprocess_times],
                ['tab:blue', 'tab:orange', 'tab:green'],
                ['-', '--', ':'],
                ['Total Time', 'Inference Time', 'Preprocess Time']
            )):
                if len(unique_sizes_mb) >= 2 : # Need at least 2 unique points for polyfit deg 1
                    z = np.polyfit(unique_sizes_mb, data[unique_indices], 1)
                    p = np.poly1d(z)
                    plt.plot(x_line, p(x_line), linestyle=style, color=color, alpha=1.0) # No label
                    # Add equation text in upper left, using axes coordinates
                    eq_text = f"y = {z[0]:.2f}x"
                    current_text_y_ax_coord = text_y_start_ax_coord - (i * text_y_offset_ax_coord)
                    plt.text(0.02, current_text_y_ax_coord, eq_text, 
                             transform=plt.gca().transAxes, 
                             fontsize=BASE_FONT_SIZE, # Slightly smaller to fit more lines if needed
                             color=color, 
                             ha='left', va='top')

    plt.xlabel('Size of code bytes (MB)')
    plt.ylabel('Time (seconds)')
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout(pad=0.5)

    filename_base = f"{output_prefix}figure9_model_efficiency{output_suffix}"
    plt.savefig(f"{filename_base}.pdf", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(f"{filename_base}.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename_base}.pdf and {filename_base}.png")

def plot_pruning_time_vs_size(prune_data, output_suffix="", output_prefix=""):
    """Plot pruning algorithm time components vs. code size."""
    plt.figure(figsize=(6, 4))
    
    sizes_mb = np.array(prune_data['sizes_mb'])
    disassemble_times = np.array(prune_data['disassemble_times'])
    pdt_times = np.array(prune_data['pdt_times'])
    prune_times = np.array(prune_data['prune_times'])
    error_times = np.array(prune_data['error_times'])

    plt.scatter(sizes_mb, pdt_times, label="PDT Construction", marker='s')
    plt.scatter(sizes_mb, prune_times, label="Pruning", marker='^')
    plt.scatter(sizes_mb, error_times, label="Error Calculation", marker='D')
    plt.scatter(sizes_mb, disassemble_times, label="Disassembly", marker='o')
    
    fit_x = np.linspace(min(sizes_mb), max(sizes_mb), 100)
    text_y_start_ax_coord_fig10 = 0.54
    text_y_offset_ax_coord_fig10 = 0.10
    
    for i, (data, color, style, data_label) in enumerate(zip(
        [pdt_times, prune_times, error_times, disassemble_times],
        ['tab:blue', 'tab:orange', 'tab:green', 'tab:red'],
        ['-', '--', ':', '-.'],
        ["PDT Construction", "Pruning", "Error Calculation", "Disassembly"]
    )):
        if len(sizes_mb) > 1:
            unique_sizes_mb, unique_indices = np.unique(sizes_mb, return_index=True)
            if len(unique_sizes_mb) > 1:
                 z = np.polyfit(unique_sizes_mb, data[unique_indices], 1)
                 p = np.poly1d(z)
                 plt.plot(fit_x, p(fit_x), color=color, linestyle=style, alpha=1.0) # No label
                 # Add equation text in upper left, using axes coordinates
                 eq_text = f"y = {z[0]:.2f}x"
                 current_text_y_ax_coord = text_y_start_ax_coord_fig10 - (i * text_y_offset_ax_coord_fig10)
                 plt.text(0.02, current_text_y_ax_coord, eq_text, 
                          transform=plt.gca().transAxes, 
                          fontsize=BASE_FONT_SIZE, 
                          color=color, 
                          ha='left', va='top')
            
    plt.xlabel("Size of code bytes (MB)")
    plt.ylabel("Time (seconds)")
    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout(pad=0.5)

    filename_base = f"{output_prefix}figure10_pruning_time_vs_size{output_suffix}"
    plt.savefig(f"{filename_base}.pdf", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(f"{filename_base}.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename_base}.pdf and {filename_base}.png")

def plot_pruning_peak_memory(prune_data, output_suffix="", output_prefix=""):
    """Plot peak memory usage vs. largest component size for pruning."""
    plt.figure(figsize=(6, 4))
    
    sizes_mb = np.array(prune_data['sizes_mb'])
    memories = np.array(prune_data['peak_memories_mb'])

    plt.scatter(sizes_mb, memories, label="Peak Memory", marker='o')
    
    if len(sizes_mb) > 1:
        unique_sizes_mb, unique_indices = np.unique(sizes_mb, return_index=True)
        if len(unique_sizes_mb) > 1:
            fit_x_memory = np.linspace(min(unique_sizes_mb), max(unique_sizes_mb), 100)
            z = np.polyfit(unique_sizes_mb, memories[unique_indices], 1)
            p = np.poly1d(z)
            plt.plot(fit_x_memory, p(fit_x_memory), color='tab:blue', linestyle='--', alpha=1.0) # No label

    plt.xlabel("Size of code bytes (MB)")
    plt.ylabel("Peak Memory (MB)")
    if len(sizes_mb)>0 and len(memories)>0:
        plt.xlim(0, max(sizes_mb) * 1.1 if len(sizes_mb) > 0 else 1)
        plt.ylim(0, max(memories) * 1.1 if len(memories) > 0 else 1)
    else:
        plt.xlim(0,1)
        plt.ylim(0,1)

    plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=False)
    plt.tight_layout(pad=0.5)

    filename_base = f"{output_prefix}figure11_pruning_peak_memory{output_suffix}"
    plt.savefig(f"{filename_base}.pdf", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.savefig(f"{filename_base}.png", dpi=FIGURE_DPI, bbox_inches='tight')
    plt.close()
    print(f"Generated {filename_base}.pdf and {filename_base}.png")

def main():
    # Ensure artifacts directory exists for output
    pathlib.Path("artifacts/generated_figures").mkdir(parents=True, exist_ok=True)
    output_prefix = "artifacts/generated_figures/"


    # --- Generate Figure 8 ---
    print("\n--- Generating Figure 8 variants ---")
    disassembler_data = load_benchmark_cache_data()
    # plot_disassembler_runtime_scaling(disassembler_data, linear_scale=False, output_suffix="_uniform", output_prefix=output_prefix)
    plot_disassembler_runtime_scaling(disassembler_data, linear_scale=True, output_suffix="_uniform", output_prefix=output_prefix)

    # --- Generate Figure 9 ---
    print("\n--- Generating Figure 9 ---")
    model_results_data = load_benchmark_results_data()
    plot_model_efficiency(model_results_data, output_suffix="_uniform", output_prefix=output_prefix)

    # --- Generate Figure 10 & 11 ---
    print("\n--- Generating Figures 10 and 11 ---")
    prune_data_fig10_11 = load_prune_data()
    plot_pruning_time_vs_size(prune_data_fig10_11, output_suffix="_uniform", output_prefix=output_prefix)
    plot_pruning_peak_memory(prune_data_fig10_11, output_suffix="_uniform", output_prefix=output_prefix)
    
    print("\nAll figures generated in artifacts/generated_figures/ with _uniform suffix.")

if __name__ == "__main__":
    main() 