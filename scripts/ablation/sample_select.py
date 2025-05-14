import pathlib
import lief
from multiprocessing import Pool

def get_text_section_size(elf_path):
    """
    Get the size of the .text section in an ELF file.
    """
    binary = lief.parse(elf_path)
    text_section = binary.get_section('.text')
    if text_section:
        return text_section.size
    else:
        raise ValueError(f"No .text section found in {elf_path}")
    
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Get the size of the .text section in an ELF file.")
    parser.add_argument("--dir", type=str, help="Path to the root directory.")
    args = parser.parse_args()
    root_dir = pathlib.Path(args.dir)
    tasks = []
    for path in root_dir.rglob("*"):
        if path.is_file():
            tasks.append(path)
    with Pool() as pool:
        results = pool.map(get_text_section_size, tasks)
    
    # Sort the binary by size
    sorted_results = sorted(zip(tasks, results), key=lambda x: x[1], reverse=True)
    import matplotlib.pyplot as plt
    # plot the distribution of the size
    sizes = [size for _, size in sorted_results]
    plt.hist(sizes, bins=100)
    plt.xlabel("Size of .text section (bytes)")
    plt.ylabel("Number of binaries")
    plt.title("Distribution of .text section size in ELF files")
    plt.savefig("artifacts/text_section_size_distribution.pdf")

    # Sample one binary from each size range with power of 2 steps
    min_size = min(sizes)
    max_size = max(sizes)
    
    # Create power-of-2 size buckets
    power = 1
    samples = []
    while power <= max_size:
        # Find binaries that fall within this power-of-2 bucket
        lower_bound = power
        upper_bound = power * 2
        
        # Get candidates in this range
        candidates = [(path, size) for path, size in sorted_results 
                     if lower_bound <= size < upper_bound]
        
        if candidates:
            # Take up to 3 samples evenly distributed across the bucket
            num_samples = min(3, len(candidates))
            if num_samples == 1:
                sample_indices = [0]
            else:
                step = (len(candidates) - 1) / (num_samples - 1) if num_samples > 1 else 0
                sample_indices = [int(i * step) for i in range(num_samples)]
                
                bucket_samples = [candidates[i] for i in sample_indices]
                samples.extend(bucket_samples)
                print(f"Selected {num_samples} samples from bucket {lower_bound}-{upper_bound}")
            for sample in bucket_samples:
                print(f"  - {sample[0]} with size {sample[1]} bytes")
            
        power *= 2
    
    print(f"Selected {len(samples)} samples across size distribution")
    
    # sort samples by size
    samples = sorted(samples, key=lambda x: x[1], reverse=False)
    # Print paths of selected samples for further processing
    print("Sample paths:")
    for path, size in samples:
        print(f"{path} : {size} bytes")
    import json
    # Save the selected samples to a JSON file
    with open("artifacts/selected_samples.json", "w") as f:
        json.dump({str(key): value for key, value in samples}, f, indent=4)
if __name__ == "__main__":
    main()