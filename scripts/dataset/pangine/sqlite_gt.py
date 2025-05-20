import multiprocessing
import pathlib
from tady.utils.loader import load_text
import json
from tqdm import tqdm
import numpy as np
import sqlite3

def process_file(arg):
    args, file_path = arg
    gt_path = str(file_path).replace("/bin/", "/gt/") + ".sqlite"
    rel_path = file_path.relative_to(args.input)
    target_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
    conn = sqlite3.connect(gt_path)
    cursor = conn.cursor()
    try:
        # Fetch instruction offsets and their supplementary info
        cursor.execute("SELECT offset, supplementary FROM insn ORDER BY offset")
        all_instructions = []
        optional_ranges = []
        last_optional = None
        
        for offset, supp in cursor.fetchall():
            is_optional = False
            if supp:
                supp_dict = json.loads(supp)
                is_optional = supp_dict.get("Optional", False)
            
            if is_optional:
                last_optional = offset
            else:
                if last_optional is not None:
                    optional_ranges.append((last_optional, offset))
                    last_optional = None
            all_instructions.append(offset)
        
        # Fetch function ranges
        cursor.execute("SELECT start, end FROM func")
        function_ranges = [(start, end) for start, end in cursor.fetchall()]
    except:
        return
    finally:
        conn.close()
    
    text_array, use_64_bit, base_addr = load_text(file_path)
    labels = np.zeros(text_array.shape[0], dtype=np.bool_)
    masks = np.zeros(text_array.shape[0], dtype=np.bool_)
    
    # Convert instruction offsets to relative offsets
    points = np.array(all_instructions, dtype=np.uint64)
    points = points[(points >= base_addr) & (points < base_addr + text_array.shape[0])]
    points = points - base_addr
    
    # Set labels for instruction starts (excluding optional ones)
    labels[points] = True
    
    # Create masks for function ranges
    for start, end in function_ranges:
        # Convert function boundaries to relative offsets
        rel_start = start - base_addr
        rel_end = end - base_addr
        
        # Ensure boundaries are within array bounds
        rel_start = max(0, rel_start)
        rel_end = min(text_array.shape[0], rel_end)
        
        if rel_start < rel_end:
            masks[rel_start:rel_end] = True
    
    # Mark optional instruction ranges as false in masks
    for start, end in optional_ranges:
        rel_start = start - base_addr
        rel_end = end - base_addr
        
        # Ensure boundaries are within array bounds
        rel_start = max(0, rel_start)
        rel_end = min(text_array.shape[0], rel_end)
        
        if rel_start < rel_end:
            masks[rel_start:rel_end] = False
    
    result = {
        "text_array": text_array,
        "labels": labels,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array(use_64_bit, dtype=np.bool_),
        "mask": masks,
    }
    target_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(target_path, **result)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--process", type=int, default=1)
    args = parser.parse_args()

    files = [i for i in pathlib.Path(args.input).rglob("*") if i.is_file() and i.suffix != ".sqlite"]
    with multiprocessing.Pool(args.process) as pool, tqdm(total = len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            pbar.update()
            pbar.refresh()

if __name__ == "__main__":
    main()