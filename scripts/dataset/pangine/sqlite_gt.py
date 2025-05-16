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
    target_path = pathlib.Path(args.output) / (str(rel_path) + ".json")
    conn = sqlite3.connect(gt_path)
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT offset FROM insn")
        instructions = [i[0] for i in cursor.fetchall()]
    except:
        return
    finally:
        conn.close()
    text_array, use_64_bit, base_addr = load_text(file_path)
    labels = np.zeros(text_array.shape[0], dtype=np.bool_)
    masks = np.zeros(text_array.shape[0], dtype=np.bool_)
    points = np.array(instructions, dtype=np.uint64)
    points = points[(points >= base_addr) & (points < base_addr + text_array.shape[0])]
    start = min(points) - base_addr
    end = max(points) - base_addr
    masks[start:end] = True
    labels[points - base_addr] = True
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