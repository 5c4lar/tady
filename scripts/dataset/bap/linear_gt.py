import multiprocessing
import pathlib
from tady.utils.loader import load_text
from tady import cpp
import json
from tqdm import tqdm
import numpy as np

disassembler = cpp.Disassembler()
def process_file(arg):
    args, file_path = arg
    rel_path = file_path.relative_to(args.input)
    target_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
    text_array, use_64_bit, base_addr = load_text(file_path)
    offsets = disassembler.linear_disasm(text_array, use_64_bit).astype(np.int64)
    masks = np.zeros(text_array.shape[0], dtype=np.bool_)
    labels = np.zeros(text_array.shape[0], dtype=np.bool_)
    start = min(offsets)
    end = max(offsets)
    masks[start:end] = True
    labels[offsets] = True
    result = {
        "text_array": text_array,
        "labels": labels,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array([use_64_bit], dtype=np.bool_),
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

    files = [i for i in pathlib.Path(args.input).rglob("*") if i.is_file()]
    with multiprocessing.Pool(args.process) as pool, tqdm(total = len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            pbar.update()
            pbar.refresh()

if __name__ == "__main__":
    main()