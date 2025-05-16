import multiprocessing
import pathlib
from tady.utils.loader import load_text
from tady import cpp
import json
from tqdm import tqdm
import numpy as np
import lief

disassembler = cpp.Disassembler()
def process_file(arg):
    args, file_path = arg
    rel_path = file_path.relative_to(args.input)
    target_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
    text_array, use_64_bit, base_addr = load_text(file_path)
    binary = lief.parse(file_path)
    offsets = []
    masks = np.zeros(text_array.shape[0], dtype=np.bool_)
    for function in binary.functions:
        if function.address >= base_addr and function.address < base_addr + text_array.shape[0]:
            function_data = np.frombuffer(binary.get_content_from_virtual_address(function.address, function.size), dtype=np.uint8)
            function_offsets = disassembler.linear_disasm(function_data, use_64_bit).astype(np.int64) + (function.address - base_addr)
            offsets.extend(function_offsets.tolist())
            masks[function.address - base_addr:function.address - base_addr + function.size] = True
    labels = np.zeros(text_array.shape[0], dtype=np.bool_)
    offsets = np.array(offsets, dtype=np.int64)
    offsets = offsets[(offsets >= 0) & (offsets < text_array.shape[0])]
    labels[offsets] = True
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
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--process", type=int, default=1)
    args = parser.parse_args()

    files = [i for i in pathlib.Path(args.input).rglob(f"*{args.suffix}") if i.is_file()]
    with multiprocessing.Pool(args.process) as pool, tqdm(total = len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            pbar.update()
            pbar.refresh()

if __name__ == "__main__":
    main()