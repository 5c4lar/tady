import argparse
import pathlib
import json
from multiprocessing import Pool
from .idalib_disassemble import parse_binary
from tqdm import tqdm
import numpy as np
from tady.utils.loader import load_text

def disassemble(file, section_name=None):
    text_array, use_64_bit, base_addr = load_text(file, section_name)
    
    res = parse_binary(file, persist_database=False)
    offsets = np.array(res["instructions"], dtype=np.int64) - base_addr
    offsets = offsets[(offsets >= 0) & (offsets < text_array.shape[0])]
    pred = np.zeros(text_array.shape[0], dtype=np.bool_)
    pred[offsets] = True
    result = {
        "pred": pred,
        "base_addr": base_addr,
    }
    return result

def process_file(args):
    args, file = args
    if file.is_file():
        result = disassemble(file, args.section_name)
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
        if output_path.exists():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the binaries")
    parser.add_argument("--file", type=str, help="Path to the file to be disassembled")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
    parser.add_argument("--section_name", default=None, type=str, help="Section name")
    args = parser.parse_args()
    
    if args.file:
        # print(parse_binary(pathlib.Path(args.file), False)['functions'])
        process_file((args, pathlib.Path(args.file)))
    else:
        root_dir = pathlib.Path(args.dir)
        
        pool = Pool(args.process)
            
        files = [file for file in root_dir.rglob("*") if file.is_file()]  # Convert generator to list
        with Pool(args.process) as pool, tqdm(total = len(files)) as pbar:
            for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
                pbar.update()
                pbar.refresh()

if __name__ == "__main__":
    main()