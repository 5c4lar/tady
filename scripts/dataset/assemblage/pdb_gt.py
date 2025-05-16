'''
Use pdb-markers to parse the pdb files to get disassembly ground truth.
'''
import argparse
import pathlib
import yaml
import subprocess
from multiprocessing import Pool
from tady import cpp
from tady.utils.loader import load_text
from tqdm import tqdm
import numpy as np
import json
disassembler = cpp.Disassembler()

def parse_result(l, text_bytes, use_64_bit, base_address):
    """Parse the result of pdb-markers.

    Args:
        l: list like ['0x411000-0x411005 $i', '0x411005-0x411343 $a', '0x411343-0x411681 $i', '0x411681-0x411697 $a']
        where lines end with $a indicate code, $i indicate ignore and $d indicate data.
        text_bytes: bytes of the text section.
        use_64_bit: whether the binary is 64-bit.
        
    """
    l = [r.split(" ") for r in l]
    mask = np.zeros(len(text_bytes), dtype=np.bool_)
    label_array = np.zeros(len(text_bytes), dtype=np.bool_)
    for r in l:
        (start, end), label = r[0].split("-"), r[1]
        start = int(start, 16)
        end = int(end, 16)
        match(label):
            case "$a":
                offsets = disassembler.linear_disasm(text_bytes[start - base_address : end - base_address], use_64_bit)
                mask[start - base_address:end - base_address] = True
                label_array[offsets + (start - base_address)] = True
            case "$i":
                mask[start - base_address:end - base_address] = False
            case "$d":
                mask[start - base_address:end - base_address] = True
                label_array[start - base_address:end - base_address] = False
    return mask, label_array

def parse_pdb(arg):
    pe_path, pdb_path, output_path, executable_path = arg
    if output_path.exists():
        return
    if not pe_path.exists() or not pdb_path.exists():
        return
    args = [executable_path, pe_path, pdb_path]
    res = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        print(f"Failed to parse pdb file {pdb_path} with executable {pe_path}")
        return
    res = yaml.safe_load(res.stdout.decode("utf-8"))
    if res is None:
        print(f"Failed to parse pdb file {pdb_path} with executable {pe_path}")
        return
    text_array, use_64_bit, base_addr = load_text(pe_path)
    mask, label_array = parse_result(res['.text'], text_array, use_64_bit, base_addr)
    result = {
        "text_array": text_array,
        "labels": label_array,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array(use_64_bit, dtype=np.bool_),
        "mask": mask
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, **result)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bin_dir", type=str, required=True)
    parser.add_argument("--pdb_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--mapping_file", type=str, required=True)
    parser.add_argument("--process", type=int, default=1)
    parser.add_argument("--executable", type=str, required=True)
    args = parser.parse_args()
    
    bin_dir = pathlib.Path(args.bin_dir)
    pdb_dir = pathlib.Path(args.pdb_dir)
    output_dir = pathlib.Path(args.output_dir)
    
    with open(args.mapping_file, "r") as f:
        mapping = json.load(f)
    
    with Pool(args.process) as p, tqdm(total=len(mapping)) as pbar:
        for _ in p.imap_unordered(parse_pdb, [(bin_dir / k, pdb_dir / mapping[k], output_dir / (k + ".npz"), args.executable) for k in mapping]):
            pbar.update()
            pbar.refresh()
            
        

if __name__ == "__main__":
    main()
    
    
    
    