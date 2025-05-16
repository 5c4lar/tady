import os
import shutil
import argparse
import lief
import pathlib
import json

import numpy as np
from tady.utils.loader import load_text


def is_elf(file_path):
    """Check if a file is an ELF executable."""
    with open(file_path, 'rb') as f:
        header = f.read(4)
    return header == b'\x7fELF'  # ELF files start with '\x7fELF'

def is_elf_executable_or_shared(file_path):
    """Check if the file is an ELF executable or shared library."""
    binary = lief.parse(file_path)
    file_type = binary.header.file_type
    return binary and (file_type == lief.ELF.Header.FILE_TYPE.EXEC or file_type == lief.ELF.Header.FILE_TYPE.DYN)

def copy_elf_binaries(src_folder, dest_folder, obf):
    """Recursively find and copy ELF executables and shared libraries from src_folder to dest_folder."""
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if file.endswith(".obf"):
                if not obf:
                    continue
            else:
                if obf:
                    continue
            if is_elf(file_path):# and is_elf_executable_or_shared(file_path):
                # Create destination folder structure
                relative_path = os.path.relpath(file_path, src_folder)
                dest_path = pathlib.Path(os.path.join(dest_folder, relative_path)).parent
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                shutil.copy2(file_path, dest_path)
                print(f"Copied: {file_path} to {dest_path}")
                
def read_text_addrs(file_path):
    """Read the text addresses from a file."""
    with open(file_path, 'r') as f:
        lines = f.readlines()
    addrs = []
    for line in lines:
        addr = int(line.strip(), 16)
        addrs.append(addr)
    return addrs

def gather_ground_truth(src_folder, dest_folder):
    """Recursively find and copy ground truth files from src_folder to dest_folder."""
    for root, _, files in os.walk(src_folder):
        for file in files:
            if file == "instr-addresses-actual.txt":
                file_path = pathlib.Path(os.path.join(root, file))
                relative_path = os.path.relpath(file_path, src_folder)
                dest_path = pathlib.Path(os.path.join(dest_folder, relative_path)).parent
                elf_path = file_path.parent / "g.rr.O3.obf"
                text_array, use_64_bit, base_addr = load_text(elf_path)
                addrs = read_text_addrs(file_path)
                start_addr = min(addrs)
                end_addr = max(addrs) + 1
                labels = np.zeros(text_array.shape[0], dtype=np.bool_)
                masks = np.zeros(text_array.shape[0], dtype=np.bool_)
                points = np.array(addrs, dtype=np.uint64)
                points = points[(points >= base_addr) & (points < base_addr + text_array.shape[0])]
                start = min(points) - base_addr
                end = max(points) - base_addr
                labels[points - base_addr] = True
                masks[start:end] = True
                result = {
                    "text_array": text_array,
                    "use_64_bit": np.array(use_64_bit, dtype=np.bool_),
                    "base_addr": np.array(base_addr, dtype=np.uint64),
                    "labels": labels,
                    "mask": masks
                }
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                np.savez(dest_path, **result)
                print(f"Handling: {file_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Source directory to gather binaries")
    parser.add_argument("--target", type=str, help="Target directory to store binaries")
    parser.add_argument("--obf", type=bool, default=False, help="Whether to gather obfuscated binaries")
    parser.add_argument("--gt", type=bool, default=False, help="Whether to gather ground truth")
    args = parser.parse_args()
    # Usage
    src_folder = args.source
    dest_folder = args.target
    if args.gt:
        gather_ground_truth(src_folder, dest_folder)
    else:
        copy_elf_binaries(src_folder, dest_folder, args.obf)
    
if __name__ == "__main__":
    main()
