import os
import shutil
import argparse
from typing import Dict
import lief
import pathlib
import json
import quokka
import numpy as np

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

def copy_elf_binaries(src_folder, dest_folder):
    """Recursively find and copy ELF executables and shared libraries from src_folder to dest_folder."""
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if is_elf(file_path):# and is_elf_executable_or_shared(file_path):
                # Create destination folder structure
                relative_path = os.path.relpath(file_path, src_folder)
                dest_path = os.path.join(dest_folder, relative_path)
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

def get_quokka_gt(file_path: str, quokka_path: str):
    """Get the ground truth from the Quokka file."""
    prog = quokka.Program(quokka_path, file_path)
    elf = lief.parse(file_path)
    text_section = elf.get_section('.text') # type: ignore
    text_array = np.frombuffer(text_section.content, dtype=np.uint8)
    base_addr = text_section.virtual_address
    instructions = []
    labels = np.zeros(text_section.size, dtype=np.bool_)
    masks = np.zeros(text_section.size, dtype=np.bool_)
    func: quokka.Function
    for func in prog.values():
        if func.start < base_addr:
            continue
        if func.end > base_addr + text_section.size:
            continue
        masks[func.start - base_addr:func.end - base_addr] = True
        for block_start in func.graph.nodes:
            block = func.get_block(block_start)
            for instr in block:
                instructions.append(instr)
    labels[np.array(instructions) - base_addr] = True
    result: Dict[str, np.ndarray] = {
        "text_array": text_array,
        "labels": labels,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array(elf.header.machine_type == lief.ELF.ARCH.X86_64, dtype=np.bool_), # type: ignore
        "mask": masks,
    }
    return result

def gather_ground_truth(src_folder, dest_folder):
    """Recursively find and copy ground truth files from src_folder to dest_folder."""
    for root, _, files in os.walk(src_folder):
        for file in files:
            file_path = os.path.join(root, file)
            if is_elf(file_path):
                relative_path = os.path.relpath(file_path, src_folder)
                quokka_path = os.path.join(src_folder, relative_path.replace(".exe", ".Quokka"))
                dest_path = os.path.join(dest_folder, relative_path)
                gt_file_path = os.path.join(dest_folder, relative_path + ".npz")
                # if os.path.exists(gt_file_path):
                #     continue
                os.makedirs(os.path.dirname(dest_path), exist_ok=True)
                results = get_quokka_gt(file_path, quokka_path)
                np.savez(gt_file_path, **results)
                print(f"Gathered ground truth: {gt_file_path}")
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, help="Source directory to gather binaries")
    parser.add_argument("--target", type=str, help="Target directory to store binaries")
    parser.add_argument("--gt", type=bool, default=False, help="Whether to gather ground truth")
    args = parser.parse_args()
    # Usage
    src_folder = args.source
    dest_folder = args.target
    if args.gt:
        gather_ground_truth(src_folder, dest_folder)
    else:
        copy_elf_binaries(src_folder, dest_folder)
    
if __name__ == "__main__":
    main()
