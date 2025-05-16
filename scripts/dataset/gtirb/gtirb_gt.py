import lief
import capstone
import json
import pathlib
import gtirb
from multiprocessing import Pool
from io import BytesIO
import numpy as np
from tady.utils.loader import load_text
from tqdm import tqdm

def parse_binary(binary):
    elf: lief.ELF.Binary = lief.parse(binary)
    arch = elf.header.machine_type
    match arch:
        case lief.ELF.ARCH.X86_64:
            cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        case lief.ELF.ARCH.RISCV:
            cs = capstone.Cs(capstone.CS_ARCH_RISCV, capstone.CS_MODE_RISCV64)
        case lief.ELF.ARCH.AARCH64:
            cs = capstone.Cs(capstone.CS_ARCH_ARM64, capstone.CS_MODE_ARM)
        case lief.ELF.ARCH.SPARC:
            cs = capstone.Cs(capstone.CS_ARCH_SPARC, capstone.CS_MODE_LITTLE_ENDIAN)
        case _:
            print("Unknown architecture")
    if elf is None:
        return
    gtirb_section = elf.get_section('.gtirb')
    if gtirb_section is None:
        return
    try:
        ir = gtirb.IR.load_protobuf_file(BytesIO(bytes(gtirb_section.content)))
    except Exception as e:
        print(f"Failed to process {binary}")
        print(e)
        return
    
    last_addr = 0
    text_array, use_64_bit, base_addr = load_text(binary)
    labels = np.zeros(text_array.shape[0], dtype=np.bool_)
    masks = np.zeros(text_array.shape[0], dtype=np.bool_)
    for module in ir.modules:
        # print(module.aux_data['functionNames'])
        names = module.aux_data['functionNames'].data
        blocks = module.aux_data['functionBlocks'].data
        entries = module.aux_data['functionEntries'].data
        # for func_uuid, sym in sorted(names.items(), key=lambda x:list(entries[x[0]])[0].address):
        for func_uuid, sym in names.items():
            func_blocks = blocks[func_uuid]
            try:
                entry_block = next(iter(entries[func_uuid]))
                text_section = elf.get_section(entry_block.section.name)
            except Exception as e:
                print(entry_block)
                print(e)
                print(binary)
                exit(-1)
            addr = (text_section.file_offset - elf.get_section('.text').file_offset + entry_block.offset) if entry_block.address is None else entry_block.address
            for block in sorted(func_blocks, key=lambda x:x.offset):
                # block_entries.append(block.offset)
                block_addr = block.offset if block.address is None else block.address
                if block_addr < base_addr:
                    continue
                if block_addr >= base_addr + text_array.shape[0]:
                    break
                offset = block.offset if block.address is None else block.address - block.section.address
                block_bytes = bytes(text_section.content[offset: offset + block.size])
                for i in cs.disasm(block_bytes, block_addr):
                    if i.address < base_addr:
                        continue
                    if i.address >= base_addr + text_array.shape[0]:
                        break
                    labels[i.address - base_addr] = True
                    last_addr = max(last_addr, i.address + len(i.bytes))
                masks[block_addr - base_addr:block_addr - base_addr + block.size] = True
    return {
        "text_array": text_array,
        "labels": labels,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array(use_64_bit, dtype=np.bool_),
        "mask": masks,
    }

def process_file(arg):
    args, file = arg
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
        if output_path.exists():
            return
        info = parse_binary(file)
        if info is None:
            return
        rel_path = file.relative_to(args.dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **info)
        
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the binary")
    parser.add_argument("--dir", type=str, help="Path to the directory containing binaries")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
    args = parser.parse_args()
    if args.file is not None:
        file_path = pathlib.Path(args.file)
        result = parse_binary(file_path)
        print(result)
    else:
        root_dir = pathlib.Path(args.dir)
        files = list(root_dir.rglob("*"))  # Convert generator to list
        with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
            for _ in pool.imap_unordered(process_file, [(args, file) for file in files]):
                pbar.update()
                pbar.refresh()
    
    
if __name__ == "__main__":
    main()