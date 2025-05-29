import argparse
import json
import pathlib
import subprocess
from multiprocessing import Pool
import gtirb
import capstone
from io import BytesIO
import tempfile

import lief
import numpy as np
def load_text(file_path, section_name=None):
    binary = lief.parse(str(file_path))

    if isinstance(binary, lief.ELF.Binary):
        name = ".text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.identity_class == lief.ELF.ELF_CLASS.CLASS64

    elif isinstance(binary, lief.MachO.Binary):
        name = "__text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.cpu_type == lief.MachO.CPU_TYPES.X86_64

    elif isinstance(binary, lief.PE.Binary):
        name = ".text" if section_name is None else section_name
        text_section = binary.get_section(name)
        base_addr = text_section.virtual_address + binary.optional_header.imagebase
        text_bytes = bytes(text_section.content)
        use_64_bit = binary.header.machine == lief.PE.MACHINE_TYPES.AMD64
    else:
        raise ValueError(f"Unsupported binary type: {type(binary)}")
    print(f"{use_64_bit=}")
    return text_bytes, use_64_bit, base_addr

def parse_gtirb(ir, use_64_bit=True):
    if use_64_bit:
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    else:
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_32)
    instructions = []
    function_entries = []
    functions = []
    last_addr = 0
    for module in ir.modules:
        names = module.aux_data['functionNames'].data
        blocks = module.aux_data['functionBlocks'].data
        entries = module.aux_data['functionEntries'].data
        for func_uuid, sym in names.items():
            func_blocks = blocks[func_uuid]
            try:
                entry_block = next(iter(entries[func_uuid]))
            except Exception as e:
                print(entry_block)
                print(e)
                exit(-1)
            addr = (entry_block.offset + 0x400000) if entry_block.address is None else entry_block.address
            function_entries.append(addr)
            function_blocks = {"entry": addr, "blocks":[]}
            for block in sorted(func_blocks, key=lambda x:x.offset):
                block_bytes = block.contents
                block_addr = (block.offset + 0x400000) if block.address is None else block.address
                function_blocks["blocks"].append(block_addr)
                for i in cs.disasm(block_bytes, block_addr):
                    instructions.append(i.address)
                    last_addr = max(last_addr, i.address + len(i.bytes))
            functions.append(function_blocks)
    if len(instructions) == 0:
        return None
    start = min(instructions)
    end = last_addr
    return {"instructions": sorted(instructions), "start": start, "end": end, "functions": function_entries, "blocks": functions}

def process_file(args, file):
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
        if output_path.exists():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(suffix=".gtirb") as gtirb_file:
            cmd = [args.ddisasm, f"--ir={gtirb_file.name}", str(file)]
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode:
                print(result.stderr)
                exit(-1)
            ir = gtirb.IR.load_protobuf_file(gtirb_file)
            text_bytes, use_64_bit, base_addr = load_text(file, args.section_name)
            results = parse_gtirb(ir, use_64_bit)
            pred = np.zeros(len(text_bytes), dtype=np.bool_)
            points = np.array(results["instructions"], dtype=np.uint64)
            points = points[(points >= base_addr) & (points < base_addr + len(text_bytes))]
            pred[points - base_addr] = True
            result = {
                "base_addr": np.array(base_addr, dtype=np.uint64),
                "pred": pred,
            }
            np.savez(output_path, **result)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the binaries")
    parser.add_argument("--file", type=str, help="File to process")
    parser.add_argument("--ddisasm", type=str, default="ddisasm", help="Path to ddisasm executable")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--gtirb", type=str, help="Path to the gtirb output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
    parser.add_argument("--section_name", type=str, help="Section name to process")
    args = parser.parse_args()
    
    if args.file:
        process_file(args, pathlib.Path(args.file))
        exit(0)
    
    root_dir = pathlib.Path(args.dir)
    
    pool = Pool(args.process)
    
    files = list(root_dir.rglob("*"))  # Convert generator to list
    pool.starmap(process_file, [(args, file) for file in files])
    
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()