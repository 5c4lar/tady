import argparse
import json
import pathlib
import subprocess
from multiprocessing import Pool
import gtirb
import capstone
from io import BytesIO
import tempfile
cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)

def parse_gtirb(ir):
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
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".json")
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
            results = parse_gtirb(ir)
            with open(output_path, "w") as f:
                json.dump(results, f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the binaries")
    parser.add_argument("--file", type=str, help="File to process")
    parser.add_argument("--ddisasm", type=str, default="ddisasm", help="Path to ida executable")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--gtirb", type=str, help="Path to the gtirb output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
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