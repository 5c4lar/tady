import pathlib
import pyghidra
import numpy as np
import os
import tempfile
from tady.utils.loader import load_text
def get_text_section_base(program, section_name):
    """
    Finds and returns the base address of the .text section.
    
    :param program: Current Ghidra program (provided automatically in Ghidra's script environment)
    :return: The base address of the .text section, or None if not found
    """
    memory = program.getMemory()
    text_block = None

    # Iterate over all memory blocks to find .text
    for block in memory.getBlocks():
        if block.getName() == section_name:
            text_block = block
            break

    if text_block:
        start_addr = text_block.getStart()
        print("Found %s section at: %s" % (section_name, start_addr))
        return start_addr
    else:
        print("Could not find %s section" % section_name)
        return None
def disassemble(file, section_name=None):
    with tempfile.TemporaryDirectory() as tmpdir:
        with pyghidra.open_program(file, tmpdir) as flat_api:
            program = flat_api.getCurrentProgram()
            listing = program.getListing()
            instructions = []
            for instruction in listing.getInstructions(True):
                instructions.append(instruction.getAddress().getOffset())
            text_array, use_64_bit, base_addr = load_text(file, section_name)
            ghidra_text_base = get_text_section_base(program, section_name)
            pred = np.zeros(text_array.shape[0], dtype=np.bool_)
            if ghidra_text_base is None:
                print("No text section found")
                exit()
            points = np.array(instructions, dtype=np.int64) - ghidra_text_base.getOffset()
            points = points[(points >= 0) & (points < text_array.shape[0])]
            pred[points] = True
            result = {
                "pred": pred,
                "base_addr": base_addr,
            }
            return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the file to be disassembled")
    parser.add_argument("--output", type=str, help="Path to the output file")
    parser.add_argument("--dir", type=str, help="Use 64-bit mode")
    parser.add_argument("--section_name", default=None, type=str, help="Section name")
    args = parser.parse_args()
    result = disassemble(args.file, args.section_name)
    if args.output:
        rel_path = pathlib.Path(args.file).relative_to(args.dir)
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(output_path, **result)
    else:
        print(result)