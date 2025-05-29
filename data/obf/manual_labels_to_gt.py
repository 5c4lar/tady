import numpy as np
import pathlib
import lief
import argparse
from tady.utils.loader import load_text
from tady import cpp

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", type=str, required=True)
    parser.add_argument("--bin", type=str, required=True)
    args = parser.parse_args()

    disassembler = cpp.Disassembler()
    text_bytes, use_64_bit, base_addr = load_text(args.bin, ".vmp0")
    print(hex(base_addr), hex(base_addr), hex(base_addr + len(text_bytes)))
    print(bytes(text_bytes)[:10].hex())
    labels = np.zeros(text_bytes.shape[0], dtype=np.bool_)
    masks = np.zeros(text_bytes.shape[0], dtype=np.bool_)
    with open(args.labels, "r") as f:
        for line in f:
            start, end = line.strip().split("-")
            start = int(start, 16)
            end = int(end, 16)
            if start < base_addr or end > base_addr + len(text_bytes):
                print(f"Invalid range: {line.strip()}")
                continue
            block_bytes = text_bytes[start - base_addr:end - base_addr]
            offsets = disassembler.linear_disasm(block_bytes, use_64_bit)
            instr_strings = disassembler.disasm_to_str(block_bytes, use_64_bit, start)
            addrs = offsets + start
            print([hex(i) for i in addrs])
            print(instr_strings)
            labels[addrs - base_addr] = True
            masks[start - base_addr:end - base_addr] = True
    result = {
        "text_array": text_bytes,
        "labels": labels,
        "base_addr": np.array(base_addr, dtype=np.uint64),
        "use_64_bit": np.array(use_64_bit, dtype=np.bool_),
        "mask": masks,
    }
    addrs = np.where(labels)[0] + base_addr
    print("Labels: ", [hex(i) for i in addrs])
    np.savez(args.bin + ".npz", **result)
            

if __name__ == "__main__":
    main()