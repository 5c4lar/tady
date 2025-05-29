import DeepDiCore
import numpy as np
# from elftools.common.exceptions import ELFError
# from elftools.elf.constants import SH_FLAGS
# from elftools.elf.elffile import ELFFile
# from pefile import PE
import argparse
import json
import pathlib
import time
from multiprocessing.pool import ThreadPool as Pool
import lief


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

class DeepDi:
    def __init__(self, key, gpu, batch_size):
        self.disasm = DeepDiCore.Disassembler(key, gpu)
        self.batch_size = batch_size

    def disassemble(self, path, section_name=None):
        code, x64, base_addr = load_text(path, section_name)
        inst_pred_buffer = np.zeros(len(code), dtype=np.bool_)
        score_buffer = np.zeros(len(code), dtype=np.float32)
        for i in range(0, len(code), self.batch_size):
            self.disasm.Disassemble(code[i:min(i+self.batch_size, len(code))], x64)
            prob = self.disasm.GetInstructionProb()
            # score is inv sigmoid of prob
            score = -np.log(1 / np.clip(prob, 1e-6, 1-1e-6) - 1)
            inst_pred = prob >= 0.5
            inst_pred_buffer[i:min(i+self.batch_size, len(code))] = inst_pred
            score_buffer[i:min(i+self.batch_size, len(code))] = score
        return inst_pred_buffer, score_buffer, base_addr

def process_file(args, file, deepdi, section_name=None):
    try:
        if file.is_file():
            rel_path = file.relative_to(args.dir)
            output_path = pathlib.Path(args.output) / (str(rel_path) + ".npz")
            if output_path.exists():
                print(f"{output_path} already exists")
                return
            # print(f"processing {file}")
            inst_pred, score, base_addr = deepdi.disassemble(file, section_name)
            print(f"processed {file}")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            result = {
                "base_addr": base_addr,
                "pred": inst_pred,
                "logits": score,
            }
            np.savez(output_path, **result)
    except Exception as e:
        print(e)
        return
    
class Worker:
    def __init__(self, args, tasks):
        self.deepdi = None
        self.args = args
        self.tasks = tasks
        
    def __call__(self):
        if self.deepdi is None:
            self.deepdi = DeepDi(self.args.key, self.args.gpu, 4 * 1024 * 1024)
        for file in self.tasks:
            print(f"Processing {file}")
            process_file(self.args, file, self.deepdi)
            
def main():
    parser = argparse.ArgumentParser(description='DeepDi example')
    parser.add_argument('--key', help='DeepDi key', required=True)
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    parser.add_argument('--file', type=str, help='Path to the file to disassemble')
    parser.add_argument('--task', type=str, help='Path to the task file')
    parser.add_argument('--dir',  type=str, help='Path to the directory of binary to disassemble')
    parser.add_argument('--output', type=str, help='Path to the directory to store output')
    parser.add_argument("--process", type=int, default=1,help="Num of process used to process the stats")
    parser.add_argument("--section", type=str, help="Section name to disassemble")
    args = parser.parse_args()
    if args.file:
        deepdi = DeepDi(args.key, args.gpu, 4 * 1024 * 1024)
        start_time = time.time()
        # res = example(deepdi, args.file)
        process_file(args, pathlib.Path(args.file), deepdi, args.section)
        # print(res)
        end_time = time.time()
        # Calculate the elapsed time
        elapsed_time = end_time - start_time
        print(f"Time taken for processing: {elapsed_time:.4f} seconds")
        return
        
    
    root_dir = pathlib.Path(args.dir)
    pool = Pool(args.process)
    if args.task:
        tasks = json.load(open(args.task))
        tasks = [pathlib.Path(task) for task in tasks]
    else:
        tasks = [file for file in root_dir.rglob("*") if file.is_file()]
    workers = [Worker(args, tasks[i::args.process]) for i in range(args.process)]
    
    pool.map(Worker.__call__, workers)
    pool.close()
    pool.join()
    


if __name__ == '__main__':
    main()
