import argparse
import pathlib
import subprocess
import tempfile
from multiprocessing import Pool
from tqdm import tqdm
import numpy as np


def process_file(args):
    args, file = args
    if file.is_file():
        rel_path = file.relative_to(args.dir)
        output_path = pathlib.Path(args.output) / (str(rel_path) + ".json")
        if output_path.exists():
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        ghidra_path = pathlib.Path(args.ghidra_root)
        with tempfile.TemporaryDirectory(dir="/dev/shm") as tmpdirname:
            cmd = [str(ghidra_path / "support" / "analyzeHeadless"), tmpdirname, "disassemble", "-deleteproject", "-import", str(file), "-postScript", args.script, args.dir, args.output]
            print(' '.join(cmd))
            result = subprocess.run(cmd, capture_output=True)
            if result.returncode:
                print(result.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the binaries")
    parser.add_argument("--file", type=str, help="Path to the file to be disassembled")
    parser.add_argument("--ghidra_root", type=str, help="Path to ghidra root directory")
    parser.add_argument("--script", type=str, help="Script used to disassemble binaries")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
    args = parser.parse_args()
    
    if args.file:
        process_file(args, pathlib.Path(args.file))
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