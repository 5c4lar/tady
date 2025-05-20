import argparse
import shutil
import pathlib
import subprocess
from multiprocessing import Pool

def process_file(file):
    cmd = ["strip", str(file)]
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output = True)
    if result.returncode:
        print(result.stderr)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=str, help="Directory containing the binaries")
    parser.add_argument("--process", default=8, type=int, help="Num of process used to process the stats")

    args = parser.parse_args()
    
    root_dir = pathlib.Path(args.dir)
    pool = Pool(args.process)
    
    files = list(root_dir.rglob("*"))  # Convert generator to list
    pool.map(process_file, [file for file in files if file.is_file()])
    
    pool.close()
    pool.join()
    
if __name__ == "__main__":
    main()