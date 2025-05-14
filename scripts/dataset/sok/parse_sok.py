from parse_sok_pb import parse_pb
import pathlib
from multiprocessing import Pool
import json
def process_file(args, file):
    rel_path = file.relative_to(root_dir)
    # print(rel_path)
    # compiler, opt = rel_path.parent.name.split("_", 1) if not rel_path.name.startswith('openssl_') else ("GCC", rel_path.name.split("_")[1])
    gt_file = pathlib.Path(args.pb_dir) / rel_path.parent / f"gtBlock_{rel_path.name}.pb"
    if not gt_file.exists():
        print(f"GT file not found: {gt_file}")
        return
    target_path = pathlib.Path(args.output) / (str(rel_path) + ".json")
    target_path.parent.mkdir(parents=True, exist_ok=True)
    # if not target_path.exists():
    gt = parse_pb(gt_file)
    if len(gt['instructions']) == 0:
        print(f"GT file is empty: {gt_file}")
        return
    # print(f"{target_path}")
    if not target_path.exists():
        with open(target_path, "w") as f:
            json.dump(gt, f)
    return
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, help="Path to the binary")
    parser.add_argument("--dir", type=str, help="Path to the directory containing binaries")
    parser.add_argument("--pb_dir", type=str, help="Path to the gt directory")
    parser.add_argument("--output", type=str, help="Path to the output directory")
    parser.add_argument("--process", type=int, help="Number of processes used for batch processing")
    args = parser.parse_args()
    if args.file is not None:
        file_path = pathlib.Path(args.file)
        result = parse_pb(file_path)
        print(result)
    else:
        root_dir = pathlib.Path(args.dir)
        pool = Pool(args.process)
        files = [i for i in root_dir.rglob("*") if i.is_file()]  # Convert generator to list
        pool.starmap(process_file, [(args, file) for file in files])
        pool.close()
        pool.join()