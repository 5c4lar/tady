import json
import pathlib
import shutil
from multiprocessing import Pool

import datasets
import hydra
import lief
import numpy as np
import pyarrow.parquet as pq
from datasets import Dataset
from omegaconf import DictConfig
from tqdm import tqdm

def process_file(arg):
    args, gt_path = arg
    if not gt_path.exists():
        print(f"File not found: {gt_path}")
        return None
    rel_path = gt_path.relative_to(args.gt_dir)

    return {
        "path": rel_path.with_suffix(""),
        "gt": gt_path.read_bytes()
    }


def load_parquets(path):
    root_dir = pathlib.Path(path)
    parquets = [i for i in root_dir.rglob("*.parquet")]
    for parquet in parquets:
        try:
            pq.ParquetFile(parquet)
        except:
            print(f"Error reading {parquet}, unlinking")
            parquet.unlink()
            continue
    parquets = [str(i) for i in root_dir.rglob("*.parquet")]
    ds = datasets.load_dataset("parquet", data_files=parquets, split="train")
    return ds


@hydra.main(version_base=None, config_path="conf/data", config_name="preprocess")
def main(args: DictConfig):
    root_dir = pathlib.Path(args.gt_dir)
    files = [i for i in root_dir.rglob("*.npz") if i.is_file()]
    dss = []
    i = 0
    ds: Dataset
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            if result is not None:
                dss.append(result)
            if len(dss) >= args.batch_size:
                ds = Dataset.from_list(dss, features=datasets.Features({"path": datasets.Value("string"), "gt": datasets.Value("binary")}))
                ds.to_parquet(pathlib.Path(args.tmp_dir) / f"{i}.parquet")
                i += 1
                dss = []
            pbar.update()
            pbar.refresh()
    if len(dss) > 0:
        ds = Dataset.from_list(dss, features=datasets.Features({"path": datasets.Value("string"), "gt": datasets.Value("binary")}))
        if i == 0:
            ds.save_to_disk(args.output_path)
            return
        ds.to_parquet(pathlib.Path(args.tmp_dir) / f"{i}.parquet")
        i += 1
    ds = load_parquets(args.tmp_dir)
    ds.save_to_disk(args.output_path)
    shutil.rmtree(args.tmp_dir)

if __name__ == "__main__":
    main()
