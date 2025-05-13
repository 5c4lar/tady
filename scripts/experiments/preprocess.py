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

from tady.utils.loader import chunk_data


def load_gt(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data


def get_labels(data, start, end):
    # Turn the labels into a np boolean array
    labels = np.zeros(end - start, dtype=bool)
    instructions_array = np.array(data["instructions"])
    instructions = instructions_array[(instructions_array >= start) & (
        instructions_array < end)] - start
    labels[instructions] = True
    return labels


def process_file(arg):
    args, file_path = arg
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    rel_path = file_path.relative_to(args.binary_dir)
    output_path = pathlib.Path(args.tmp_dir) / (str(rel_path) + ".parquet")
    gt_path = pathlib.Path(args.gt_dir) / (str(rel_path) + ".json")
    if not gt_path.exists():
        print(f"GT file not found: {gt_path}")
        return None

    binary = lief.parse(str(file_path))
    is_64 = binary.header.machine_type == lief.ELF.ARCH.X86_64
    text_start = binary.get_section(".text").virtual_address
    text_end = text_start + binary.get_section(".text").size
    data = load_gt(gt_path)
    start = max(data["start"], text_start)
    end = min(data["end"], text_end)
    byte_sequence = binary.get_content_from_virtual_address(
        start, end - start).tobytes()
    # Turn byte sequence into a np array
    byte_sequence = np.frombuffer(byte_sequence, dtype=np.uint8)
    labels_array = get_labels(data, start, end)
    if args.chunk:
        byte_chunks, labels, masks = chunk_data(
            byte_sequence, args.seq_len, args.window_size, labels_array)

        assert np.sum(masks) == len(byte_sequence)
        return {
            "bytes": byte_chunks,
            "labels": labels,
            "mask": masks,
            "is_64": [is_64] * len(byte_chunks),
        }
    else:
        return {
            "bytes": byte_sequence,
            "labels": labels_array,
            "is_64": is_64,
        }
    # output_path.parent.mkdir(parents=True, exist_ok=True)
    # ds.to_parquet(output_path)


def load_parquets(path):
    root_dir = pathlib.Path(path)
    parquets = [i for i in root_dir.rglob("*.parquet")]
    for parquet in parquets:
        try:
            pq.ParquetFile(parquet)
        except:
            # unlink the file
            print(f"Error reading {parquet}, unlinking")
            parquet.unlink()
            continue
    parquets = [str(i) for i in root_dir.rglob("*.parquet")]
    ds = datasets.load_dataset("parquet", data_files=parquets, split="train")
    return ds


@hydra.main(version_base=None, config_path="conf/data", config_name="preprocess")
def main(args: DictConfig):
    root_dir = pathlib.Path(args.binary_dir)
    files = [i for i in root_dir.rglob("*") if i.is_file()]
    dss = []
    with Pool(args.process) as pool, tqdm(total=len(files)) as pbar:
        for result in pool.imap_unordered(process_file, [(args, file) for file in files]):
            if result is not None:
                dss.append(result)
            pbar.update()
            pbar.refresh()
    ds: Dataset = Dataset.from_list(dss)
    # ds.to_parquet(args.output_path)
    ds.save_to_disk(args.output_path)
    # shutil.rmtree(args.tmp_dir)

if __name__ == "__main__":
    main()
