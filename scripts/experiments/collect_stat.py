import pathlib
import json
import hydra
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="collect_stat")
def main(args: DictConfig):
    all_results = {}
    for dataset in args.datasets:
        result_dir = pathlib.Path(args.prune_dir) / dataset
        dataset_stat = result_dir / "prune_result.json"
        with open(dataset_stat, "r") as f:
            result = json.load(f)
        all_results[dataset] = result
    print(all_results)
    with open(pathlib.Path(args.prune_dir) / "all_prune_result.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()
