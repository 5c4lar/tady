# tady
Tady: A Neural Disassembler without Consistency Violations

## Dev Environment
Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
```
Use Docker
```bash
docker build -t tady -f docker/Dockerfile .
```

## Datasets
Assemblage:
```bash
uv run scripts/dataset/assemblage/pdb_gt.py --bin_dir data/bin/assemblage/ --pdb_dir binaries --mapping_file scripts/dataset/assemblage/assemblage_locations.json --process 24 --output_dir data/gt_npz/assemblage --executable scripts/dataset/assemblage/pdb-markers
```
pangine
```bash
uv run  scripts/dataset/pangine/sqlite_gt.py --input data/tools/ddisasm-wis-evaluation/datasets/pangine-dataset/ --output data/gt_npz/pangine --process 24
```
bap
```bash
uv run  scripts/dataset/pangine/sqlite_gt.py --input data/tools/ddisasm-wis-evaluation/datasets/pangine-dataset/ --output data/gt_npz/pangine --process 24
```
quarks
```bash
uv run scripts/dataset/quarks/quokka_gt.py --source ../disassemble/datasets/diffing_obfuscation_dataset/data/ --target data/gt_npz/quarks --gt=True
```
x86_sok
```bash
uv run scripts/dataset/sok/parse_sok.py --dir data/bin/x86_dataset/ --pb_dir /mnt/disk/data/x86_dataset --output data/gt_npz/x86_dataset --process 24
```

## Train
Preprocess binary file
```bash
python3 scripts/preprocess.py dataset=x86_dataset
```

Train the model
```bash
python3 scripts/train.py
```

Export the model
```bash
python3 scripts/export.py
```

Generate config
```bash
python3 scripts/generate_model_conf.py
```

Serve the model
```bash
docker run --rm --gpus device=0 -p 8500:8500 -v  ./models/tf_models:/models -t --name tensorflow-serving tensorflow/serving:latest-gpu --xla_gpu_compilation_enabled=true --enable_batching=true --batching_parameters_file=/models/batching.conf --model_config_file=/models/model.conf
```

Test over dataset
```bash
python3 scripts/test.py process=24
```
