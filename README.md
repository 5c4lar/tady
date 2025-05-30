# tady
Tady: A Neural Disassembler without Consistency Violations

## Environment
Run the experiments either on host or in docker. Since some of the baselines (DeepDi and ddisasm) are also run in docker, we recommand setting up the environment
on the host. Otherwise, prepare the machine for nested docker. Or use the provided disasssembly results for simplicity.
### Host
Install uv and run locally on Ubuntu 24.04
```bash
apt-get update && apt-get install -y \
    build-essential \
    llvm-dev \ # for llvm-18
    libboost-graph-dev \  # for BGL
    libpython3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
```
### Docker
**Or** Use Docker
```bash
docker build -t tady -f docker/Dockerfile .
```
## Datasets

Either download our processed datasets or rerun our the preprocessing steps.

We provide the binaries in `bin.tar.gz`, which should be decompressed to `data/bin`, and the ground truth data `gt_npz.tar.gz`, which should be decompressed to `data/gt_npz`. with these two directory, download the original datasets is not necessary.

### Download

### To reproduce the datasets from their provider
Download raw datasets
```bash
bash data/download/download_dataset.sh
```

build rw dataset
```
docker run -it -v (pwd):/work -w /work --name gt bin2415/x86_gt:0.1 /bin/bash
# Inside Docker
python3 -m pip install hydra-core hydra-joblib-launcher
python3 scripts/dataset/sok/compile.py -m project=curl,diffutils,gmp,ImageMagick,libmicrohttpd,libtomcrypt,openssl,putty,sqlite,zlib compiler=clang32 opt=O0,O1,O2,O3
python3 scripts/dataset/gather_binaries.py --source data/install --target data/raw/rw
bash extract_gt/run_extract_linux.sh -d /work/data/raw/rw/ -s extract_gt/extractBB.py -p gtBlock -j 24
```

Pangine
```bash
# Gather Binaries
uv run scripts/dataset/gather_binaries.py --source data/download/pangine-dataset --target data/bin/pangine
# Process Labels
uv run  scripts/dataset/pangine/sqlite_gt.py --input data/download/pangine-dataset --output data/gt_npz/pangine --process 24
```

Assemblage:
```bash
# Gather Binaries
uv run scripts/dataset/gather_binaries.py --source data/download/assemblage-dataset --target data/bin/assemblage
# Process Labels
# pdb-markers is built from https://github.com/GrammaTech/disasm-benchmark/tree/main/pdb
uv run scripts/dataset/assemblage/pdb_gt.py --bin_dir data/bin/assemblage/ --pdb_dir data/download/assemblage-dataset --mapping_file scripts/dataset/assemblage/assemblage_locations.json --process 24 --output_dir data/gt_npz/assemblage --executable scripts/dataset/assemblage/pdb-markers
```

x86_sok
```bash
# Gather Binaries
uv run scripts/dataset/gather_binaries.py --source data/download/x86-dataset --target data/bin/x86_dataset
# Process Labels
uv run scripts/dataset/sok/parse_sok.py --dir data/bin/x86_dataset/ --pb_dir data/download/x86-dataset --output data/gt_npz/x86_dataset --process 24
```

rw
```bash
# Gather Binaries
uv run scripts/dataset/gather_binaries.py --source data/install --target data/bin/rw
# Process Labels
uv run scripts/dataset/sok/parse_sok.py --dir data/bin/rw/ --pb_dir data/raw/rw --output data/gt_npz/rw --process 24
```

quarks
```bash
# Gather Binaries
uv run scripts/dataset/quarks/quokka_gt.py --source data/download/quarks-dataset --target data/bin/quarks
# Process Labels
uv run scripts/dataset/quarks/quokka_gt.py --source data/download/quarks-dataset --target data/gt_npz/quarks --gt=True
```

obf-benchmark
```bash
# Gather Binaries
uv run scripts/dataset/obf-benchmark/obf_gt.py --source data/download/obf-dataset --target data/bin/obf-benchmark --obf=True
# Process Labels
uv run scripts/dataset/obf-benchmark/obf_gt.py --source data/download/obf-dataset --target data/gt_npz/obf-benchmark --gt=True
```

Strip the binaries
```bash
uv run scripts/dataset/strip.py --dir data/bin --process 24
```

Preprocess to huggingface datasets format for training
```bash
uv run scripts/experiments/preprocess.py -m dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks
```

## Train
```bash
# Train the Tady Model (lite, all) and variants for ablation study
uv run scripts/experiments/train.py -m dataset=pangine epoch=1 process=16 model.attention=lite,sliding connections=none,all
# Train the TadyA Model the same architecture with Tady but on a mixure of the dataset setting scripts/experiments/conf/dataset/mix_all.yaml
uv run scripts/experiments/train.py -m dataset=mix_all epoch=1 process=16
```
## Eval
The data for the tables are provided in `artifacts.tar.gz`, which can be reproduced with the commands listed below.

### Tady
Export the model
```bash
# Export model for Tady and ablations
uv run scripts/experiments/export.py -m dataset=pangine  model.attention=lite,sliding connections=all,none
# Export model for TadyA
uv run scripts/experiments/export.py dataset=mix_all
```

Serve the models
```bash
docker run --rm --gpus device=0 -p 8500:8500 -v  $PWD/models/tf_models:/models -t --name tensorflow-serving tensorflow/serving:latest-gpu --xla_gpu_compilation_enabled=true --enable_batching=true --batching_parameters_file=/models/batching.conf --model_config_file=/models/model.conf
```

Test over datasets
```bash
# For Tady and ablations
uv run scripts/experiments/eval.py -m dataset=pangine test_dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks process=24 model.attention=lite,sliding connections=all,none num_samples=1000
# For TadyA
uv run scripts/experiments/eval.py -m dataset=mix_all test_dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks process=24 num_samples=1000
```

Stop the tensorflow-serving after testing to free the GPU Memory.
```bash
docker stop tensorflow-serving
```
### Baselines

Reproducing the disassembly results is time consuming for those baselines, we provide the `eval_strip_baselines.tar.gz` which stores the disassemble results of the baselines. It should be decompressed to `data/eval_strip`. Of course, feel free to reproduce the results if you have enough resources.

**ddisasm**
```bash
# Build the docker for ddisasm baseline, add numpy dependency
docker build -t baseline_ddisasm scripts/baselines/ddisasm
docker run -it -d -v $PWD:/work --gpus all --name ddisasm baseline_ddisasm /bin/bash
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ddisasm process=8 num_samples=1000
```
**deepdi**
```bash
git clone https://github.com/DeepBitsTechnology/DeepDi.git
docker build -t deepdi -f DeepDi/Dockerfile-gpu DeepDi
docker build -t baseline_deepdi scripts/baselines/DeepDi
docker run -it -d -v (pwd):/work --gpus all --name deepdi baseline_deepdi /bin/bash
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=deepdi process=1 num_samples=1000
```
**ghidra**

Download [Ghidra](https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3.2_build/ghidra_11.3.2_PUBLIC_20250415.zip) and uncompress it into data/tools/ghidra
```bash
uv pip install -f data/tools/ghidra/Ghidra/Features/PyGhidra/pypkg/dist/ pyghidra
uv pip install -f data/tools/ghidra/docs/ghidra_stubs/ ghidra-stubs
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ghidra process=24 num_samples=1000
```
**ida**

Make sure to have IDA pro 9.1 installed on your machine
```bash
uv pip install $PATH_TO_IDA_PRO/idalib/python/
uv run $PATH_TO_IDA_PRO/idalib/python/py-activate-idalib.py
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ida process=24 num_samples=1000
```
**xda**

Download xda_model_reproduce.tar.gz in artifact and decompress it to scripts/baselines/XDA
```bash
tar -zxvf xda_model_reproduce.tar.gz -C scripts/baselines/XDA
```
Prepare the environment
```bash
git clone https://github.com/CUMLSec/XDA.git && cd XDA
conda create -n xda python=3.7 numpy scipy scikit-learn colorama
conda activate xda
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install --editable .
pip install lief
```
eval
```bash
# Back to project root
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=xda process=24 num_samples=1000
```

### Prune
```bash
# To configure the disassemblers to test, update the list at scripts/experiments/conf/prune.yaml models
uv run scripts/experiments/prune.py -m test_dataset=obf-benchmark,rw,x86_dataset,pangine,quarks,assemblage process=24 num_samples=1000
# Collect stat for table1 and table3 and table5, results at data/prune/all_prune_result.json
uv run scripts/experiments/collect_stat.py # This generate artifacts/all_prune_result_table3_5.json
# For table2, we test over all files available in the dataset without sampling
uv run scripts/experiments/prune.py -m test_dataset=obf-benchmark,rw,x86_dataset,pangine,quarks,assemblage process=24 models="[gt]"
```

### Errors
```bash
# This will summarize the errors detected for table1, table2 and table4, to configure the disassemblers to summarize, update the list at scripts/experiments/conf/error_stat.yaml models, results are at data/error/error_stat.json
uv run scripts/experiments/error_stat.py # This generate artifacts/error_stat_table1_2_4.json
```
After running above script, the details of the detected errors can be found at `data/error/{dataset}/{disassembler}_error_stat.log`


### Efficiency

Select samples
```bash
mkdir artifacts
uv run scripts/ablation/sample_select.py --dir data/bin/x86_dataset/ # This generate artifacts/selected_samples.json
```

Benchmark for the disassemblers
```bash
# Make sure ddisasm and deepdi docker are running
# Tensorflow-serving aggresively take all GPU memory, so we need to evaluate tady after finishing others
uv run scripts/experiments/bench.py process=1 dataset=pangine test_dataset=x86_dataset
# Start tensorflow-serving and run again
uv run scripts/experiments/bench.py process=1 dataset=pangine test_dataset=x86_dataset
# Result at data/eval_strip/x86_dataset/benchmark_cache.json gives the data for drawing Figure 8, which is artifacts/benchmark_cache_figure8.json
```
Benchmark Tady
```bash
uv run scripts/ablation/model_efficiency.py --samples artifacts/selected_samples.json --model_id instruction_cpp_pa
ngine_lite_all_64lw_64rw_16h_2l_prev000 --batch_size 32 --disassembler cpp --dir (pwd) --output artifacts/benchmark_results.json --plot artifacts/time_vs_size_cpp.pdf # This generate the data for drawing Figure 9, which is artifacts/benchmark_results_figure9.json
```
Benchmark PDT
```bash
# Prepare the scores for Pruning
uv run scripts/experiments/batch_run.py --files artifacts/selected_samples.json --dir data/bin --model instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 --output_dir artifacts/scores
uv run scripts/ablation/prune_efficiency.py --samples artifacts/selected_samples.json --output artifacts/prune --model_id=instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 # This generate the data for Figure 10, 11, which is artifacts/prune_data_figure10_11.json
```
### VMProtect
Generate the npz format ground truth for the binary with the given label.
```bash
uv run data/obf/manual_labels_to_gt.py --labels data/obf/labels.txt --bin data/obf/TestApp.vmp.exe
```
The target is at .vmp0 section, we specify that manually for the disassemblers.

Tady
```bash
# This shows an example that our trained model can extrapolate to arbitrary length, though trained only on 8192
# Tady
uv run -m tady.infer --path data/obf/TestApp.vmp.exe --model instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 --section_name .vmp0 --output_path data/obf/tady/TestApp.vmp.exe.npz --seq_len 569038 --batch_size 1
# TadyA
uv run -m tady.infer --path data/obf/TestApp.vmp.exe --model instruction_cpp_mix_all_lite_all_64lw_64rw_16h_2l_prev000 --section_name .vmp0 --output_path data/obf/tadya/TestApp.vmp.exe.npz --seq_len 569038 --batch_size 1
```

DeepDi

```bash
docker exec -it deepdi /bin/bash -c ' PYTHONPATH=. python3 /work/scripts/baselines/DeepDi/DeepDiLief.py --gpu --file /work/data/obf/TestApp.vmp.exe --output /work/data/obf/deepdi --dir /work/data/obf --key aaf9bb2902c6d7eeaf5a8c7156ab77113a9d02db46e33edaf5f66dc53f8c7caa5c0d35a18ee8197250c06cad37eca340a47d79dee0ed266355999ec358a040f1 --process 1 --section .vmp0'
```

XDA

```bash
conda run -n xda python scripts/baselines/XDA/batch_eval_lief.py --gpu --file data/obf/TestApp.vmp.exe --output data/obf/xda/ --dir data/obf --model_path scripts/baselines/XDA/checkpoints/finetune_instbound_new_dataset --dict_path scripts/baselines/XDA/xda_dataset/processed --section_name .vmp0
```

ddisasm

```bash
docker exec -it ddisasm python3 /work/scripts/baselines/ddisasm/batch_run_lief.py --dir /work/data/obf --file /work/data/obf/TestApp.vmp.exe --output /work/data/obf/ddisasm --section_name .vmp0
```

IDA pro
```bash
 uv run -m scripts.baselines.ida.batch_run --dir data/obf --file data/obf/TestApp.vmp.exe --output data/obf/ida --section_name .vmp0
```

Ghidra
```bash
GHIDRA_INSTALL_DIR=data/tools/ghidra uv run -m scripts.baselines.ghidra.pyghidra_disassemble --file data/obf/TestApp.vmp.exe --output data/obf/ghidra --section_name .vmp0 --dir data/obf
```

To evaluate the results
```bash
uv run data/obf/eval.py --gt data/obf/TestApp.vmp.exe.npz --pred data/obf/{disassembler}/TestApp.vmp.exe.npz
# After prune
uv run -m tady.prune --gt data/obf/TestApp.vmp.exe.npz --pred data/obf/{disassembler}/TestApp.vmp.exe.npz
```