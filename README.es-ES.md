# Tady

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15541311.svg)](https://doi.org/10.5281/zenodo.15541311)
[![arXiv](https://img.shields.io/badge/arXiv-2506.13323-b31b1b.svg?style=flat)](https://arxiv.org/abs/2506.13323)

Tady: Un desensamblador neuronal sin violaciones de restricciones estructurales

## Entorno
Ejecute los experimentos ya sea en el host o en docker. Dado que algunas de las líneas base (DeepDi y ddisasm) también se ejecutan en docker, recomendamos configurar el entorno en el host. De lo contrario, prepare la máquina para docker anidado. O utilice los resultados de desensamblado proporcionados por simplicidad.

### Host
Instale uv y ejecútelo localmente en Ubuntu 24.04
```bash
apt-get update && apt-get install -y \
    build-essential \
    llvm-dev \ # para llvm-18
    libboost-graph-dev \  # para BGL
    libpython3-dev
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Docker
**O** use Docker
```bash
docker build -t tady -f docker/Dockerfile .
```

## Conjuntos de Datos (Datasets)

Descargue nuestros conjuntos de datos procesados o repita los pasos de preprocesamiento.

Proporcionamos los binarios en `bin.tar.gz`, que deben descomprimirse en `data/bin`, y los datos de verdad fundamental (ground truth) `gt_npz.tar.gz`, que deben descomprimirse en `data/gt_npz`. Con estos dos directorios, no es necesario descargar los conjuntos de datos originales.

### Descarga

### Para reproducir los conjuntos de datos desde su proveedor
Descargue los conjuntos de datos raw
```bash
bash data/download/download_dataset.sh
```

Construya el conjunto de datos rw; los archivos fuente se encuentran en `source.tar.gz`. Descomprímalo en `data/source`.
```
docker run -it -v $PWD:/work -w /work --name gt bin2415/x86_gt:0.1 /bin/bash
# Dentro de Docker
python3 -m pip install hydra-core hydra-joblib-launcher
python3 scripts/dataset/sok/compile.py -m project=curl,diffutils,gmp,ImageMagick,libmicrohttpd,libtomcrypt,openssl,putty,sqlite,zlib compiler=clang32 opt=O0,O1,O2,O3
python3 scripts/dataset/gather_binaries.py --source data/install --target data/raw/rw
bash extract_gt/run_extract_linux.sh -d /work/data/raw/rw/ -s extract_gt/extractBB.py -p gtBlock -j 24
```

Pangine
```bash
# Recolectar Binarios
uv run scripts/dataset/gather_binaries.py --source data/download/pangine-dataset --target data/bin/pangine
# Procesar Etiquetas
uv run  scripts/dataset/pangine/sqlite_gt.py --input data/download/pangine-dataset --output data/gt_npz/pangine --process 24
```

Assemblage:
```bash
# Recolectar Binarios
uv run scripts/dataset/gather_binaries.py --source data/download/assemblage-dataset --target data/bin/assemblage
# Procesar Etiquetas
# pdb-markers se construye desde https://github.com/GrammaTech/disasm-benchmark/tree/main/pdb
uv run scripts/dataset/assemblage/pdb_gt.py --bin_dir data/bin/assemblage/ --pdb_dir data/download/assemblage-dataset --mapping_file scripts/dataset/assemblage/assemblage_locations.json --process 24 --output_dir data/gt_npz/assemblage --executable scripts/dataset/assemblage/pdb-markers
```

x86_sok
```bash
# Recolectar Binarios
uv run scripts/dataset/gather_binaries.py --source data/download/x86-dataset --target data/bin/x86_dataset
# Procesar Etiquetas
uv run scripts/dataset/sok/parse_sok.py --dir data/bin/x86_dataset/ --pb_dir data/download/x86-dataset --output data/gt_npz/x86_dataset --process 24
```

rw
```bash
# Recolectar Binarios
uv run scripts/dataset/gather_binaries.py --source data/install --target data/bin/rw
# Procesar Etiquetas
uv run scripts/dataset/sok/parse_sok.py --dir data/bin/rw/ --pb_dir data/raw/rw --output data/gt_npz/rw --process 24
```

quarks
```bash
# Recolectar Binarios
uv run scripts/dataset/quarks/quokka_gt.py --source data/download/quarks-dataset --target data/bin/quarks
# Procesar Etiquetas
uv run scripts/dataset/quarks/quokka_gt.py --source data/download/quarks-dataset --target data/gt_npz/quarks --gt=True
```

obf-benchmark
```bash
# Recolectar Binarios
uv run scripts/dataset/obf-benchmark/obf_gt.py --source data/download/obf-dataset --target data/bin/obf-benchmark --obf=True
# Procesar Etiquetas
uv run scripts/dataset/obf-benchmark/obf_gt.py --source data/download/obf-dataset --target data/gt_npz/obf-benchmark --gt=True
```

Eliminar símbolos de los binarios (Strip)
```bash
uv run scripts/dataset/strip.py --dir data/bin --process 24
```

Preprocesar al formato de conjuntos de datos de huggingface para el entrenamiento
```bash
uv run scripts/experiments/preprocess.py -m dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks
```

## Entrenamiento (Train)
```bash
# Entrenar el modelo Tady (lite, all) y variantes para el estudio de ablación
uv run scripts/experiments/train.py -m dataset=pangine epoch=1 process=16 model.attention=lite,sliding connections=none,all
# Entrenar el modelo TadyA, misma arquitectura que Tady pero sobre una mezcla de los ajustes de dataset en scripts/experiments/conf/dataset/mix_all.yaml
uv run scripts/experiments/train.py -m dataset=mix_all epoch=1 process=16
```

Exportar los modelos al formato TF SavedModel para servicio (serving)
```bash
# Exportar modelo para Tady y ablaciones
uv run scripts/experiments/export.py -m dataset=pangine  model.attention=lite,sliding connections=all,none
# Exportar modelo para TadyA
uv run scripts/experiments/export.py dataset=mix_all
```

## Evaluación (Eval)
Los datos para las tablas se proporcionan en `artifacts.tar.gz`, los cuales pueden reproducirse con los comandos listados a continuación.

### Tady

Servir los modelos
```bash
docker run --rm --gpus device=0 -p 8500:8500 -v  $PWD/models/tf_models:/models -t --name tensorflow-serving tensorflow/serving:latest-gpu --xla_gpu_compilation_enabled=true --enable_batching=true --batching_parameters_file=/models/batching.conf --model_config_file=/models/model.conf
```

Probar sobre conjuntos de datos
```bash
# Para Tady y ablaciones
uv run scripts/experiments/eval.py -m dataset=pangine test_dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks process=24 model.attention=lite,sliding connections=all,none num_samples=1000
# Para TadyA
uv run scripts/experiments/eval.py -m dataset=mix_all test_dataset=pangine,assemblage,x86_dataset,rw,obf-benchmark,quarks process=24 num_samples=1000
```

Detenga tensorflow-serving después de las pruebas para liberar la memoria de la GPU.
```bash
docker stop tensorflow-serving
```

### Líneas Base (Baselines)

Reproducir los resultados de desensamblado es costoso en tiempo para estas líneas base; proporcionamos `eval_strip_baselines.tar.gz` que almacena los resultados de desensamblado de las líneas base. Debe descomprimirse en `data/eval_strip`. Por supuesto, siéntase libre de reproducir los resultados si tiene suficientes recursos.

**ddisasm**
```bash
# Construir el docker para la línea base ddisasm, agregar dependencia numpy
docker build -t baseline_ddisasm scripts/baselines/ddisasm
docker run -it -d -v $PWD:/work --gpus all --name ddisasm baseline_ddisasm /bin/bash
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ddisasm process=8 num_samples=1000
```

**deepdi**
```bash
git clone https://github.com/DeepBitsTechnology/DeepDi.git
docker build -t deepdi -f DeepDi/Dockerfile-gpu DeepDi
docker build -t baseline_deepdi scripts/baselines/DeepDi
docker run -it -d -v $PWD:/work --gpus all --name deepdi baseline_deepdi /bin/bash
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=deepdi process=1 num_samples=1000
```

**ghidra**

Descargue [Ghidra](https://github.com/NationalSecurityAgency/ghidra/releases/download/Ghidra_11.3.2_build/ghidra_11.3.2_PUBLIC_20250415.zip) y descomprímalo en data/tools/ghidra
```bash
uv pip install -f data/tools/ghidra/Ghidra/Features/PyGhidra/pypkg/dist/ pyghidra
uv pip install -f data/tools/ghidra/docs/ghidra_stubs/ ghidra-stubs
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ghidra process=24 num_samples=1000
```

**ida**

Asegúrese de tener IDA pro 9.1 instalado en su máquina
```bash
uv pip install $PATH_TO_IDA_PRO/idalib/python/
uv run $PATH_TO_IDA_PRO/idalib/python/py-activate-idalib.py
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=ida process=24 num_samples=1000
```

**xda**

Descargue xda_model_reproduce.tar.gz en artifacts y descomprímalo en scripts/baselines/XDA
```bash
tar -zxvf xda_model_reproduce.tar.gz -C scripts/baselines/XDA
```

Prepare el entorno
```bash
git clone https://github.com/CUMLSec/XDA.git && cd XDA
conda create -n xda python=3.7 numpy scipy scikit-learn colorama
conda activate xda
conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
pip install --editable .
pip install lief
```

Evaluación
```bash
# Regresar a la raíz del proyecto
uv run scripts/experiments/eval.py -m test_dataset=pangine,assemblage,x86_dataset,obf-benchmark,quarks model_id=xda process=24 num_samples=1000
```

### Prune (Poda)
```bash
# Para configurar los desensambladores a probar, actualice la lista en scripts/experiments/conf/prune.yaml models
uv run scripts/experiments/prune.py -m test_dataset=obf-benchmark,rw,x86_dataset,pangine,quarks,assemblage process=24 num_samples=1000
# Recolectar estadísticas para la tabla 1, tabla 3 y tabla 5; resultados en data/prune/all_prune_result.json
uv run scripts/experiments/collect_stat.py # Esto genera artifacts/all_prune_result_table3_5.json
# Para la tabla 2, probamos sobre todos los archivos disponibles en el dataset sin muestreo
uv run scripts/experiments/prune.py -m test_dataset=obf-benchmark,rw,x86_dataset,pangine,quarks,assemblage process=24 models="[gt]"
```

### Errores
```bash
# Esto resumirá los errores detectados para la tabla 1, tabla 2 y tabla 4; para configurar los desensambladores a resumir, actualice la lista en scripts/experiments/conf/error_stat.yaml models; los resultados están en data/error/error_stat.json
uv run scripts/experiments/error_stat.py # Esto genera artifacts/error_stat_table1_2_4.json
```
Después de ejecutar el script anterior, los detalles de los errores detectados se pueden encontrar en `data/error/{dataset}/{disassembler}_error_stat.log`


### Eficiencia

Seleccionar muestras
```bash
mkdir artifacts
uv run scripts/ablation/sample_select.py --dir data/bin/x86_dataset/ # Esto genera artifacts/selected_samples.json
```

Benchmark para los desensambladores
```bash
# Asegúrese de que los docker de ddisasm y deepdi estén ejecutándose
# Tensorflow-serving toma agresivamente toda la memoria de la GPU, por lo que necesitamos evaluar tady después de terminar los demás
uv run scripts/experiments/bench.py process=1 dataset=pangine test_dataset=x86_dataset
# Iniciar tensorflow-serving y ejecutar nuevamente
uv run scripts/experiments/bench.py process=1 dataset=pangine test_dataset=x86_dataset
# El resultado en data/eval_strip/x86_dataset/benchmark_cache.json proporciona los datos para dibujar la Figura 8, que es artifacts/benchmark_cache_figure8.json
```

Benchmark Tady
```bash
uv run scripts/ablation/model_efficiency.py --samples artifacts/selected_samples.json --model_id instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 --batch_size 32 --disassembler cpp --dir $PWD --output artifacts/benchmark_results.json --plot artifacts/time_vs_size_cpp.pdf # Esto genera los datos para dibujar la Figura 9, que es artifacts/benchmark_results_figure9.json
```

Benchmark PDT
```bash
# Preparar los puntajes para Pruning
uv run scripts/experiments/batch_run.py --files artifacts/selected_samples.json --dir data/bin --model instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 --output_dir artifacts/scores
uv run scripts/ablation/prune_efficiency.py --samples artifacts/selected_samples.json --output artifacts/prune --model_id=instruction_cpp_pangine_lite_all_64lw_64rw_16h_2l_prev000 # Esto genera los datos para las Figuras 10 y 11, que es artifacts/prune_data_figure10_11.json
```

### VMProtect
Generar la verdad fundamental en formato npz para el binario con la etiqueta dada.
```bash
uv run data/obf/manual_labels_to_gt.py --labels data/obf/labels.txt --bin data/obf/TestApp.vmp.exe
```
El objetivo está en la sección `.vmp0`, especificamos esto manualmente para los desensambladores.

Tady
```bash
# Esto muestra un ejemplo de que nuestro modelo entrenado puede extrapolar a longitudes arbitrarias, aunque fue entrenado solo en 8192
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

Para evaluar los resultados
```bash
uv run data/obf/eval.py --gt data/obf/TestApp.vmp.exe.npz --pred data/obf/{disassembler}/TestApp.vmp.exe.npz
# Después de prune
uv run -m tady.prune --gt data/obf/TestApp.vmp.exe.npz --pred data/obf/{disassembler}/TestApp.vmp.exe.npz
```

## Citar

```
@inproceedings{qin2025tady,
  author       = {Siliang Qin and Fengrui Yang and Hao Wang and Bolun Zhang and Zeyu Gao and Chao Zhang and Kai Chen},
  title        = {Tady: A Neural Disassembler without Structural Constraint Violations},
  publisher    = {USENIX Association},
  booktitle    = {Proceedings of the 34th USENIX Conference on Security Symposium},
  year         = {2025},
  series       = {SEC '25},
  address      = {USA},
  location     = {Seattle, WA, USA},
}
```
