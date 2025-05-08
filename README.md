# tady
Tady: A Neural Disassembler without Consistency Violations

## Dev Environment
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
pip install -e .
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
