# DeepDi Eval
Build Docker following [guide](https://github.com/DeepBitsTechnology/DeepDi/tree/master)

```bash
git clone https://github.com/DeepBitsTechnology/DeepDi.git
cd DeepDi
docker build -t deepdi -f Dockerfile-gpu .
```

In our project root:
```bash
docker run -it --rm -v (pwd):/work --gpus all deepdi /bin/bash -c ' PYTHONPATH=. python3 /work/baselines/DeepDi/DeepDi.py --gpu --dir /work/test/obf-benchmark-obf/ --output /work/results/deepdi_obf --key aaf9bb2902c6d7eeaf5a8c7156ab77113a9d02db46e33edaf5f66dc53f8c7caa5c0d35a18ee8197250c06cad37eca340a47d79dee0ed266355999ec358a040f1 --process 1'
```