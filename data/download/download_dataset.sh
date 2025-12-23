#!/bin/bash
set -x

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Pangines
# See https://github.com/pangine/disasm-benchmark?tab=readme-ov-file#datasets
if [ ! -d pangine-dataset ]; then
    if [ ! -f pangine-gt-data-20200701.tar.xz ]; then
        curl -L "https://drive.usercontent.google.com/download?id=1r7Xa1RY7DAhB58Xz6xSNVVZsM9EW8zJj&export=download&authuser=0&confirm=t" > pangine-gt-data-20200701.tar.xz
    fi
    tar -xf pangine-gt-data-20200701.tar.xz
    mv pangine-gt-data-20200701 pangine-dataset
fi
pushd $script_dir

# SOK
if [ ! -d x86-dataset ]; then
    if [ ! -f x86_dataset.tar.xz ]; then
        wget https://zenodo.org/records/6566082/files/x86_dataset.tar.xz
    fi
    tar -xf x86_dataset.tar.xz
    mv x86-dataset
fi

# Assemblage 
if [ ! -d assemblage-dataset ]; then
    git clone https://huggingface.co/datasets/changliu8541/Assemblage_PE
    pushd Assemblage_PE
        git checkout 9600eb2d7b81b0a367b4e6dc1d03f4454f87a17f
        git lfs fetch binaries.tar.xz
        tar -xf binaries.tar.xz
        mv binaries ../assemblage-dataset
    popd
fi

if [ ! -d obf-dataset ]; then
    wget https://www2.cs.arizona.edu/~debray/binary-obfuscation/obf-benchmarks.tar.gz
    tar -xf obf-benchmarks.tar.gz
    mv obf-benchmarks obf-dataset
fi

if [ ! -d quarks-dataset ]; then
    git clone git@github.com:quarkslab/diffing_obfuscation_dataset.git
    pushd diffing_obfuscation_dataset
        mkdir ../quarks-dataset
        uv run obfu-dataset-cli download-plain -r ../quarks-dataset -t 8
        uv run obfu-dataset-cli download-obfuscated -r ../quarks-dataset -l 100 -t 8 -p zlib
        uv run obfu-dataset-cli download-obfuscated -r ../quarks-dataset -l 100 -t 8 -p lz4
        uv run obfu-dataset-cli download-obfuscated -r ../quarks-dataset -l 100 -t 8 -p minilua
        uv run obfu-dataset-cli download-obfuscated -r ../quarks-dataset -l 100 -t 8 -p sqlite
        uv run obfu-dataset-cli download-obfuscated -r ../quarks-dataset -l 100 -t 8 -p freetype   
    popd
fi

RW-Sources
if [ ! -d rw-source ]; then
    wget https://zenodo.org/records/10300010/files/rw-source.tar.gz
    tar -xf rw-source.tar.gz
    # move to data/source for latter use
    mv rw-source ../source
fi

popd