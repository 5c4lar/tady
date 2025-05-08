#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages
import os

# Read the README.md file for the long description
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# Read the requirements.txt file for the dependencies
with open('requirements.txt', 'r', encoding='utf-8') as f:
    requirements = [line.strip() for line in f.readlines() if line.strip()]

setup(
    name="tady",
    version="0.1.0",
    author="",
    author_email="",
    description="Tady: A Neural Disassembler without Consistency Violations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    package_dir={"": "python"},
    packages=find_packages(where="python"),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    include_package_data=True,
    zip_safe=False,
)