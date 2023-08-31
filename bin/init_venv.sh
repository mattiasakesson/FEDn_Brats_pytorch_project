#!/bin/bash
set -e

# Init venv
python3 -m venv .assist-pytorch-venv

# Pip deps
.assist-pytorch-venv/bin/pip install --upgrade pip
#.assist-pytorch-venv/bin/pip install -e ../../fedn
.assist-pytorch-venv/bin/pip install -r requirements.txt