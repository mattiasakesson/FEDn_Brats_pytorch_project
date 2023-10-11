#!/bin/bash
set -e

# Init venv
python -m venv .assist-pytorch-venv

# Pip deps
.assist-pytorch-venv/bin/pip install --upgrade pip
.assist-pytorch-venv/bin/pip install -r requirements.txt
#git clone https://github.com/scaleoutsystems/fedn.git
cd fedn
git checkout release/v0.5.0
cd ..
.assist-pytorch-venv/bin/pip install -e fedn/fedn
