#!/bin/bash
set -e

# Init seed
.assist-pytorch-venv/bin/python3.9 client/entrypoint.py init_seed

# Make compute package
tar -czvf package.tgz client

echo "done"
