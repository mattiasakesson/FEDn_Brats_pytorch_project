#!/bin/bash
set -e

# Init seed
python client/entrypoint.py init_seed

# Make compute package
tar -czvf package.tgz client

echo "done"