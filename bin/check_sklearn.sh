#!/usr/bin/env bash

# Cause the script to exit if a single command fails
set -eo pipefail -v

# pip install animus
# pip install gym numpy optuna pandas pyglet scikit-learn

python examples/sklearn/sklearn_cem.py
python examples/sklearn/sklearn_optuna.py
python examples/sklearn/sklearn_run.py
