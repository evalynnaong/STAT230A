#!/bin/bash
set -e

source /Users/evie/miniconda3/etc/profile.d/conda.sh
conda activate proj_230

jupyter nbconvert --to notebook --execute --inplace eda.ipynb
jupyter nbconvert --to notebook --execute --inplace clean.ipynb
jupyter nbconvert --to notebook --execute --inplace model.ipynb

echo "Done."
