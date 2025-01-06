#!/bin/bash

# Change directory to script location
cd "$(dirname "$0")"/../../ || exit

# Setup project on NLPGPU
python3.10 -m venv venv
source ./venv/bin/activate
export HF_HOME="/nlp/data/ajayp/hiatus_huggingface_cache"
pip3 install -r requirements.txt
pip install datadreamer.dev==0.20.0

# Run evaluation (from run_eval.sh)
export sample="crossGenre" # crossGenre, perGenre-HRS1.1
export split="dev" # train, dev, test
echo "START TIME: $(date)"
python3 -u SIVs/datadreamer_lora_training/trainer.py
echo "END TIME: $(date)"
