#!/bin/bash

# Initialize conda for the current shell
source $(conda info --base)/etc/profile.d/conda.sh

# Update conda and install cudatoolkit
conda update -n base -c conda-forge conda -y
conda install -n base -c conda-forge cudatoolkit-dev -y

# Create the actual target conda environment from the environment.yml file
conda env create -f environment.yml

conda activate llm_threat_model

pip install -r requirements.txt

python -m spacy download en_core_web_sm
