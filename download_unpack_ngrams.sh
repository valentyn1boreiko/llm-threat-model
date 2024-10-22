#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh

conda activate llm_threat_model
pip install gdown
gdown 1HOhjxkjwi-gLG2-gAZ2u5ZxNrpCt3RU7
gdown 17Y2QJzDrsb5yUxLrY313ArAxTzM_s4dS

python -c "import zipfile; zipfile.ZipFile('ngrams_results_final.zip', 'r').extractall()" # or unzip ngrams_results_final.zip
rm ngrams_results_final.zip

