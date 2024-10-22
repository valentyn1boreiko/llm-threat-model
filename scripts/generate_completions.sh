#!/bin/bash

# Automatically estimate WORK_DIR to the repository's root directory
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$REPO_DIR
export WORK_DIR=$REPO_DIR

# Initialize default values
df_name=""
gpu_ids=0,1
huggingface_api_key=""
work_dir=$WORK_DIR

# Parse long options manually
while [[ $# -gt 0 ]]; do
    case "$1" in
        --df_name)
            df_name="$2"
            shift 2
            ;;
        --gpu_ids)
            gpu_ids="$2"
            shift 2
            ;;
        --huggingface_api_key)
            huggingface_api_key="$2"
            shift 2
            ;;
        --work_dir)
            work_dir="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if mandatory argument --wandb_run_name is provided
if [ -z "df_name" ]; then
    echo "Error: --df_name is required."
    exit 1
fi

# Check if Hugging Face API Key is provided, exit if not
if [ -z "$huggingface_api_key" ]; then
    echo "Error: Hugging Face API Key is required. Please provide it using the --huggingface_api_key argument."
    exit 1
fi

# Construct paths
results_dir="${work_dir}/results/"

# Print information
echo "Selected GPU IDs: $gpu_ids"
echo "Working directory: $work_dir"
echo "DF name: $df_name"
echo "Results directory: $results_dir"

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_threat_model

export CUDA_VISIBLE_DEVICES=$gpu_ids
export HUGGINGFACE_API_KEY=$huggingface_api_key

# Run the Python script with provided arguments
python generate_completions.py "$df_name" "$results_dir"
