#!/bin/bash

# Automatically estimate WORK_DIR to the repository's root directory
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$REPO_DIR

# Initialize default values
wandb_run_name=""
work_dir=$WORK_DIR

# Parse long options manually
while [[ $# -gt 0 ]]; do
    case "$1" in
        --wandb_run_name)
            wandb_run_name="$2"
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
if [ -z "$wandb_run_name" ]; then
    echo "Error: --wandb_run_name is required."
    exit 1
fi

# Construct paths
wandb_dir="${work_dir}/wandb/"
results_dir="${work_dir}/results/"

# Print information
echo "Working directory: $work_dir"
echo "WandB run name: $wandb_run_name"
echo "WandB directory: $wandb_dir"
echo "Results directory: $results_dir"

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_threat_model

# Run the Python script with provided arguments
python aggregate_csv.py "$wandb_run_name" "$wandb_dir" "$results_dir"
