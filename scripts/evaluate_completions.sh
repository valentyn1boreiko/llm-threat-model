#!/bin/bash

# Automatically estimate WORK_DIR to the repository's root directory
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
WORK_DIR=$REPO_DIR
export WORK_DIR=$REPO_DIR

# Initialize default values
df_path=""
model_id=""
gpu_ids=0,1
huggingface_api_key=""
base_path=""
batch_size=1
output_path=""
results_dir="$WORK_DIR/results"
work_dir=$WORK_DIR

# Parse long options manually
while [[ $# -gt 0 ]]; do
    case "$1" in
        --df_name)
            df_path="$2"
            shift 2
            ;;
        --results_dir)
            results_dir="$2"
            shift 2
            ;;
        --model_id)
            model_id="$2"
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
        --base_path)
            base_path="$2"
            shift 2
            ;;
        --batch_size)
            batch_size="$2"
            shift 2
            ;;
        --output_path)
            output_path="$2"
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

# Check if mandatory arguments are provided
if [ -z "$df_path" ]; then
    echo "Error: --df_path is required."
    exit 1
fi

if [ -z "$model_id" ]; then
    echo "Error: --model_id is required."
    exit 1
fi

if [ -z "$huggingface_api_key" ]; then
    echo "Error: Hugging Face API Key is required. Please provide it using the --huggingface_api_key argument."
    exit 1
fi

# Print information
echo "Selected GPU IDs: $gpu_ids"
echo "Working directory: $work_dir"
echo "DataFrame path: $df_path"
echo "Model ID: $model_id"
echo "Results directory: $results_dir"

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_threat_model

export CUDA_VISIBLE_DEVICES=$gpu_ids
export HUGGINGFACE_API_KEY=$huggingface_api_key
export WORK_DIR=$work_dir

# Construct the python command
python_cmd="python evaluate_completions.py --df_name \"$df_path\" --model_id \"$model_id\" --results_dir \"$results_dir\""

# Add optional arguments if provided
if [ -n "$base_path" ]; then
    python_cmd+=" --base_path \"$base_path\""
fi

if [ -n "$batch_size" ]; then
    python_cmd+=" --batch_size $batch_size"
fi

if [ -n "$output_path" ]; then
    python_cmd+=" --output_path \"$output_path\""
fi

# Run the Python script with provided arguments
eval $python_cmd
