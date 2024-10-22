#!/bin/bash

# Set WORK_DIR to the repository's root directory (assuming the script is located in a subfolder)
REPO_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export WORK_DIR=$REPO_DIR

echo "WORK_DIR is set to: $WORK_DIR"

# Set default values
adaptive=false
gpu_ids=0,1
method_name="BEAST"
huggingface_api_key=""
experiment_name="llama2_13b_fast"
start_idx=0
end_idx=1000
behaviors_path="./data/400_behaviors_no_copyright.csv"
wandb_run_name=None
delete_prev_wandb_run_name=None

# Parse long options manually
while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu_ids)
            gpu_ids="$2"
            shift 2
            ;;
        --method_name)
            method_name="$2"
            shift 2
            ;;
        --huggingface_api_key)
            huggingface_api_key="$2"
            shift 2
            ;;
        --experiment_name)
            experiment_name="$2"
            shift 2
            ;;
        --start_idx)
            start_idx="$2"
            shift 2
            ;;
        --end_idx)
            end_idx="$2"
            shift 2
            ;;
        --behaviors_path)
            behaviors_path="$2"
            shift 2
            ;;
        --adaptive_flag)
            adaptive=true
            shift
            ;;
        --target_TPR)
            target_TPR="$2"
            shift 2
            ;;
        --wandb_run_name)
            wandb_run_name="$2"
            shift 2
            ;;
        --delete_prev_wandb_run_name)
            delete_prev_wandb_run_name="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

# Check if Hugging Face API Key is provided, exit if not
if [ -z "$huggingface_api_key" ]; then
    echo "Error: Hugging Face API Key is required. Please provide it using the --huggingface_api_key argument."
    exit 1
fi

# Handle deletion of previous W&B runs if a valid name is provided
if [[ -n "$delete_prev_wandb_run_name" && "$delete_prev_wandb_run_name" != "None" ]]; then
    echo "Attempting to delete previous W&B run data for: $delete_prev_wandb_run_name"

    # Make sure the pattern isn't dangerous
    if [[ "$delete_prev_wandb_run_name" =~ ^[a-zA-Z0-9_-]+$ ]]; then
        find wandb/ -name "*$delete_prev_wandb_run_name" -exec rm -rf {} \;
	echo "Deleted W&B run data matching: $delete_prev_wandb_run_name"
    else
        echo "Warning: Invalid W&B run name pattern, deletion skipped."
    fi
fi

# Define the save directory
base_dir="results"
experiment="debug"
save_dir="${base_dir}/${method_name}/${experiment}/test_cases_BEAST_debug_control_llama2_7b_fast"

# Print the input values for confirmation
echo "Selected GPU IDs: $gpu_ids"
echo "Method Name: $method_name"
echo "Experiment Name: $experiment_name"
echo "Behaviors Path: $behaviors_path"
echo "Save Directory: $save_dir"
echo "Behavior Start Index: $start_idx"
echo "Behavior End Index: $end_idx"
echo "Adaptive Method Enabled: $adaptive"

# Set environment variables
export CUDA_VISIBLE_DEVICES=$gpu_ids
export HUGGINGFACE_API_KEY=$huggingface_api_key

# Activate the conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm_threat_model

# Construct the command differently based on method_name
# Construct the command for PRS based on adaptive flag
if [ "$method_name" = "PRS" ]; then
    # Set script name based on adaptive flag
    script_name="main.py"
    if [ "$adaptive" = true ] && [ "$method_name" = "PRS" ]; then
        script_name="main_adaptive_attack.py"
    fi

    cmd="python -u baselines/PRS/llm-adaptive-attacks/$script_name \
        --n-iterations 10000 \
        --start_index $start_idx \
        --end_index $end_idx \
        --prompt-template 'refined_best' \
        --target-model $experiment_name \
        --n-tokens-adv 25 \
        --n-tokens-change-max 4 \
        --schedule_prob \
        --dataset 'harmbench' \
        --judge-max-n-calls 10 \
        --wandb_run_name $wandb_run_name"
else
    # Default command for other method names
    cmd="python -u generate_test_cases.py \
        --method_name $method_name \
        --experiment_name $experiment_name \
        --behaviors_path $behaviors_path \
        --save_dir $save_dir --verbose \
        --behavior_start_idx $start_idx \
        --behavior_end_idx $end_idx \
        --wandb_run_name $wandb_run_name"

    # Add --adaptive flag to the command if adaptive is set to true
    if [ "$adaptive" = true ]; then
        cmd="$cmd --adaptive"
    fi
fi

# Print the full command for confirmation and run it
echo "Executing command: $cmd"
eval $cmd
