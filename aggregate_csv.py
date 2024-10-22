import os
import pandas as pd
from datetime import datetime
import sys

model_size_dict = {
    "gemma_7b_it": 8537680896,
    "llama2_7b": 6738415616,
    "llama2_13b": 13015864320,
    "vicuna_7b_v1_5": 6738415616,
    "vicuna_13b_v1_5": 13015864320,
    "mixtral_8x7b": 46702792704,
    "llama3_8b": 8030261248,
    "llama3_1_8b": 8030261248,
    "llama3_2_3b": 3212749824,
    "llama3_2_1b": 1235814400,
    "r2d2": 7241732096,
    "starling_lm_7B_alpha": 7241748480,
    "gemma2_2b_it": 2614341888,
    "gemma2_9b_it": 9241705984,
}

def get_last_entry_from_csv(csv_path):
    """Get the last entry from a given CSV file."""
    try:
        df = pd.read_csv(csv_path)
        return df.iloc[-1]
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None


def find_folders_with_run_name(wandb_dir, wandb_run_name):
    """Find all folders that contain the specified wandb_run_name."""
    matching_folders = []
    for root, dirs, files in os.walk(wandb_dir):
        for dir_name in dirs:
            if dir_name.endswith(wandb_run_name):
                matching_folders.append(os.path.join(root, dir_name))
    return matching_folders


def find_csv_in_folder(folder_path):
    """Find the .csv file in the given folder."""
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".csv"):
            return os.path.join(folder_path, file_name)
    return None


def concatenate_last_entries(wandb_dir, wandb_run_name):
    """Concatenate the last entries of all CSV files into one."""
    matching_folders = find_folders_with_run_name(wandb_dir, wandb_run_name)
    if not matching_folders:
        print(f"No folders found with run name: {wandb_run_name}")
        return

    last_entries = []
    for folder in matching_folders:
        csv_file = find_csv_in_folder(folder)
        last_entry = get_last_entry_from_csv(csv_file)
        if last_entry is not None:
            last_entries.append(last_entry)

    if last_entries:
        result_df = pd.DataFrame(last_entries)
        return result_df
    else:
        print(f"No valid data to concatenate for run name: {wandb_run_name}")
        return None

def process_responses(runs_df, attack):
    """
    Renaming columns to match the rest of the codebase
    """
    if attack in ["PEZ"]:
        runs_df["best_msg"] = (
            runs_df["test_case_best"].astype(str).apply(lambda x: x.lstrip("['").rstrip("']"))
        )
    elif attack in ["maksyms_attack"]:
        runs_df["best_msg"] = (
            runs_df["best_msg"].astype(str).apply(lambda x: x.lstrip("['").rstrip("']"))
        )
    elif attack in ["GCG", "AutoDan", "PAIR", "TAP"]:
        runs_df["best_msg"] = (
            runs_df["test_case"].astype(str).apply(lambda x: x.lstrip("['").rstrip("']"))
        )
    elif attack in ["BEAST", "COLD"]:
        runs_df["best_msg"] = (
            runs_df["test_case"].astype(str)
        )
    return runs_df

def calculate_tokens_and_flops(runs_df, attack, model):
    """
    Calculating the number of tokens and FLOPs for each model/attack
    """

    if attack == "PRS":
        runs_df["#tokens"] = (
            runs_df.n_fwd_target_model_input_tokens
            + runs_df.n_target_model_output_tokens
        )
        runs_df["FLOPs"] = runs_df["#tokens"] * model_size_dict[model] * 2

    elif attack == "GCG":
        runs_df["#tokens"] = (
            runs_df.n_fwd_target_model_input_tokens
            + runs_df.n_bcwd_target_model_input_tokens
            * 2
            + runs_df.n_target_model_output_tokens
        )
        runs_df["FLOPs"] = runs_df["#tokens"] * model_size_dict[model] * 2

    elif attack == "AutoDan":
        target_tokens = (
            runs_df.n_fwd_target_model_input_tokens + runs_df.n_target_model_output_tokens
        )
        attacker_tokens = (
            runs_df.n_fwd_mutating_model_input_tokens + runs_df.n_mutating_model_output_tokens
        )
        runs_df["#tokens"] = target_tokens + attacker_tokens
        runs_df["FLOPs"] = (
            target_tokens * model_size_dict[model] * 2
            + attacker_tokens * model_size_dict["mixtral_8x7b"] * 2
        )
    elif attack == "PAIR":
        target_tokens = runs_df.n_input_tokens_target + runs_df.n_output_tokens_target
        attacker_tokens = runs_df.n_input_tokens_attacker + runs_df.n_output_tokens_attacker
        judge_tokens = runs_df.n_input_tokens_judge + runs_df.n_output_tokens_judge
        runs_df["#tokens"] = target_tokens + attacker_tokens + judge_tokens
        runs_df["FLOPs"] = (
            target_tokens * model_size_dict[model] * 2
            + attacker_tokens * model_size_dict["mixtral_8x7b"] * 2
            + judge_tokens * model_size_dict["mixtral_8x7b"] * 2
        )
    elif attack in ["BEAST", "COLD"]:
        runs_df["#tokens"] = runs_df.n_fwd_target_model_input_tokens
        runs_df["FLOPs"] = runs_df["#tokens"] * model_size_dict[model] * 2
    else:
        runs_df["#tokens"] = runs_df.n_input_tokens
        runs_df["FLOPs"] = runs_df["#tokens"] * model_size_dict[model] * 2

    return runs_df

def save_concatenated_csv(concatenated_df, wandb_run_name, base_results_dir):
    """Save the concatenated DataFrame to a CSV file with a timestamped name."""
    output_dir = os.path.join(base_results_dir, wandb_run_name)
    os.makedirs(output_dir, exist_ok=True)

    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{wandb_run_name}_{current_time}.csv")

    concatenated_df.to_csv(output_file, index=False)
    print(f"Concatenated CSV saved to {output_file}")

if __name__ == "__main__":
    # Get command-line arguments
    wandb_run_name = sys.argv[1]
    wandb_dir = sys.argv[2]
    results_dir = sys.argv[3]

    concatenated_df = concatenate_last_entries(wandb_dir, wandb_run_name)

    # Process response and compute FLOPs
    attacks_all = concatenated_df.method_name.values.tolist()
    attack_name = attacks_all[0]
    assert all(attacks_all[0] == x for x in attacks_all)

    models_all = concatenated_df.experiment_name.values.tolist()
    model_name = models_all[0].replace('_fast', '')
    assert all(models_all[0] == x for x in models_all)

    concatenated_df = process_responses(concatenated_df, attack=attack_name)
    concatenated_df = calculate_tokens_and_flops(concatenated_df, attack=attack_name, model=model_name)

    if concatenated_df is not None:
        save_concatenated_csv(concatenated_df, wandb_run_name, results_dir)
