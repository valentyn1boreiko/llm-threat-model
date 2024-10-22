import argparse
import csv
import os
from os.path import join

import pandas as pd
import torch
import transformers
import yaml

from baselines import init_method
from eval_utils import get_experiment_config

# Set this to disable warning messages in the generation mode.
transformers.utils.logging.set_verbosity_error()

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Running red teaming with baseline methods.")
    parser.add_argument("--method_name", type=str, default='GCG',
                        help="The name of the red teaming method")
    parser.add_argument("--adaptive", action="store_true",
                        help="Flag to enable the adaptive attack")
    parser.add_argument("--experiment_name", type=str,
                        help="The name of the experiment (options are in configs/methods_config/{{method_name}}_config.yaml)")
    parser.add_argument("--wandb_run_name", type=str,
                        help="The name of the wandb run saved locally")
    parser.add_argument("--method_config_file", type=str, default=None,
                        help="The path to the config file with method hyperparameters. This will default to configs/method_configs/{{method_name}}_config.yaml")
    parser.add_argument("--models_config_file", type=str, default='./configs/model_configs/models.yaml',
                        help="The path to the config file with model hyperparameters")
    parser.add_argument("--behaviors_path", type=str, default='./data/400_behaviors.csv',
                        help="The path to the behaviors file")
    parser.add_argument("--save_dir", type=str, default='./test_cases',
                        help="The directory used for saving test cases")
    parser.add_argument("--behavior_start_idx", type=int, default=None,
                        help="Start index for behaviors_path data")
    parser.add_argument("--behavior_end_idx", type=int, default=None,
                        help="End index for behaviors_path data")
    parser.add_argument("--run_id", type=str, default=None,
                        help="An optional run id for output files, will be appended to output filenames")
    parser.add_argument("--overwrite", action="store_true",
                        help="Whether to overwrite existing saved test cases")
    parser.add_argument("--verbose", action="store_true",
                        help="Whether to print out intermediate results")
    parser.add_argument("--force_rerun", action="store_false", help="Whether to force rerun the test cases")
    parser.add_argument("--token", type=str, default="",
                    help="Token for Hugging Face private model or OpenAI key for GPT models")
    parser.add_argument("--jb_filter_window_size", type=int, default=8)
    args = parser.parse_args()
    return args


def main():
    # ========== load arguments and config ========== #
    args = parse_args()
    method_name = args.method_name
    adaptive = args.adaptive
    experiment_name = args.experiment_name
    wandb_run_name = args.wandb_run_name
    save_dir = os.path.abspath(args.save_dir)

    # Load model config file
    config_file = f"configs/model_configs/models.yaml" if not args.models_config_file else args.models_config_file
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    # Load method config file
    config_file = f"configs/method_configs/{method_name}.yaml" if not args.method_config_file else args.method_config_file
    with open(config_file) as file:
        method_configs = yaml.full_load(file)

    # Load default method parameters
    if 'default_method_hyperparameters' not in method_configs:
        raise ValueError(f"Default method hyperparameters not found in {config_file}")
    method_config = method_configs.get('default_method_hyperparameters') or {}

    # Update method parameters with experiment parameters
    experiment_config = get_experiment_config(experiment_name, model_configs, method_configs)
    method_config.update(experiment_config)

    # ========== generate test cases ========== #
    # load behaviors csv
    with open(args.behaviors_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        behaviors = list(reader)
        print("Total avaliable behaviors: ", len(behaviors))

        behavior_start_idx, behavior_end_idx = args.behavior_start_idx, args.behavior_end_idx
        if args.behavior_start_idx is not None and args.behavior_end_idx is not None:
            print(f"Selecting subset from [{behavior_start_idx}, {behavior_end_idx}]")
            behaviors = behaviors[args.behavior_start_idx:args.behavior_end_idx+1]
    
    print("Total behaviors left: ", len(behaviors))
    world_size = torch.cuda.device_count()

    print(f"============={method_name} Method Config=============")
    method_config['adaptive'] = adaptive
    method_config['method_name'] = method_name
    method_config['experiment_name'] = experiment_name
    method_config['wandb_run_name'] = wandb_run_name

    print('\n'.join([f"{k}: {v}" for k, v in method_config.items()]))
    print("=============GPU Allocation Configs=============")
    print(f"Total GPUs: {world_size}")
    print("=============")

    # ==== Init Method ====
    print("Method config", method_config)
    method = init_method(method_name, method_config)

    # ==== Filtering existed runs =====
    if not args.overwrite:
        filtered_behaviors = []
        for behavior in behaviors:
            # flatten csv row
            behavior_id = behavior['BehaviorID']
            test_case_file = method.get_output_file_path(save_dir, behavior_id, file_type='test_cases', run_id=args.run_id)

            if (not os.path.exists(test_case_file) or args.force_rerun):
                print(behavior_id)
                filtered_behaviors.append(behavior)
            else:
                print(f"===>Skipping {behavior_id}, force_rerun={args.force_rerun}")
        behaviors = filtered_behaviors

    # ====== Run attack ======
    print("#Behaviors to run:", len(behaviors))
    all_test_cases, all_logs = method.generate_test_cases(behaviors=behaviors, verbose=args.verbose)

    # ====== Save test cases and logs ======
    print("Saving outputs to: ", save_dir)
    method.save_test_cases(save_dir, all_test_cases, all_logs, method_config=method_config, run_id=args.run_id)
    print("Finished")


if __name__ == "__main__":
    main()