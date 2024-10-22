import os
import sys
from datetime import datetime
from functools import partial

import pandas as pd
import torch
import yaml
from accelerate.utils import find_executable_batch_size
from huggingface_hub import login as hf_login
from tqdm import tqdm

from baselines.model_utils import get_template, load_model_and_tokenizer

hf_login(token=os.environ.get("HUGGINGFACE_API_KEY"))

def _hf_generate_with_batching(model, tokenizer, test_cases, template, **generation_kwargs):
    @find_executable_batch_size(starting_batch_size=1)
    def inner_generation_loop(batch_size):
        nonlocal model, tokenizer, test_cases, template, generation_kwargs
        generations = []
        for i in tqdm(range(0, len(test_cases), batch_size)):
            batched_test_cases = test_cases[i : i + batch_size]
            inputs = [template["prompt"].format(instruction=s) for s in batched_test_cases]
            print("inputs", inputs)
            inputs = tokenizer(inputs, return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            with torch.no_grad():
                outputs = model.generate(
                    inputs=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    **generation_kwargs,
                ).cpu()
            generated_tokens = outputs[:, inputs["input_ids"].shape[1] :]
            batch_generations = [
                tokenizer.decode(o, skip_special_tokens=True).strip() for o in generated_tokens
            ]
            print("batch_generations", batch_generations)
            generations.extend(batch_generations)
        return generations

    return inner_generation_loop()

def load_generation_function(
    config_file, model_name, max_new_tokens
):
    with open(config_file) as file:
        model_configs = yaml.full_load(file)

    num_gpus = model_configs[model_name].get("num_gpus", 0)
    # check num gpus available to see if greater than num_gpus in config
    num_gpus_available = torch.cuda.device_count()
    if num_gpus_available != num_gpus:
        print(
            f"Warning: num_gpus in config ({num_gpus}) does not match num_gpus available ({num_gpus_available}). Using {num_gpus_available} GPUs."
        )
        num_gpus = num_gpus_available
    model_config = model_configs[model_name]["model"]
    model_config["num_gpus"] = num_gpus
    model_config["dtype"] = "float16"

    model_name_or_path = model_config["model_name_or_path"]

    print("Using HF generation")
    model, tokenizer = load_model_and_tokenizer(**model_config)
    generation_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=False)
    TEMPLATE = get_template(
        model_name_or_path, chat_template=model_config.get("chat_template", None)
    )
    return partial(
        _hf_generate_with_batching,
        model=model,
        tokenizer=tokenizer,
        template=TEMPLATE,
        **generation_kwargs,
    )

def generate_main(test_cases, generation_function):

    # ==== Generate ====
    print("Generating completions...")
    generations = generation_function(test_cases=test_cases)
    print("Done")

    return generations

def generate_completions(df, model_name):

    generation_function = load_generation_function(
        config_file=os.environ.get("WORK_DIR") + "/configs/model_configs/models.yaml",
        model_name=model_name,
        max_new_tokens=512,
    )

    msg_list = df.best_msg.values.tolist()

    # Generate completions for the missing messages
    temp = generate_main(msg_list, generation_function)

    assert len(df) == len(temp), "Length of generated completions does not match the length of the DataFrame"

    df['final_completion'] = temp

    return df

def get_df(results_dir, df_name):
    """Concatenate the last entries of all CSV files into one."""
    folder_name = '_'.join(df_name.split("_")[:-2])
    if not df_name.endswith('.csv'):
        df_name = df_name + '.csv'

    path = os.path.join(results_dir, folder_name, df_name)
    df = pd.read_csv(path)

    return df, path

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
    df_name = sys.argv[1]
    results_dir = sys.argv[2]

    print("Generating completions for", df_name)

    concatenated_df, df_path = get_df(results_dir, df_name)

    # Generate completions
    models_all = concatenated_df.experiment_name.values.tolist()
    model_name = models_all[0]
    assert all(models_all[0] == x for x in models_all)

    concatenated_df = generate_completions(concatenated_df, model_name)

    concatenated_df.to_csv(df_path, index=False)

    print("The file is saved to", df_path)
