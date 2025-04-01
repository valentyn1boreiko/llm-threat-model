## ☕ Quick Start ☕
You might need to call `chmod +777 script_name.sh` on your .sh scripts.

### ⚙️ Installation
Clone the repository and do the following commands:

```bash
cd llm-threat-model 
./install_environment.sh
./download_unpack_ngrams.sh
```

## 🗡️ Step 1 - Run Attacks

The first step is to execute a selected attack method on a specified model. 
 
These models can be found in the corresponding model configurations defined in the YAML files under `configs/model_configs/`.

<details><summary> Command Breakdown: </summary><p>
To run an attack on a model, you need to specify the following:

1. **gpu_ids**: GPU IDs used for model execution (e.g., 0,1,2).
2. **method_name**: The name of the attack method you wish to run (e.g., BEAST).
3. **huggingface_api_key**: The API key for accessing Hugging Face models (replace `YOUR_TOKEN_HERE` with your actual API key).
4. **experiment_name**: The name of the experiment, typically referring to the model (e.g., vicuna_13b_v1_5_fast).
5. **adaptive_flag**: (Optional) If included, enables the adaptive attack.
6. **wandb_run_name**: The name of the wandb run name to use in the next step.
7. **delete_prev_wandb_run_name**: (Optional) If included, it cleans the previous results with the same name to save space.

</p></details>

### Command:
```bash
# Run the BEAST attack on the Vicuna model with specified behaviors
./scripts/run_attack.sh \
    --gpu_ids 0,1,2 \
    --method_name BEAST \
    --huggingface_api_key YOUR_TOKEN_HERE \
    --experiment_name vicuna_7b_v1_5_fast \
    --adaptive_flag \
    --wandb_run_name vicuna_7b_v1_5_fast_BEAST \
    --delete_prev_wandb_run_name vicuna_7b_v1_5_fast_BEAST > log_BEAST
```

## 🔄 Step 2 - Aggregate the Results

In this step, aggregate the attack outcomes using the same `wandb_run_name` as in Step 1. This ensures all generated data is summarized for analysis.

### Command:
```bash
./scripts/aggregate_results.sh --wandb_run_name vicuna_7b_v1_5_fast_BEAST
```

## 📊 Step 3 - Generate Completions

Generate model completions based on a specified DataFrame of jailbreak attempts from the previous step, using configurations saved under the `./results/` directory.

### Command:
```bash
./scripts/generate_completions.sh \
    --gpu_ids 0,1,2 \
    --df_name DF_NAME \
    --huggingface_api_key YOUR_TOKEN_HERE 
```
DF_NAME is the name of a .csv file generated in step 2. For example `--df_name gemma2_2b_it_fast_PRS_20241018_104852.csv`.

## 🔍 Step 4 - Evaluate with a Judge 

Generate evaluation with the HarmBench judge **cais/HarmBench-Llama-2-13b-cls** [[1]](#-acknowledgements-and-citation-) based on a specified DataFrame from the previous steps. For example `--df_name gemma2_2b_it_fast_PRS_20241018_104852.csv`. 

### Command:
```bash
./scripts/evaluate_completions.sh \
    --gpu_ids 0,1,2 \
    --df_name DF_NAME \
    --huggingface_api_key YOUR_TOKEN_HERE \
    --model_id cais/HarmBench-Llama-2-13b-cls 
```

### ➕ Using your own model 
You can easily add new Hugging Face transformers models in [configs/model_configs/models.yaml](configs/model_configs/models.yaml) by simply adding an entry for your model. This model can then be directly evaluated on most red teaming methods without modifying the method configs. 

### ➕ Adding/Customizing your own red teaming methods in HarmBench
All of the red teaming methods are implemented in [baselines](baselines), imported through [baselines/__init__.py](baselines/__init__.py), and managed by [configs/method_configs](configs/method_configs). You can easily improve on top of existing red teaming methods or add new methods by simply making a new subfolder in the `baselines` directory.
