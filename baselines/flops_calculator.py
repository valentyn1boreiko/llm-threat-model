model_size_dict = {
    "google/gemma-7b-it": 8537680896,
    "meta-llama/Llama-2-7b-chat-hf": 6_738_415_616,
    "meta-llama/Llama-2-13b-chat-hf": 13015864320,
    "lmsys/vicuna-13b-v1.5": 13015864320,
    "mistralai/Mixtral-8x7B-Instruct-v0.1": 46_702_792_704,
    "meta-llama/Meta-Llama-3-8B-Instruct": 8030261248,
    "meta-llama/Meta-Llama-3-8B": 8030261248,
    "meta-llama/Meta-Llama-3.1-8B-Instruct": 8030261248,
    "meta-llama/Llama-3.2-3B-Instruct": 3212749824,
    "meta-llama/Llama-3.2-1B-Instruct": 1235814400,
    "cais/zephyr_7b_r2d2": 7241732096,
    "berkeley-nest/Starling-LM-7B-alpha": 7241748480,
    "google/gemma-2-2b-it": 2614341888,
    "google/gemma-2-9b-it": 9241705984,
}


def calcualte_gcg_flops(model_size, fwd_input_tokens, fwd_output_tokens, bcwd_input_tokens):
    _tokens = fwd_input_tokens + fwd_output_tokens + 2 * bcwd_input_tokens
    _flops = 2 * model_size * _tokens

    return _flops


def calcualte_pair_flops(
    judge_model_size,
    attack_model_size,
    target_model_size,
    fwd_judge_input_tokens,
    fwd_judge_output_tokens,
    fwd_attack_input_tokens,
    fwd_attack_output_tokens,
    fwd_target_input_tokens,
    fwd_target_output_tokens,
):
    judge_tokens = (fwd_judge_input_tokens + fwd_judge_output_tokens) * judge_model_size
    attack_tokens = (fwd_attack_input_tokens + fwd_attack_output_tokens) * attack_model_size
    target_tokens = (fwd_target_input_tokens + fwd_target_output_tokens) * target_model_size
    _flops = 2 * (judge_tokens + attack_tokens + target_tokens)

    return _flops


def calculate_autodan_flops(
    target_model_size,
    mutate_model_size,
    fwd_target_input_tokens,
    fwd_target_output_tokens,
    fwd_mutate_input_tokens,
    fwd_mutate_output_tokens,
):
    target_tokens = (fwd_target_input_tokens + fwd_target_output_tokens) * target_model_size
    mutate_tokens = (fwd_mutate_input_tokens + fwd_mutate_output_tokens) * mutate_model_size
    _flops = 2 * (target_tokens + mutate_tokens)
    return _flops