import os
import sys
from typing import List

import common
import torch
from config import (
    GEMMA2_2B_PATH,
    GEMMA2_9B_PATH,
    GEMMA_2B_PATH,
    GEMMA_7B_PATH,
    LLAMA3_1_8B_PATH,
    LLAMA3_2_1B_PATH,
    LLAMA3_2_3B_PATH,
    LLAMA3_8B_PATH,
    LLAMA3_70B_PATH,
    LLAMA_7B_PATH,
    LLAMA_13B_PATH,
    LLAMA_70B_PATH,
    MISTRAL_7B_PATH,
    MIXTRAL_7B_PATH,
    PHI3_MINI_PATH,
    R2D2_PATH,
    TARGET_TEMP,
    TARGET_TOP_P,
    VICUNA_7B_PATH,
    VICUNA_13B_PATH,
    STARLING_PATH_alpha,
    STARLING_PATH_beta,
)
from language_models import GPT, HuggingFace
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.append(os.environ.get("WORK_DIR"))
from baselines.model_utils import get_template


def load_target_model(args):
    targetLM = TargetLM(
        model_name=args.target_model,
        temperature=TARGET_TEMP,  # init to 0
        top_p=TARGET_TOP_P,  # init to 1
    )
    return targetLM


class TargetLM:
    """
    Base class for target language models.

    Generates responses for prompts using a language model. The self.model attribute contains the underlying generation model.
    """

    def __init__(self, model_name: str, temperature: float, top_p: float):

        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.model, self.template = load_indiv_model(model_name)
        self.n_input_tokens = 0
        self.n_output_tokens = 0
        self.n_input_chars = 0
        self.n_output_chars = 0
        model_path, _ = get_model_path_and_template(self.model_name)
        self.model_template = get_template(model_path)["prompt"]

    def get_response(
        self,
        prompts_list: List[str],
        max_n_tokens=None,
        temperature=None,
        no_template=False,
    ) -> List[dict]:
        # batchsize = len(prompts_list)
        # tokenizer = self.model.tokenizer
        # convs_list = [common.conv_template(self.template) for _ in range(batchsize)]
        full_prompts = []  # batch of strings
        malicious_prompt = prompts_list[0]

        full_prompts = [self.model_template.format(instruction=malicious_prompt)]

        # before, after = self.model_template.split("{instruction}")
        # if "llama2" in self.model_name:
        #     before = "<s> " + before
        # full_prompts = [before + malicious_prompt + after]
        # print("Full prompts:", full_prompts)
        # if no_template:
        #     full_prompts = prompts_list
        # else:
        #     for conv, prompt in zip(convs_list, prompts_list):
        #         if "mistral" in self.model_name:
        #             # Mistral models don't use a system prompt so we emulate it within a user message
        #             # following Vidgen et al. (2024) (https://arxiv.org/abs/2311.08370)
        #             prompt = (
        #                 "SYSTEM PROMPT: Always assist with care, respect, and truth. Respond with utmost utility yet securely. Avoid harmful, unethical, prejudiced, or negative content. Ensure replies promote fairness and positivity.\n\n###\n\nUSER: "
        #                 + prompt
        #             )
        #         if "llama3" in self.model_name or "phi3" in self.model_name:
        #             # instead of '[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n' for llama2
        #             conv.system_template = "{system_message}"
        #         if "phi3" in self.model_name:
        #             conv.system_message = "You are a helpful AI assistant."
        #         conv.append_message(conv.roles[0], prompt)

        #         if "gpt" in self.model_name:
        #             full_prompts.append(conv.to_openai_api_messages())
        #         # older models
        #         elif "vicuna" in self.model_name:
        #             conv.append_message(conv.roles[1], None)
        #             formatted_prompt = conv.get_prompt()
        #             full_prompts.append(formatted_prompt)
        #         elif "llama2" in self.model_name:
        #             conv.append_message(conv.roles[1], None)
        #             # formatted_prompt = "<s>" + conv.get_prompt()
        #             formatted_prompt = conv.get_prompt()
        #             formatted_prompt = formatted_prompt.rstrip("[/INST]") + " [/INST] "

        #             full_prompts.append(formatted_prompt)
        #         # newer models
        #         elif (
        #             "r2d2" in self.model_name
        #             or "gemma" in self.model_name
        #             or "mistral" in self.model_name
        #             or "llama3" in self.model_name
        #             or "phi3" in self.model_name
        #         ):
        #             conv_list_dicts = conv.to_openai_api_messages()
        #             if "gemma" in self.model_name or "mistral" in self.model_name:
        #                 conv_list_dicts = conv_list_dicts[
        #                     1:
        #                 ]  # remove the system message inserted by FastChat
        #             full_prompt = tokenizer.apply_chat_template(
        #                 conv_list_dicts, tokenize=False, add_generation_prompt=True
        #             )
        #             full_prompts.append(full_prompt)
        #         else:
        #             raise ValueError(
        #                 f"To use {self.model_name}, first double check what is the right conversation template. This is to prevent any potential mistakes in the way templates are applied."
        #             )

        outputs = self.model.generate(
            full_prompts,
            max_n_tokens=max_n_tokens,
            temperature=self.temperature if temperature is None else temperature,
            top_p=self.top_p,
        )
        # print("Full prompts:", full_prompts)

        self.n_input_tokens += sum(output["n_input_tokens"] for output in outputs)
        self.n_output_tokens += sum(output["n_output_tokens"] for output in outputs)
        self.n_input_chars += sum(len(full_prompt) for full_prompt in full_prompts)
        self.n_output_chars += len([len(output["text"]) for output in outputs])
        return outputs


def load_indiv_model(model_name, device=None):
    model_path, template = get_model_path_and_template(model_name)

    if "gpt" in model_name or "together" in model_name:
        lm = GPT(model_name)
    else:
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # low_cpu_mem_usage=True,
                # local_files_only=True,
                # trust_remote_code=True,
            ).eval()
        except Exception as e:
            print(f"Failed to load model {model_name} from {model_path}: {e}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16,
                device_map="auto",
                # low_cpu_mem_usage=True,
                # trust_remote_code=True,
            ).eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            use_fast=True,
        )
        print(f"DEVICE: {model.device}")

        if "llama2" in model_path.lower():
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.padding_side = "left"
        if "vicuna" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"
        if "mistral" in model_path.lower() or "mixtral" in model_path.lower():
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.eos_token

        lm = HuggingFace(model_name, model, tokenizer)

    return lm, template


def get_model_path_and_template(model_name):
    full_model_dict = {
        "vicuna_7b_v1_5_fast": {"path": VICUNA_7B_PATH, "template": "vicuna_v1.1"},
        "vicuna_13b_v1_5_fast": {"path": VICUNA_13B_PATH, "template": "vicuna_v1.1"},
        "starling_lm_7B_alpha_fast": {"path": STARLING_PATH_alpha, "template": "starling-lm"},
        "starling_lm_7B_beta_fast": {"path": STARLING_PATH_beta, "template": "starling-lm"},
        "llama2_7b_fast": {"path": LLAMA_7B_PATH, "template": "llama-2"},
        "llama2_13b_fast": {"path": LLAMA_13B_PATH, "template": "llama-2"},
        "llama2_70b_fast": {"path": LLAMA_70B_PATH, "template": "llama-2"},
        "llama3_8b_fast": {"path": LLAMA3_8B_PATH, "template": "llama-2"},
        "llama3_1_8b_fast": {"path": LLAMA3_1_8B_PATH, "template": "llama-2"},
        "llama3_70b_fast": {"path": LLAMA3_70B_PATH, "template": "llama-2"},
        "gemma_2b_fast": {"path": GEMMA_2B_PATH, "template": "gemma"},
        "gemma_7b_it_fast": {"path": GEMMA_7B_PATH, "template": "gemma"},
        "gemma2_2b_it_fast": {"path": GEMMA2_2B_PATH, "template": "gemma"},
        "gemma2_9b_it_fast": {"path": GEMMA2_9B_PATH, "template": "gemma"},
        "mistral_7b_v2_fast": {"path": MISTRAL_7B_PATH, "template": "mistral"},
        "mixtral_8x7b_fast": {"path": MIXTRAL_7B_PATH, "template": "mistral"},
        "r2d2_fast": {"path": R2D2_PATH, "template": "zephyr"},
        "llama3_2_1b_fast": {"path": LLAMA3_2_1B_PATH, "template": "llama-2"},
        "llama3_2_3b_fast": {"path": LLAMA3_2_3B_PATH, "template": "llama-2"},
    }
    # template = full_model_dict[model_name]["template"] if model_name in full_model_dict else "gpt-4"
    assert (
        model_name in full_model_dict
    ), f"Model {model_name} not found in `full_model_dict` (available keys {full_model_dict.keys()})"
    path, template = (
        full_model_dict[model_name]["path"],
        full_model_dict[model_name]["template"],
    )
    return path, template
