import gc
import json
import os
import random
import time

from functools import partial
import numpy as np
import torch

# https://huggingface.co/docs/accelerate/v0.11.0/en/memory#accelerate.find_executable_batch_size
from accelerate.utils import find_executable_batch_size
from torch.nn import CrossEntropyLoss
from tqdm import tqdm

from baselines.ppl_filter import Filter

from ..model_utils import get_template
from ..baseline import SingleBehaviorRedTeamingMethod
from ..wandb_logger import WandBLogger

from .arutils import AutoRegressor



torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================== GCG CLASS DEFINITION ============================== #
class BEAST_adaptive(SingleBehaviorRedTeamingMethod):
    def __init__(
        self,
        target_model,
        targets_path,
        num_steps=50,
        adv_string_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        allow_non_ascii=False,
        search_width=512,
        use_prefix_cache=True,
        eval_steps=10,
        eval_with_check_refusal=False,
        check_refusal_min_loss=0.1,
        early_stopping=False,
        early_stopping_min_loss=0.1,  # early stopping min loss criteria
        starting_search_batch_size=None,  # preset search_batch_size which will auto go half each time go OOM
        adaptive=True,
        target_len=-1,
        jb_filter_window_size=16,
        jb_filter_metric_name="perplexity",
        jb_filter_threshold=-3150,
        seed=0,
        budget=600000,
        wandb_run_name=None,
        **model_kwargs,
    ):
        """
        :param num_steps: the number of optimization steps to use
        :param adv_string_init: the initial adversarial string to start with
        :param allow_non_ascii: whether to allow non-ascii characters
        :param search_width: the number of candidates to sample at each step ("search_width" in the original GCG code)
        :param early_stopping: whether to stop early if the loss is less than early_stopping_min_loss
        """
        super().__init__(
            target_model=target_model, **model_kwargs
        )  # assigns self.model, self.tokenizer, and self.model_name_or_path

        # Increase number of steps for adaptive attack, given that it filters out a lot of candidates
        # if adaptive:
        #    num_steps *= 3

        self.wandb_run_name = wandb_run_name
        ## Initialize the filter
        self.adaptive = adaptive
        self.adaptive_warmup_epochs = 1
        if self.adaptive:
            base_path = "ngrams_results_final/joined_final"

            self.jb_filter = Filter(
                unigram_path=base_path
                             + "/arity=1/df_gutenberg_unigrams_dict_normalized_hashed.parquet",
                bigram_path=base_path
                            + "/arity=2/df_gutenberg_bigrams_dict_normalized_hashed.parquet",
                mutation_count=1,
                dummy_load=False
            )
            self.jb_filter_window_size = jb_filter_window_size
            self.jb_filter_metric_name = jb_filter_metric_name
            self.jb_filter_threshold = jb_filter_threshold

            filter_config = {
                "jb_filter_window_size": jb_filter_window_size,
                "jb_filter_metric_name": jb_filter_metric_name,
                "jb_filter_threshold": jb_filter_threshold
            }
            filter_partial_text = partial(self.jb_filter.apply_filter,
                                          window_size=jb_filter_window_size,
                                          metric_name=jb_filter_metric_name,
                                          threshold=jb_filter_threshold,
                                          return_metrics=True,
                                          verbose=False
                                          )
            print("Loaded the filter")
        else:
            self.jb_filter = None
            filter_config = None
            filter_partial_text = None



        self.ar = AutoRegressor(self.tokenizer, self.model, target_model['model_name_or_path'], budget=budget,
                                filter=filter_partial_text, filter_config=filter_config)
        self.ar.max_bs = 50
        self.params = {"top_p": 1., 'top_k': None, 'temperature': 1.,
                       'new_gen_length': 40, 'k1': 15, 'k2': 15, 'ngram': 1,
                        'multi_model_list': None}

        self.num_steps = num_steps
        self.adv_string_init = adv_string_init
        self.allow_non_ascii = allow_non_ascii
        self.search_width = search_width
        self.use_prefix_cache = use_prefix_cache

        with open(targets_path, "r", encoding="utf-8") as file:
            self.behavior_id_to_target = json.load(file)

        ### Eval Vars ###
        self.eval_steps = eval_steps
        self.eval_with_check_refusal = eval_with_check_refusal
        self.check_refusal_min_loss = check_refusal_min_loss
        self.early_stopping = early_stopping
        self.early_stopping_min_loss = early_stopping_min_loss
        self.starting_search_batch_size = starting_search_batch_size

        ## Instruction template
        self.target_len = target_len
        template = get_template(**target_model)["prompt"]
        print('Using template', template)
        self.template = template
        self.before_tc, self.after_tc = template.split("{instruction}")
        if 'llama-2' in target_model['model_name_or_path'].lower():
            self.before_tc = '<s>' + self.before_tc
            print('before_tc changed to', self.before_tc)

        if "llama-3" in self.model_name_or_path.lower():
            self.idx_to_val = {
                v: k.replace("Ġ", " ") for k, v in self.tokenizer.get_vocab().items()
            }
        else:
            self.idx_to_val = self.tokenizer.get_vocab().items()



        ## seed everything
        self.logging_dict = {
            "adaptive": adaptive,
            "use_prefix_cache": use_prefix_cache,
            "eval_steps": 1,
            "eval_with_check_refusal": True,
            "early_stopping": True,
            "jb_filter_window_size": jb_filter_window_size,
            "jb_filter_metric_name": jb_filter_metric_name,
            "jb_filter_threshold": jb_filter_threshold,
            "template": self.template,
            **model_kwargs,
        }

        self.logging_dict.update(self.params)

        self.seed = seed

        print("Using the seed", self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)

        print(f"DEVICE: {self.model.device}")

    def generate_test_cases_single_behavior(self, behavior, verbose=False, **kwargs):
        """
        Generate a test cases for the behavior

        :param behavior: the behavior to generate test cases for
        :param verbose: whether to print progress
        """
        ######### Tokens counting ##########
        self.n_fwd_target_model_input_tokens = 0
        self.n_fwd_target_model_cache_tokens = 0

        self.n_bcwd_target_model_input_tokens = 0  # tokens for which input we called backward pass
        self.n_bcwd_target_model_optim_tokens = (
            0  # subset of tokens in n_bcwd_target_model_input_tokens
        )
        self.n_bcwd_target_model_cache_tokens = 0  # same, but cache

        self.n_target_model_output_tokens = 0

        self.n_target_model_fwd_passes = 0
        self.n_target_model_bcwd_passes = 0

        # only counts evaluation passes
        self.n_fwd_target_model_input_tokens_eval = 0
        self.n_fwd_target_model_cache_tokens_eval = 0

        self.n_target_model_output_tokens_eval = 0

        self.n_target_model_fwd_passes_eval = 0

        self.count_filter_not_passed = 0
        self.count_not_passed = 0
        self.start_time = time.time()

        # starting search_batch_size, will auto reduce batch_size later if go OOM (resets for every new behavior)
        self.search_batch_size = (
            self.starting_search_batch_size
            if self.starting_search_batch_size
            else self.search_width
        )

        # ========== Behavior and Target str ==========
        behavior_dict = behavior
        behavior = behavior_dict["Behavior"]
        context_str = behavior_dict["ContextString"]
        behavior_id = behavior_dict["BehaviorID"]

        self.target = self.behavior_id_to_target[behavior_id][:self.target_len]
        behavior += " "
        if context_str:
            behavior = f"{context_str}\n\n---\n\n{behavior}"

        self.logging_dict.update(
            {
                "behavior": behavior,
                "behavior_id": behavior_id,
                "target": self.target,
                "behavior_no_context": behavior_dict["Behavior"],
                "context": behavior_dict["ContextString"]
            }
        )
        adaptive_str = "_adaptive" if self.adaptive else ""
        self.logger = WandBLogger(
                    args=self.logging_dict,
                    run_name=self.wandb_run_name,
                    project=f"harmbench.beast_attack.no_copyright.final_adaptive.llama2_system_message.8th_iteration{adaptive_str}"
           # args=self.logging_dict, project=f"harmbench.charizard.beast_attack.no_copyright.llama2_system_message.11th_iteration{adaptive_str}"
        )

        print(f"Behavior: {behavior_id} || Target: {self.target}")

        for param in self.model.parameters():
            param.requires_grad = False

        prompts = [self.before_tc + behavior]
        prefixes = [self.before_tc]

        y = self.ar.self_attack_chat_batch(prompts=prompts, target=self.target,
                                           after_tc=self.after_tc, logger=self.logger,
                                           prefixes=prefixes,
                                           **self.params)

        print('res_dict', y)

        test_case = self.template.format(instruction=behavior)

        logs = {
            "final_loss": None,
            "all_losses": None,
            "all_test_cases": None,
        }

        self.logger.finish()
        return [test_case], [logs]

    def compute_candidates_loss(self, search_batch_size, input_embeds, target_ids):
        if self.search_batch_size != search_batch_size:
            print(f"INFO: Setting candidates search_batch_size to {search_batch_size})")
            self.search_batch_size = search_batch_size
            torch.cuda.empty_cache()
            gc.collect()

        all_loss = []
        for i in range(0, input_embeds.shape[0], search_batch_size):
            with torch.no_grad():
                input_embeds_batch = input_embeds[i : i + search_batch_size]

                if self.use_prefix_cache:
                    # Expand prefix cache to match batch size
                    current_batch_size = input_embeds_batch.shape[0]
                    prefix_cache_batch = []
                    for i in range(len(self.prefix_cache)):
                        prefix_cache_batch.append([])
                        for j in range(len(self.prefix_cache[i])):
                            prefix_cache_batch[i].append(
                                self.prefix_cache[i][j].expand(current_batch_size, -1, -1, -1)
                            )
                    outputs = self.model(
                        inputs_embeds=input_embeds_batch,
                        past_key_values=prefix_cache_batch,
                    )
                    self.n_fwd_target_model_input_tokens += (
                        input_embeds_batch.shape[1] * input_embeds_batch.shape[0]
                    )  # len * batch_size
                    self.n_fwd_target_model_cache_tokens += (
                        self.cache_size * input_embeds_batch.shape[0]
                    )
                    self.n_target_model_fwd_passes += input_embeds_batch.shape[0]
                    self.n_target_model_output_tokens += (
                        target_ids.shape[1] * input_embeds_batch.shape[0]
                    )
                else:
                    outputs = self.model(inputs_embeds=input_embeds_batch)
                    self.n_fwd_target_model_input_tokens += (
                        input_embeds_batch.shape[1] * input_embeds_batch.shape[0]
                    )
                    self.n_target_model_fwd_passes += input_embeds_batch.shape[0]
                    self.n_target_model_output_tokens += (
                        target_ids.shape[1] * input_embeds_batch.shape[0]
                    )

            logits = outputs.logits

            # compute loss
            # Shift so that tokens < n predict n
            tmp = input_embeds_batch.shape[1] - target_ids.shape[1]
            shift_logits = logits[..., tmp - 1 : -1, :].contiguous()
            shift_labels = target_ids.repeat(input_embeds_batch.shape[0], 1)
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = loss.view(input_embeds_batch.shape[0], -1).mean(dim=1)
            all_loss.append(loss)

            del outputs, logits, loss
            torch.cuda.empty_cache()
            gc.collect()

        return torch.cat(all_loss, dim=0)