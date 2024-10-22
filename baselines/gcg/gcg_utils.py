import math

import numpy as np
import torch
from tqdm import tqdm


########## GCG Utils ##########
def sample_control(
    control_toks,
    grad,
    search_width,
    not_allowed_tokens=None,
    jb_filter=None,
    tokenizer=None,
    window_size=None,
    threshold=None,
    behavior_ids=None,
    max_indices_check=32000,
):

    def check_new_window(tokens, position, new_token, window_size, threshold):
        start = max(0, position - window_size + 1)
        end = min(len(tokens), position + window_size)
        for i in range(start, end - window_size + 1):
            window = tokens[i : i + window_size].copy()
            window[position - i] = new_token  # Update only the affected token
            perplexity, entropy = jb_filter.calculate_window_metrics(window)

            if -1 * perplexity < threshold:
                return False, i

        return True, None

    if not_allowed_tokens is not None:
        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    control_toks = control_toks.to(grad.device)
    num_tokens = control_toks.size(0)
    candidates_per_token = math.ceil(search_width / num_tokens)
    

    
    top_sequences = []

    list_control_toks = control_toks.tolist()

    for position in tqdm(range(num_tokens), desc="Searching tokens passing the filter"):
        sorted_gradients = torch.argsort(-grad[position, :], descending=True)
        top_indices = sorted_gradients[:max_indices_check]

        top_sequences_inner = []

        for token_idx in top_indices:
            if len(top_sequences_inner) >= candidates_per_token:
                break
            modified_list = list_control_toks[:]
            modified_list[position] = token_idx.item()

            tmp_prev = torch.tensor(modified_list).unsqueeze(0)
            ids_check = torch.cat([behavior_ids, tmp_prev], dim=1).squeeze()

            sequence_text = tokenizer.decode(ids_check.tolist())
            tokens = jb_filter.tokenizer(sequence_text, add_special_tokens=False)["input_ids"]

            passed, failing_window = check_new_window(
                tokens,
                position,
                token_idx.item(),
                window_size,
                threshold,
            )

            if passed:
                modified_sequence = torch.tensor(
                    modified_list, dtype=torch.long, device=grad.device
                )
                top_sequences_inner.append(modified_sequence)

            else:
                pass

        top_sequences.extend(top_sequences_inner)
        print("len top_sequences", len(top_sequences))

    if top_sequences:
        return torch.stack(top_sequences)
    else:
        return None


def sample_control_original(
    control_toks, grad, search_width, topk=256, temp=1, not_allowed_tokens=None
):

    if not_allowed_tokens is not None:

        grad = grad.clone()
        grad[:, not_allowed_tokens.to(grad.device)] = grad.max() + 1

    top_indices = (-grad).topk(topk, dim=1).indices
    control_toks = control_toks.to(grad.device)

    original_control_toks = control_toks.repeat(search_width, 1)
    new_token_pos = torch.arange(
        0, len(control_toks), len(control_toks) / search_width, device=grad.device
    ).type(torch.int64)

    new_token_val = torch.gather(
        top_indices[new_token_pos], 1, torch.randint(0, topk, (search_width, 1), device=grad.device)
    )
    new_control_toks = original_control_toks.scatter_(1, new_token_pos.unsqueeze(-1), new_token_val)

    return new_control_toks


def get_nonascii_toks(tokenizer, device="cpu"):

    def is_ascii(s):
        return s.isascii() and s.isprintable()

    ascii_toks = []
    for i in range(3, tokenizer.vocab_size):
        if not is_ascii(tokenizer.decode([i])):
            ascii_toks.append(i)

    if tokenizer.bos_token_id is not None:
        ascii_toks.append(tokenizer.bos_token_id)
    if tokenizer.eos_token_id is not None:
        ascii_toks.append(tokenizer.eos_token_id)
    if tokenizer.pad_token_id is not None:
        ascii_toks.append(tokenizer.pad_token_id)
    if tokenizer.unk_token_id is not None:
        ascii_toks.append(tokenizer.unk_token_id)

    if "Baichuan2" in tokenizer.name_or_path:
        ascii_toks += [i for i in range(101, 1000)]

    return torch.tensor(ascii_toks, device=device)
