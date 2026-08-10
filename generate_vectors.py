"""
Generates steering vectors for each layer of the model by averaging the activations of all the positive and negative examples.

Example usage:
python generate_vectors.py --layers $(seq 0 31) --save_activations --use_base_model --model_size 7b --behaviors sycophancy
"""

import json
import torch as t
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os
from dotenv import load_dotenv
from llama_wrapper import LlamaWrapper
import argparse
from typing import List
from utils.tokenize import tokenize_llama_base, tokenize_llama_chat
from behaviors import (
    get_vector_dir,
    get_activations_dir,
    get_ab_data_path,
    get_open_ended_data_path,
    get_vector_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()


class ComparisonDataset(Dataset):
    def __init__(self, data_path, model_name_path, use_chat, behavior):
        with open(data_path, "r") as f:
            data = json.load(f)
        self.data = []
        for item in data:
            if behavior == "label_anchor" or item["function_tags"] == behavior:
                self.data.append(item)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_path
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.use_chat = use_chat

    def prompt_to_tokens(self, instruction, model_output):
        if self.use_chat:
            tokens = tokenize_llama_chat(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        else:
            tokens = tokenize_llama_base(
                self.tokenizer,
                user_input=instruction,
                model_output=model_output,
            )
        return t.tensor(tokens).unsqueeze(0)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        p_text = item["answer_matching_behavior"]
        n_text = item["answer_not_matching_behavior"]
        q_text = item["question"]
        to_tag = item["to_tags"]
        q_tokens = self.prompt_to_tokens(q_text, None)
        p_tokens = self.prompt_to_tokens(q_text, p_text)
        n_tokens = self.prompt_to_tokens(q_text, n_text)
        return q_tokens, p_tokens, n_tokens, to_tag

def generate_save_vectors_for_behavior(
    layers: List[int],
    save_activations: bool,
    behavior: List[str],
    model: LlamaWrapper,
):
    if behavior == "label_anchor":
        data_path = get_ab_data_path(behavior)
    else:
        data_path = get_open_ended_data_path(behavior)
    if not os.path.exists(get_vector_dir(behavior)):
        os.makedirs(get_vector_dir(behavior))
    if save_activations and not os.path.exists(get_activations_dir(behavior)):
        os.makedirs(get_activations_dir(behavior))

    model.set_save_internal_decodings(False)
    model.reset_all()

    pos_activations = dict([(layer, [[] for _ in range(8)]) for layer in layers])
    neg_activations = dict([(layer, [[] for _ in range(8)]) for layer in layers])

    dataset = ComparisonDataset(
        data_path,
        model.model_name_path,
        model.use_chat,
        behavior,
    )

    for q_tokens, p_tokens, n_tokens, to_tag in tqdm(dataset, desc="Processing prompts"):

        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, q_tokens.shape[-1]:, :].mean(dim=0).detach().cpu()
            pos_activations[layer][to_tag].append(p_activations)
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, q_tokens.shape[-1]:, :].mean(dim=0).detach().cpu()
            neg_activations[layer][to_tag].append(n_activations)

    for layer in layers:
        all_pos_layer = [t.stack(tag) if tag else t.zeros((4096)) for tag in pos_activations[layer]]
        all_neg_layer = [t.stack(tag) if tag else t.zeros((4096)) for tag in neg_activations[layer]]
        vec = [(pos - neg).mean(dim=0) for pos, neg in zip(all_pos_layer, all_neg_layer)]
        t.save(
            vec,
            get_vector_path(behavior, layer, model.model_name_path),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    model_size: str,
    behaviors: List[str],
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    model_size: size of the model to use, either "7b" or "13b"
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        size=model_size, use_chat=not use_base_model, model_type="llama"
    )
    for behavior in behaviors:
        t.cuda.empty_cache()
        generate_save_vectors_for_behavior(
            layers, save_activations, behavior, model
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(32)))
    parser.add_argument("--save_activations", action="store_true", default=False)
    parser.add_argument("--use_base_model", action="store_true", default=False)
    parser.add_argument("--model_size", type=str, choices=["7b", "13b"], default="7b")
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.model_size,
        args.behaviors
    )
