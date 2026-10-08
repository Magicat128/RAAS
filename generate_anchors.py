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
    get_anchor_dir,
    get_activations_dir,
    get_ab_data_path,
    get_open_ended_data_path,
    get_vector_path,
    get_anchor_path,
    get_activations_path,
    ALL_BEHAVIORS
)

load_dotenv()

#HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")

def fast_tensor_in_list(target_tensor, tensor_list):
    if not tensor_list:
        return False
    
    stacked_tensors = t.stack(tensor_list).to("cuda")
    target_tensor = target_tensor.to("cuda")
    
    if target_tensor.is_floating_point():
        comparison_mask = t.isclose(stacked_tensors, target_tensor, rtol=1e-04, atol=1e-05)
        is_match_per_row = comparison_mask.all(dim=1)
    else:
        comparison_mask = (stacked_tensors == target_tensor)
        is_match_per_row = comparison_mask.all(dim=1)

    return is_match_per_row.any().item()

class ComparisonDataset(Dataset):
    def __init__(self, data_path, model_name_path, use_chat, behavior):
        with open(data_path, "r") as f:
            data = json.load(f)
        self.data = []
        for item in data:
            if item["function_tags"] == behavior:
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
        # pos
        model.reset_all()
        model.get_logits(p_tokens)
        for layer in layers:
            p_activations = model.get_last_activations(layer)
            p_activations = p_activations[0, p_tokens.shape[-1]:, :].mean(dim=0).detach().cpu()
            if not fast_tensor_in_list(p_activations, pos_activations[layer][to_tag]):
                pos_activations[layer][to_tag].append(p_activations)
        # neg
        model.reset_all()
        model.get_logits(n_tokens)
        for layer in layers:
            n_activations = model.get_last_activations(layer)
            n_activations = n_activations[0, q_tokens.shape[-1]:, :].mean(dim=0).detach().cpu()
            if not fast_tensor_in_list(n_activations, neg_activations[layer][to_tag]):
                neg_activations[layer][to_tag].append(n_activations)

    for layer in layers:
        # pos anchors
        all_pos_layer = [t.stack(tag) if tag else t.zeros((4096)) for tag in pos_activations[layer]]
        vec = [pos.mean(dim=0) for pos in all_pos_layer]
        t.save(
            vec,
            get_anchor_path(behavior, layer, "pos", model.model_name_path),
        )
        if save_activations:
            t.save(
                all_pos_layer,
                get_activations_path(behavior, layer, model.model_name_path, "pos"),
            )
        # neg anchors
        all_neg_layer = [t.stack(tag) if tag else t.zeros((4096)) for tag in neg_activations[layer]]
        vec = [neg.mean(dim=0) for neg in all_neg_layer]
        t.save(
            vec,
            get_anchor_path(behavior, layer, "neg", model.model_name_path),
        )
        if save_activations:
            t.save(
                all_neg_layer,
                get_activations_path(behavior, layer, model.model_name_path, "neg"),
            )

def generate_save_vectors(
    layers: List[int],
    save_activations: bool,
    use_base_model: bool,
    behaviors: List[str],
):
    """
    layers: list of layers to generate vectors for
    save_activations: if True, save the activations for each layer
    use_base_model: Whether to use the base model instead of the chat model
    behaviors: behaviors to generate vectors for
    """
    model = LlamaWrapper(
        use_chat=not use_base_model, model_type="llama"
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
    parser.add_argument("--behaviors", nargs="+", type=str, default=ALL_BEHAVIORS)

    args = parser.parse_args()
    generate_save_vectors(
        args.layers,
        args.save_activations,
        args.use_base_model,
        args.behaviors
    )
