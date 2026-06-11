import torch as t
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from matplotlib import pyplot as plt
from matplotlib.ticker import ScalarFormatter
from utils.helpers import add_vector_from_position, find_instruction_end_postion
from transformers.cache_utils import DynamicCache
from utils.tokenize import (
    tokenize_llama_chat,
    tokenize_llama_base,
    ADD_FROM_POS_BASE,
    ADD_FROM_POS_CHAT,
)
from typing import Optional
from behaviors import get_steering_vector, get_anchor_vector, ID2LABEL
import numpy as np
import re


class AttnWrapper(t.nn.Module):
    """
    Wrapper for attention mechanism to save activations
    """

    def __init__(self, attn):
        super().__init__()
        self.attn = attn
        self.activations = None

    def forward(self, *args, **kwargs):
        output = self.attn(*args, **kwargs)
        self.activations = output[0]
        return output


class BlockOutputWrapper(t.nn.Module):
    """
    Wrapper for block to save activations and unembed them
    """

    def __init__(self, block, unembed_matrix, norm, tokenizer):
        super().__init__()
        self.block = block
        self.unembed_matrix = unembed_matrix
        self.norm = norm
        self.tokenizer = tokenizer

        self.block.self_attn = AttnWrapper(self.block.self_attn)
        self.post_attention_layernorm = self.block.post_attention_layernorm

        self.attn_out_unembedded = None
        self.intermediate_resid_unembedded = None
        self.mlp_out_unembedded = None
        self.block_out_unembedded = None

        self.activations = None
        self.add_activations = None
        self.pos_anchors = None
        self.neg_anchors = None
        self.from_position = None

        self.save_internal_decodings = False

        self.calc_dot_product_with = None
        self.dot_products = []

    def forward(self, *args, **kwargs):
        output = self.block(*args, **kwargs)
        self.activations = output[0]
        if self.calc_dot_product_with is not None:
            last_token_activations = self.activations[0, -1, :]
            decoded_activations = self.unembed_matrix(self.norm(last_token_activations))
            top_token_id = t.topk(decoded_activations, 1)[1][0]
            top_token = self.tokenizer.decode(top_token_id)
            dot_product = t.dot(last_token_activations, self.calc_dot_product_with) / (
                t.norm(last_token_activations) * t.norm(self.calc_dot_product_with)
            )
            self.dot_products.append((top_token, dot_product.cpu().item()))
        if self.add_activations is not None:
            augmented_output = add_vector_from_position(
                matrix=output[0],
                vector=self.add_activations,
                anchors=(self.pos_anchors, self.neg_anchors),
                position_ids=kwargs["position_ids"],
                from_pos=self.from_position,
            )
            output = (augmented_output,) + output[1:]

        if not self.save_internal_decodings:
            return output

        # Whole block unembedded
        self.block_output_unembedded = self.unembed_matrix(self.norm(output[0]))

        # Self-attention unembedded
        attn_output = self.block.self_attn.activations
        self.attn_out_unembedded = self.unembed_matrix(self.norm(attn_output))

        # Intermediate residual unembedded
        attn_output += args[0]
        self.intermediate_resid_unembedded = self.unembed_matrix(self.norm(attn_output))

        # MLP unembedded
        mlp_output = self.block.mlp(self.post_attention_layernorm(attn_output))
        self.mlp_out_unembedded = self.unembed_matrix(self.norm(mlp_output))

        return output

    def add(self, activations, pos_anchors, neg_anchors):
        self.add_activations = activations
        self.pos_anchors = pos_anchors
        self.neg_anchors = neg_anchors

    def reset(self):
        self.add_activations = None
        self.pos_anchors = None
        self.neg_anchors = None
        self.activations = None
        self.block.self_attn.activations = None
        self.from_position = None
        self.calc_dot_product_with = None
        self.dot_products = []


class LlamaWrapper:
    def __init__(
        self,
        size: str = "7b",
        use_chat: bool = True,
        model_path: Optional[str] = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        classifier_path: Optional[str] = "../distill_bert/model",
        insert_layer: Optional[int] = 23,
        override_model_weights_path: Optional[str] = None,
    ):
        self.device = "cuda" if t.cuda.is_available() else "cpu"
        self.use_chat = use_chat
        self.model_path = model_path
        self.classifier_path = classifier_path
        self.insert_layer = insert_layer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, device_map="auto", low_cpu_mem_usage=True, torch_dtype=t.float16,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

        if override_model_weights_path is not None:
            self.model.load_state_dict(t.load(override_model_weights_path))
        if size != "7b":
            self.model = self.model.half()
        self.model = self.model.to(self.device)

        self.vector_map = {}
        self.pos_anchor_map = {}
        self.neg_anchor_map = {}
        END_STR_1 = t.tensor(self.tokenizer.encode(".\n\n.\n.")[1:]).to(self.device)
        END_STR_2 = t.tensor(self.tokenizer.encode("?\n\n?\n?")[1:]).to(self.device)
        self.END_STR = t.concatenate((END_STR_1, END_STR_2)) 
        self.THINK_END_STR = t.tensor(self.tokenizer.encode("</think>")[1:]).to(self.device)

        self.id2label = ID2LABEL
        behaviors = list(self.id2label.values())
        for behavior in behaviors:
            vectors = get_steering_vector(behavior, self.insert_layer, self.model_path, normalized=False)
            vectors = [vector.to(self.device) for vector in vectors]
            self.vector_map[behavior] = vectors
            pos_anchors = get_anchor_vector(behavior, self.insert_layer, "pos", self.model_path, normalized=False)
            pos_anchors = [anchor.to(self.device) for anchor in pos_anchors]
            self.pos_anchor_map[behavior] = pos_anchors
            neg_anchors = get_anchor_vector(behavior, self.insert_layer, "neg", self.model_path, normalized=False)
            neg_anchors = [anchor.to(self.device) for anchor in neg_anchors]
            self.neg_anchor_map[behavior] = neg_anchors

        self.cls_model = AutoModelForSequenceClassification.from_pretrained(self.classifier_path)
        self.cls_tokenizer = AutoTokenizer.from_pretrained(self.classifier_path)
        self.cls_model.eval()

        for i, layer in enumerate(self.model.model.layers):
            self.model.model.layers[i] = BlockOutputWrapper(
                layer, self.model.lm_head, self.model.model.norm, self.tokenizer
            )

    def set_save_internal_decodings(self, value: bool):
        for layer in self.model.model.layers:
            layer.save_internal_decodings = value

    def set_from_positions(self, pos: int):
        for layer in self.model.model.layers:
            layer.from_position = pos

    '''
    def generate(self, tokens, max_new_tokens=100):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            generated = self.model.generate(
                inputs=tokens, max_new_tokens=max_new_tokens, top_k=1
            )
            return self.tokenizer.batch_decode(generated)[0]
    '''

    def generate_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50) -> str:
        if self.use_chat:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
            )
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.generate(tokens, max_new_tokens=max_new_tokens)

    def steer_generate(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None, max_new_tokens: int = 50, temperature: float = 0.6) -> str:
        context_list = self.tokenizer(user_input, return_tensors="pt").to(self.device)
        generated_ids = context_list.input_ids
        prev_ids = generated_ids.clone()
        past_key_values = DynamicCache()
        cache_position = t.arange(context_list.input_ids.shape[1], dtype=t.int64, device=self.device)
        is_steer_end = False
        prev_instr_pos = -1

        for _ in range(max_new_tokens):
            instr_pos = find_instruction_end_postion(generated_ids, prev_ids, self.END_STR)
            behavior_id = None

            if not is_steer_end:
                if instr_pos == -1:
                    behavior_id = "initial_question"
                elif instr_pos != prev_instr_pos: # when previous sentence is complete
                    prev_instr_pos = instr_pos
                    behavior_id = self.behavior_classification(generated_ids)

                if behavior_id:
                    self.reset_all()
                    self.set_from_positions(instr_pos)
                    self.set_add_activations(self.insert_layer, self.vector_map[behavior_id], self.pos_anchor_map[behavior_id], self.neg_anchor_map[behavior_id])

            with t.no_grad():
                generate_outputs = self.model(**context_list, cache_position=cache_position, past_key_values=past_key_values, use_cache=True)
                generate_logits = generate_outputs.logits[:,-1,:]

            if temperature > 0:
                adjusted_logits = generate_logits / temperature
                gumbel_noise = -t.log(-t.log(t.rand_like(adjusted_logits)))
                adjusted_logits = adjusted_logits + gumbel_noise

            next_token = t.argmax(adjusted_logits, dim=-1).unsqueeze(dim=0)
            next_token_ids = next_token.repeat(generated_ids.shape[0], 1)
            generated_ids = t.cat([generated_ids, next_token_ids], dim=-1)
            attention_mask = context_list["attention_mask"]
            attention_mask = t.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0],1))], dim=-1)

            context_list = {"input_ids": next_token_ids, "attention_mask": attention_mask}
            cache_position = cache_position[-1:] + 1
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            if next_token.item() == self.THINK_END_STR:
                is_steer_end = True
                self.reset_all()

        return generated_ids

    def behavior_classification(self, generated_ids):
        current_context = self.tokenizer.batch_decode(generated_ids)[0]
        last_sentences = re.split(r'\.\n\n|\.\n|\. |\?\n\n|\?\n|\? ', current_context.split('\n<think>\n')[-1])
        last_sentence = last_sentences[-1]
        if last_sentence == "":
            last_sentence = last_sentences[-2]

        last_sentence_ids = self.cls_tokenizer(last_sentence, return_tensors='pt', truncation=True, padding=True, max_length=512)
        with t.no_grad():
            outputs = self.cls_model(**last_sentence_ids)
            logits = outputs.logits
            pred = logits.argmax(dim=-1).item()
            behavior_id = self.id2label[pred]

        return behavior_id

    def get_logits(self, tokens):
        with t.no_grad():
            instr_pos = find_instruction_end_postion(tokens[0], self.END_STR)
            self.set_from_positions(instr_pos)
            logits = self.model(tokens).logits
            return logits

    def get_logits_from_text(self, user_input: str, model_output: Optional[str] = None, system_prompt: Optional[str] = None) -> t.Tensor:
        if self.use_chat:
            tokens = tokenize_llama_chat(
                tokenizer=self.tokenizer, user_input=user_input, model_output=model_output, system_prompt=system_prompt
            )
        else:
            tokens = tokenize_llama_base(tokenizer=self.tokenizer, user_input=user_input, model_output=model_output)
        tokens = t.tensor(tokens).unsqueeze(0).to(self.device)
        return self.get_logits(tokens)

    def get_last_activations(self, layer):
        return self.model.model.layers[layer].activations

    def set_add_activations(self, layer, activations, pos_anchors, neg_anchors):
        self.model.model.layers[layer].add(activations, pos_anchors, neg_anchors)

    def set_calc_dot_product_with(self, layer, vector):
        self.model.model.layers[layer].calc_dot_product_with = vector

    def get_dot_products(self, layer):
        return self.model.model.layers[layer].dot_products

    def reset_all(self):
        for layer in self.model.model.layers:
            layer.reset()

    def print_decoded_activations(self, decoded_activations, label, topk=10):
        data = self.get_activation_data(decoded_activations, topk)[0]
        print(label, data)

    def decode_all_layers(
        self,
        tokens,
        topk=10,
        print_attn_mech=True,
        print_intermediate_res=True,
        print_mlp=True,
        print_block=True,
    ):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        for i, layer in enumerate(self.model.model.layers):
            print(f"Layer {i}: Decoded intermediate outputs")
            if print_attn_mech:
                self.print_decoded_activations(
                    layer.attn_out_unembedded, "Attention mechanism", topk=topk
                )
            if print_intermediate_res:
                self.print_decoded_activations(
                    layer.intermediate_resid_unembedded,
                    "Intermediate residual stream",
                    topk=topk,
                )
            if print_mlp:
                self.print_decoded_activations(
                    layer.mlp_out_unembedded, "MLP output", topk=topk
                )
            if print_block:
                self.print_decoded_activations(
                    layer.block_output_unembedded, "Block output", topk=topk
                )

    def plot_decoded_activations_for_layer(self, layer_number, tokens, topk=10):
        tokens = tokens.to(self.device)
        self.get_logits(tokens)
        layer = self.model.model.layers[layer_number]

        data = {}
        data["Attention mechanism"] = self.get_activation_data(
            layer.attn_out_unembedded, topk
        )[1]
        data["Intermediate residual stream"] = self.get_activation_data(
            layer.intermediate_resid_unembedded, topk
        )[1]
        data["MLP output"] = self.get_activation_data(layer.mlp_out_unembedded, topk)[1]
        data["Block output"] = self.get_activation_data(
            layer.block_output_unembedded, topk
        )[1]

        # Plotting
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(8, 6))
        fig.suptitle(f"Layer {layer_number}: Decoded Intermediate Outputs", fontsize=21)

        for ax, (mechanism, values) in zip(axes.flatten(), data.items()):
            tokens, scores = zip(*values)
            ax.barh(tokens, scores, color="skyblue")
            ax.set_title(mechanism)
            ax.set_xlabel("Value")
            ax.set_ylabel("Token")

            # Set scientific notation for x-axis labels when numbers are small
            ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="x")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

    def get_activation_data(self, decoded_activations, topk=10):
        softmaxed = t.nn.functional.softmax(decoded_activations[0][-1], dim=-1)
        values, indices = t.topk(softmaxed, topk)
        probs_percent = [int(v * 100) for v in values.tolist()]
        tokens = self.tokenizer.batch_decode(indices.unsqueeze(-1))
        return list(zip(tokens, probs_percent)), list(zip(tokens, values.tolist()))


if __name__ == "__main__":
    prompt = "Cars have lined up on the motorway. Some of the cars drive through in the first 15 minutes of the traffic jam, then 20 more cars drive through in the remaining 15 minutes of the jam. 5 cars from the line take an exit so they don't have to drive through the traffic jam. If there were originally 30 cars on the motorway, how many cars drove through the traffic jam in the first 15 minutes?"
    x = "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. Problem: " + prompt + "\n Solution:\n<think>\n"

    steer = True
    model = LlamaWrapper(model_path="/data/zhangyufeng/LLM_models/DeepSeek-R1-Distill-Qwen-7B")
    #"deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
    model.set_save_internal_decodings(False)
    input_ids = model.tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generate_input = {
        "input_ids":input_ids,
        "max_new_tokens":8192,
        "temperature": 0.6,
        "top_p": 0.95,
        "do_sample":True,
        "top_k":40,
    }

    if steer:
        generate_ids = model.steer_generate(prompt, max_new_tokens=8192)
    else:
        generate_ids = model.model.generate(**generate_input)
    generate_ids = [item[len(input_ids[0]):-1] for item in generate_ids]
    response = model.tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 
    print(response)
