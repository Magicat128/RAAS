import torch as t
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np


def set_plotting_settings():
    plt.style.use('seaborn-v0_8')
    params = {
        "ytick.color": "black",
        "xtick.color": "black",
        "axes.labelcolor": "black",
        "axes.edgecolor": "black",
        "font.family": "serif",
        "font.size": 13,
        "figure.autolayout": True,
        'figure.dpi': 600,
    }
    plt.rcParams.update(params)

    custom_colors = ['#377eb8', '#ff7f00', '#4daf4a',
                     '#f781bf', '#a65628', '#984ea3',
                     '#999999', '#e41a1c', '#dede00']
    plt.rcParams['axes.prop_cycle'] = plt.cycler(color=custom_colors)


def add_vector_from_position(matrix, vector, anchors, position_ids, from_pos=None):
    pos_anchors, neg_anchors = anchors
    
    pos_scores = np.array([(F.cosine_similarity(matrix[0][-1], anc, dim=0).item()) if anc.dim() > 0 else 0 for anc in pos_anchors])
    neg_scores = np.array([(F.cosine_similarity(matrix[0][-1], anc, dim=0).item()) if anc.dim() > 0 else 0 for anc in neg_anchors])
    mask = np.array([1 if neg > pos else 0 for pos, neg in zip(pos_scores, neg_scores)])

    scores = mask * neg_scores
    vector = [pos - neg for pos, neg in zip(pos_anchors, neg_anchors)]

    final_vector = sum([vec * score for vec, score in zip(vector, scores)])
    matrix += final_vector

    return matrix


def find_last_subtensor_position(tensor, initial_tensor, sub_tensor):
    n, m, p = tensor.size(-1), sub_tensor.size(0), initial_tensor.size(-1)

    for i in range(n - 1, p, -1):
        for j in range(m):
            if t.equal(tensor[0, i], sub_tensor[j]):
                return i
    return -1


def find_instruction_end_postion(tokens, initial_tokens, end_str):
    start_pos = find_last_subtensor_position(tokens, initial_tokens, end_str)
    if start_pos == -1:
        return -1
    return start_pos + len(end_str) - 1


def get_a_b_probs(logits, a_token_id, b_token_id):
    last_token_logits = logits[0, -1, :]
    last_token_probs = t.softmax(last_token_logits, dim=-1)
    a_prob = last_token_probs[a_token_id].item()
    b_prob = last_token_probs[b_token_id].item()
    return a_prob, b_prob


def make_tensor_save_suffix(layer, model_name_path):
    return f'{layer}_{model_name_path.split("/")[-1]}'
