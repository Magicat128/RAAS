import os
from typing import Literal, Optional
from utils.helpers import make_tensor_save_suffix
import json
import torch as t

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PROBLEM_SETUP = "problem_setup"
PLAN_GENERATION = "plan_generation"
FACT_RETRIEVAL = "fact_retrieval"
ACTIVE_COMPUTATION = "active_computation"
UNCERTAINTY_MANAGEMENT = "uncertainty_management"
RESULT_CONSOLIDATION = "result_consolidation"
SELF_CHECKING = "self_checking"
FINAL_ANSWER_EMISSION = "final_answer_emission"

HUMAN_NAMES = {
    PROBLEM_SETUP: "Problem Setup",
    PLAN_GENERATION: "Plan Generation",
    FACT_RETRIEVAL: "Fact Retrieval",
    ACTIVE_COMPUTATION: "Active Computation",
    UNCERTAINTY_MANAGEMENT: "Uncertainty Management",
    RESULT_CONSOLIDATION: "Result Consolidation",
    SELF_CHECKING: "Self Checking",
    FINAL_ANSWER_EMISSION: "Final Answer Emission",
}

ALL_BEHAVIORS = [
    PROBLEM_SETUP,
    PLAN_GENERATION,
    FACT_RETRIEVAL,
    ACTIVE_COMPUTATION,
    UNCERTAINTY_MANAGEMENT,
    RESULT_CONSOLIDATION,
    SELF_CHECKING,
    FINAL_ANSWER_EMISSION,
]

ID2LABEL = {0:"active_computation", 1:"fact_retrieval", 2:"final_answer_emission", 3:"plan_generation", 4:"problem_setup", 5:"result_consolidation", 6:"self_checking", 7:"uncertainty_management", -1:"initial_question"}

VECTORS_PATH = os.path.join(BASE_DIR, "vectors")
ANCHORS_PATH = "_anchor_vectors"
NORMALIZED_VECTORS_PATH = os.path.join(BASE_DIR, "normalized_vectors")
ANALYSIS_PATH = os.path.join(BASE_DIR, "analysis")
RESULTS_PATH = os.path.join(BASE_DIR, "results")
GENERATE_DATA_PATH = os.path.join(BASE_DIR, "datasets", "generate")
TEST_DATA_PATH = os.path.join(BASE_DIR, "distill_bert", "data")
RAW_DATA_PATH = os.path.join(BASE_DIR, "datasets", "raw")
ACTIVATIONS_PATH = os.path.join(BASE_DIR, "activations")
FINETUNE_PATH = os.path.join(BASE_DIR, "finetuned_models")


def get_vector_dir(behavior: str, normalized=False) -> str:
    return os.path.join(NORMALIZED_VECTORS_PATH if normalized else VECTORS_PATH, behavior)


def get_anchor_dir(behavior: str, prefix, normalized=False) -> str:
    return os.path.join(NORMALIZED_VECTORS_PATH if normalized else os.path.join(BASE_DIR, prefix + ANCHORS_PATH), behavior)


def get_vector_path(behavior: str, layer, model_name_path: str, normalized=False) -> str:
    return os.path.join(
        get_vector_dir(behavior, normalized=normalized),
        f"vec_layer_{make_tensor_save_suffix(layer, model_name_path)}.pt",
    )


def get_anchor_path(behavior: str, layer, prefix, model_name_path: str, normalized=False) -> str:
    return os.path.join(
        get_anchor_dir(behavior, prefix, normalized=normalized),
        f"vec_layer_{make_tensor_save_suffix(layer, model_name_path, pos_neg)}.pt",
    )


def get_raw_data_path(behavior: str) -> str:
    return os.path.join(RAW_DATA_PATH, behavior, "dataset.json")


def get_ab_data_path(behavior: str, test: bool = False) -> str:
    if test:
        path = os.path.join(TEST_DATA_PATH, behavior, "test_dataset_ab.json")
    else:
        path = os.path.join(GENERATE_DATA_PATH, behavior, "generate_dataset.json")
    return path


def get_open_ended_data_path(behavior: str) -> str:
    return os.path.join(TEST_DATA_PATH, "steer_data.json")


def get_truthful_qa_path() -> str:
    return os.path.join(TEST_DATA_PATH, "truthfulqa", "truthful_qa.json")


def get_mmlu_path() -> str:
    return os.path.join(TEST_DATA_PATH, "mmlu", "mmlu.json")


def get_analysis_dir(behavior: str) -> str:
    return os.path.join(ANALYSIS_PATH, behavior)


def get_results_dir(behavior: str) -> str:
    return os.path.join(RESULTS_PATH, behavior)


def get_activations_dir(behavior: str) -> str:
    return os.path.join(ACTIVATIONS_PATH, behavior)


def get_activations_path(
    behavior: str, layer, model_name_path: str, pos_or_neg: Literal["pos", "neg"]
) -> str:
    return os.path.join(
        get_activations_dir(behavior),
        f"activations_{pos_or_neg}_{make_tensor_save_suffix(layer, model_name_path)}.pt",
    )


def get_system_prompt(
    behavior: str, pos_or_neg: Optional[Literal["pos", "neg"]]
) -> Optional[str]:
    if pos_or_neg is None:
        return None
    return _SYSTEM_PROMPTS[behavior][pos_or_neg]


def get_ab_test_data(behavior):
    with open(get_ab_data_path(behavior, test=True), "r") as f:
        data = json.load(f)
    return data


def get_open_ended_test_data(behavior):
    with open(get_open_ended_data_path(behavior), "r") as f:
        data = json.load(f)
    return data

def get_open_ended_unique_test_data(behavior):
    with open(get_open_ended_data_path(behavior), "r") as f:
        data = json.load(f)
    unique_data = []
    unique_question = set()
    for item in data:
        question = item["question"].split("\n<think>\n")[0]
        if question not in unique_question:
            unique_data.append(item)
            unique_question.add(question)
    return unique_data

def get_open_ended_unique_behavior_test_data(behavior): 
    with open(get_open_ended_data_path(behavior), "r") as f:
        data = json.load(f)
    unique_data = []
    unique_question = set()
    for item in data:
        question = item["question"]
        if question not in unique_question and item["function_tags"] == behavior:
            unique_data.append(item)
            unique_question.add(question)
    return unique_data

def get_truthful_qa_data():
    with open(get_truthful_qa_path(), "r") as f:
        data = json.load(f)
    return data


def get_mmlu_data():
    with open(get_mmlu_path(), "r") as f:
        data = json.load(f)
    return data


def get_steering_vector(behavior, layer, model_name_path, normalized=False):
    return t.load(get_vector_path(behavior, layer, model_name_path, normalized=normalized))


def get_anchor_vector(behavior, layer, prefix, model_name_path, normalized=False):
    return t.load(get_anchor_path(behavior, layer, prefix, model_name_path, normalized=normalized))


def get_finetuned_model_path(
    behavior: str, pos_or_neg: Optional[Literal["pos", "neg"]], layer=None
) -> str:
    if layer is None:
        layer = "all"
    return os.path.join(
        FINETUNE_PATH,
        f"{behavior}_{pos_or_neg}_finetune_{layer}.pt",
    )


def get_finetuned_model_results_path(
    behavior: str, pos_or_neg: Optional[Literal["pos", "neg"]], eval_type: str, layer=None
) -> str:
    if layer is None:
        layer = "all"
    return os.path.join(
        RESULTS_PATH,
        f"{behavior}_{pos_or_neg}_finetune_{layer}_{eval_type}_results.json",
    )
