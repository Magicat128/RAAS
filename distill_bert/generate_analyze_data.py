from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import numpy as np
import traceback
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch

model_name = "llama-8b"  # qwen-14b
subpath = ['correct_base_solution/', 'incorrect_base_solution/']
path = f'./math-rollouts/deepseek-r1-distill-{model_name}/temperature_0.6_top_p_0.95/'
instruct = "Solve this math problem step by step. You MUST put your final answer in \\boxed{}. Problem: "

base_model = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.eval()

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()
    return predicted_class_id

sim_correct, sim_not_correct, not_sim_correct, not_sim_not_correct = [], [], 0, 0
num_correct, num_not_correct = 0, 0
total, opc = {"early":0,"middle":0,"late":0}, {"early":0,"middle":0,"late":0}
id2label = {-1:"initial_question", 4:"problem_setup", 3:"plan_generation", 1:"fact_retrieval", 0:"active_computation", 5:"result_consolidation", 7:"uncertainty_management", 6:"self_checking", 2:"final_answer_emission"}
correct_trans, not_correct_trans = {}, {}
for i in id2label.values():
  correct_trans[i], not_correct_trans[i] = {}, {}
  for j in id2label.values():
    correct_trans[i][j] = 0 
    not_correct_trans[i][j] = 0

out_data = []
for sp in subpath:
  if 'incorrect' in sp:
    is_current_correct = False
  else:
    is_current_correct = True
  top_dir = Path(path + sp)
  folders = [f.name for f in top_dir.iterdir() if f.is_dir()]
  for folder in folders:
    cls_file = path + sp + folder + "/chunks_labeled.json"
    problem_file = path + sp + folder + "/problem.json"
    try:
      with open(problem_file, 'r') as pf:
        problem_data = json.load(pf)
        problem = problem_data["problem"]
        problem_context = instruct + problem + " Solution: \n<think>\n"
      with open(cls_file, 'r') as cf:
        cls_data = json.load(cf)
      for item in tqdm(cls_data):
        chunk, chunk_idx = item["chunk"].strip(), item["chunk_idx"]
        function_tags = item["function_tags"][0]
        if chunk_idx == 0:
            prev_function_tags = "initial_question"
        else:
            prev_function_tags = cls_data[chunk_idx-1]["function_tags"][0]

        counterfactual_importance = item["counterfactual_importance_accuracy"]

        if prev_function_tags == "unknown" or function_tags == "unknown" or abs(counterfactual_importance) == 0:
          problem_context += chunk
          if chunk[-1] != '.':
            problem_context += '\n'
          else:
            problem_context += ' '
          continue

        if counterfactual_importance > 0:
          resample_better = True
        else:
          resample_better = False
        
        rollout_file = path + sp + folder + f"/chunk_{chunk_idx}/solutions.json"
        with open(rollout_file, 'r') as rf:
          rollout_data = json.load(rf)
          
        total_num = 0
        for rollout in rollout_data:
          if "error" in rollout:
            continue
          if resample_better == rollout["is_correct"]:
            total_num += 1

        for rollout in rollout_data:
          if "error" in rollout:
              continue
          rollout_sentence = rollout["rollout"].replace('\n',' ').split('. ')[0].strip() + '.'

          to_tags = id2label[predict(rollout_sentence)]
          if resample_better and rollout["is_correct"]:
              correct_trans[prev_function_tags][to_tags] += 1 / total_num
              not_correct_trans[prev_function_tags][function_tags] += 1 / total_num
          elif not resample_better and not rollout["is_correct"]:
              correct_trans[prev_function_tags][function_tags] += 1 / total_num
              not_correct_trans[prev_function_tags][to_tags] += 1 / total_num
          else:
              continue


        problem_context += chunk
        if chunk[-1] != '.':
          problem_context += '\n'
        else:
          problem_context += ' '

      #print(sim_correct / (num_correct + 1e-9), sim_not_correct / (num_not_correct + 1e-9))
      print("correct_trans: ", correct_trans)
      print("not_correct_trans: ", not_correct_trans)
      #print(opc, total)

    except Exception as e:
      print(f"Skipping {cls_file} ...")
      traceback.print_exc()
      exit()


with open(f'./data/{model_name}_trans_data.json', 'w') as out:
  json.dump({"sim_correct": correct_trans, "sim_not_correct": not_correct_trans}, out, indent=4)
