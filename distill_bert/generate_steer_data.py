from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import json
import numpy as np
import traceback
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch

model_name = "llama"
model_size = "8b"
subpath = ['correct_base_solution/', 'incorrect_base_solution/']
path = f'./math-rollouts/deepseek-r1-distill-{model_name}-{model_size}/temperature_0.6_top_p_0.95/'
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

avg_steps = 0
total_rollouts = 0
out_data = []
for sp in subpath:

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
      avg_steps += len(cls_data)

      for item in tqdm(cls_data):
        chunk, chunk_idx = item["chunk"].strip(), item["chunk_idx"]

        if chunk_idx == 0:
            function_tags = "initial_question"
        else:
            function_tags = cls_data[chunk_idx-1]["function_tags"][0]
        counterfactual_importance = item["counterfactual_importance_accuracy"]

        if abs(counterfactual_importance) == 0:
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
          
        for rollout in rollout_data:
          if "error" in rollout:
            continue
          total_rollouts += 1
          rollout_sentence = rollout["rollout"].replace('\n',' ').split('. ')[0].strip() + '.'

          if resample_better and rollout["is_correct"]:
            answer_matching = rollout_sentence
            answer_not_matching = chunk
          elif not resample_better and not rollout["is_correct"]:
            answer_matching = chunk
            answer_not_matching = rollout_sentence
          else:
            continue

          to_tags = predict(answer_not_matching)
          new_item = {"question": problem_context, "answer_matching_behavior": answer_matching, "answer_not_matching_behavior": answer_not_matching, "function_tags": function_tags, "to_tags": to_tags}
          if new_item not in out_data:
              out_data.append(new_item)

        problem_context += chunk
        if chunk[-1] != '.':
          problem_context += '\n'
        else:
          problem_context += ' '

    except Exception as e:
      print(f"Skipping {cls_file} ...")
      traceback.print_exc()


print("avg_steps: ", avg_steps)
print("total_rollout: ", total_rollouts)

with open(f'./data/{model_name}_steer_data.json', 'w') as out:
  json.dump(out_data, out, indent=4)
