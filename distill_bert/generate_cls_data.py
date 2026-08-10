from pathlib import Path
import json

model_name = ['llama-8b', 'qwen-14b']
subpath = ['correct_base_solution/', 'incorrect_base_solution/']

out_data = []
for mn in model_name:
  path = f'./math-rollouts/deepseek-r1-distill-{mn}/temperature_0.6_top_p_0.95/'
  for sp in subpath:
    top_dir = Path(path + sp)
    folders = [f.name for f in top_dir.iterdir() if f.is_dir()]
    for folder in folders:
      cls_file = path + sp + folder + "/chunks_labeled.json"
      try:
        with open(cls_file, 'r') as cf:
          cls_data = json.load(cf)
        for item in cls_data:
          if item["function_tags"][0] == "unknown":
            continue
          new_item = {"chunk": item["chunk"], "function_tags": item["function_tags"][0]}
          out_data.append(new_item)
      except Exception as e:
        print(f"Skipping {cls_file} ...")

with open('./data/cls_data.json', 'w') as out:
  json.dump(out_data, out, indent=4)
