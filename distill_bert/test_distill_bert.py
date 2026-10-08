from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.metrics import classification_report

base_model = "distilbert-base-uncased"
data_path = "./data/test_cls_data.json"
with open(data_path, 'r') as f:
  df = pd.read_json(f)

texts = df['chunk'].tolist()
label_encoder = preprocessing.LabelEncoder()

pred_labels = []
true_labels = label_encoder.fit_transform(df['function_tags'].tolist())

model = AutoModelForSequenceClassification.from_pretrained("./model")
tokenizer = AutoTokenizer.from_pretrained(base_model)
model.eval()

for text in texts:
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        pred = logits.argmax(dim=-1).item()
        pred_labels.append(pred)

target_names = label_encoder.classes_
print(classification_report(true_labels, pred_labels, target_names=target_names))

def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class_id = logits.argmax(dim=-1).item()
    return predicted_class_id

#id2label = {0:"active_computation", 1:"fact_retrieval", 2:"final_answer_emission", 3:"plan_generation", 4:"problem_setup", 5:"result_consolidation", 6:"self_checking", 7:"uncertainty_management"}
#print("预测标签:", id2label[predict(text)])
