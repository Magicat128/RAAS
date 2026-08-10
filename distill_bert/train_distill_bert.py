from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
import torch.nn as nn
import torch
import numpy as np

base_model = "distilbert-base-uncased"
data_path = "./data/cls_data.json"
with open(data_path, 'r') as f:
  df = pd.read_json(f)


label_encoder = preprocessing.LabelEncoder()
df['label'] = label_encoder.fit_transform(df['function_tags'].tolist())
label2id = {label: idx for idx, label in enumerate(label_encoder.classes_)}

model = AutoModelForSequenceClassification.from_pretrained(base_model, num_labels=8)
tokenizer = AutoTokenizer.from_pretrained(base_model)

dataset = Dataset.from_pandas(df)

def tokenize_data(examples):
    return tokenizer(examples["chunk"], truncation=True)

tokenized_dataset = dataset.map(tokenize_data, batched=True)
tokenized_dataset = tokenized_dataset.train_test_split(test_size=0.1, shuffle=True)

class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(df['label'].tolist()),
                                     y=df['label'].tolist())
class_weights = torch.tensor(class_weights, dtype=torch.float)

class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        loss_kwargs = {}
        if num_items_in_batch is not None:
            loss_kwargs["num_items_in_batch"] = num_items_in_batch
        inputs = {**inputs, **loss_kwargs}
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = nn.CrossEntropyLoss(weight=class_weights.to(model.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model('model')
