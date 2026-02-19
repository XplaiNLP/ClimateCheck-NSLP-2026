import os
import pandas as pd
import torch
import numpy as np
from datasets import load_dataset, Dataset, DatasetDict
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModelForSequenceClassification, EvalPrediction, EarlyStoppingCallback
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from collections import Counter
from collections import defaultdict


os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = 'answerdotai/ModernBERT-large'
#MODEL_NAME = 'FacebookAI/roberta-large'
#MODEL_NAME = 'distilbert/distilbert-base-uncased'

dataset = load_dataset('alextsiak/augmented_cc26', split='train')


# count rare labels (for train/test split)

label_counter = Counter()
for labels in dataset["narrative"]:
    label_counter.update(labels)

min_rare = 2
max_rare = 10

rare_labels = {
    label for label, count in label_counter.items()
    if min_rare <= count < max_rare 
}

df = dataset.to_pandas()
all_labels = set(l for sublist in df['narrative'] for l in sublist)

assigned_train_labels = defaultdict(bool)
train_rows = []
remaining_rows = []

#assign at least one example per label to training
for idx, row in df.iterrows():
    row_labels = row['narrative']
    #if any label hasn't yet been in training, add this row to training
    if any(not assigned_train_labels[l] for l in row_labels):
        train_rows.append(idx)
        for l in row_labels:
            assigned_train_labels[l] = True
    else:
        remaining_rows.append(idx)

remaining_df = df.loc[remaining_rows]

train_df, test_df = train_test_split(remaining_df, test_size=0.05, random_state=42)
train_df = pd.concat([df.loc[train_rows], train_df], ignore_index=True)

print(f"Training examples: {len(train_df)}")
#print(f"Validation examples: {len(val_df)}")
print(f"Test examples: {len(test_df)}")

train_dataset = Dataset.from_pandas(train_df)
#val_dataset = Dataset.from_pandas(val_df)
test_dataset_original = Dataset.from_pandas(test_df)


#apply mlb

narrative_labels = train_dataset['narrative']
print(len(narrative_labels))

mlb = MultiLabelBinarizer()
mlb.fit(narrative_labels)

#fit and transform separately by row

def encode_labels(example):
    labels = mlb.transform([example['narrative']])[0]
    example['labels'] = torch.tensor(labels, dtype=torch.float32)
    return example

labelled_dataset = train_dataset.map(encode_labels)
#val_dataset = val_dataset.map(encode_labels)
test_dataset = test_dataset_original.map(encode_labels)


# tokenization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(examples):
    text = examples['claim']
    encoding = tokenizer(text, padding="max_length", truncation=True, max_length=100)
    return encoding

final_dataset = labelled_dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
#val_dataset = val_dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
test_dataset = test_dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)


y_train = np.stack(final_dataset["labels"])
label_counts = y_train.sum(axis=0)

pos_weight = (len(y_train) - label_counts) / (label_counts + 1e-6)
pos_weight = torch.tensor(pos_weight, dtype=torch.float32)

loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

final_dataset.set_format("torch")

model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, 
                                                        num_labels=len(mlb.classes_),
                                                        problem_type="multi_label_classification",
                                                        #ignore_mismatched_sizes=True #for CARDS model
                                                        )




training_args = TrainingArguments(
    output_dir="./models/climatecheck_modernbert",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=4,
    num_train_epochs=15,
    #max_steps=5,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=50,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_f1_macro",
    greater_is_better=True,
    fp16=True,
    report_to=[],
    torch_compile=False
)



def multi_label_metrics(predictions, labels, threshold=0.5):
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.Tensor(predictions))
    y_pred = np.zeros(probs.shape)
    y_pred[np.where(probs >= threshold)] = 1
    y_true = labels
    
    f1_micro_average = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro_average = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    metrics = {'f1_micro': f1_micro_average,
               'f1_macro': f1_macro_average,
               'accuracy': accuracy}
    return metrics

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    result = multi_label_metrics(
        predictions=preds, 
        labels=p.label_ids)
    return result

class CustomTrainer(Trainer):
    def __init__(self, *args, pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.pos_weight = pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        #labels = labels.to(logits.device)
        loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        loss = loss_fct(logits, labels.float())
        return (loss, outputs) if return_outputs else loss


#import torch._dynamo
#torch._dynamo.disable()

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=final_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    pos_weight=pos_weight.to(model.device),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)
    
trainer.train()

model.save_pretrained("modernbert")
tokenizer.save_pretrained("modernbert")