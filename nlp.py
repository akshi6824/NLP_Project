# nlp_fixed.py
# ======================================================================================
# Privacy Policy Risk Detector (FINAL TRAINER CONFIGURATION) - Fixed version
# ======================================================================================

import os
import json
import glob
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

# Disable W&B logging
os.environ["WANDB_DISABLED"] = "true"

print("\n--- Setting up environment ---")

# ======================================================================================
# Load and Prepare Data
# ======================================================================================

dataset_root = './OPP-115/'
annotations_path = os.path.join(dataset_root, 'annotations/')

all_csv_files = glob.glob(os.path.join(annotations_path, "*.csv"))
if not all_csv_files:
    raise FileNotFoundError(f"No CSVs found in {annotations_path}. Please check dataset path.")

print(f"Found {len(all_csv_files)} CSV files. Combining...")

# read csvs (they appear to have no header in OPP-115)
df_list = [pd.read_csv(file, header=None) for file in all_csv_files]
master_df = pd.concat(df_list, ignore_index=True)

# assign flexible column names depending on number of columns
num_columns = master_df.shape[1]
possible_cols = [
    'col_A', 'batch_id', 'annotator_id', 'doc_id', 'segment_id',
    'category', 'json_data', 'date', 'policy_url', 'policy_url_2'
]
master_df.columns = possible_cols[:num_columns]

def extract_nested_text(data_string):
    """Extract the selectedText field from the nested JSON in column 'json_data'."""
    try:
        json_obj = json.loads(data_string)
        # json_obj keys are annotation ids; each value is a dict with 'selectedText'
        for key in json_obj:
            nested = json_obj[key]
            if isinstance(nested, dict) and 'selectedText' in nested:
                return nested['selectedText']
        return ''
    except Exception:
        return ''

tqdm.pandas(desc="Extracting text")
master_df['text_segment'] = master_df['json_data'].progress_apply(extract_nested_text)

# filter empty texts and exclude 'Other' category
df_filtered = master_df[
    (master_df['text_segment'].str.strip() != '') &
    (master_df['category'] != 'Other')
].copy()

# group labels per text segment (OPP-115 style)
grouped_labels = df_filtered.groupby('text_segment')['category'].apply(list).reset_index()

# multi-label binarizer
mlb = MultiLabelBinarizer()
one_hot = pd.DataFrame(mlb.fit_transform(grouped_labels['category']),
                       columns=mlb.classes_, index=grouped_labels.index)

label_columns = list(mlb.classes_)
final_df = pd.concat([grouped_labels['text_segment'], one_hot], axis=1)

print(f"\n--- Data ready with {final_df.shape[0]} rows and {len(label_columns)} labels ---")

# ======================================================================================
# Optional Hard-Negative Augmentation
# ======================================================================================

hard_negatives = [
    "we do not share your data with third parties.",
    "we will not sell your information to advertisers.",
    "your data is never sold to any third party.",
    "we do not disclose personal information for marketing purposes.",
    "information is not shared without your consent."
]

if 'Third Party Sharing/Collection' in label_columns:
    print("Found 'Third Party Sharing/Collection' label. Adding hard negatives...")
    safe_labels = [0.0] * len(label_columns)
    new_rows = []
    for sentence in hard_negatives:
        for _ in range(20):
            row = {'text_segment': sentence}
            for i, col_name in enumerate(label_columns):
                row[col_name] = safe_labels[i]
            new_rows.append(row)
    augmentation_df = pd.DataFrame(new_rows)
    final_df_augmented = pd.concat([final_df, augmentation_df], ignore_index=True)
    print(f"Data augmented: {len(final_df)} -> {len(final_df_augmented)}")
else:
    final_df_augmented = final_df

# ======================================================================================
# Prepare train/val splits, datasets and tokenizer
# ======================================================================================

train_texts, val_texts, train_labels, val_labels = train_test_split(
    final_df_augmented['text_segment'].tolist(),
    final_df_augmented[label_columns].values.astype(float),
    test_size=0.2,
    random_state=42
)

# create HuggingFace datasets (these are datasets.Dataset objects)
train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels.tolist()})
val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels.tolist()})

model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# prepare model for multi-label classification
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    problem_type="multi_label_classification",
    num_labels=len(label_columns)
)
# keep mapping for convenience
model.config.id2label = {i: label for i, label in enumerate(label_columns)}
model.config.label2id = {label: i for i, label in enumerate(label_columns)}

# tokenization function - must also keep 'labels'
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    tokens['labels'] = examples['labels']
    return tokens

# map tokenization (batched)
train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=['text'])
val_dataset = val_dataset.map(tokenize_function, batched=True, remove_columns=['text'])

# ======================================================================================
# Custom Trainer with weighted BCEWithLogitsLoss (uses pos_weight for per-class weighting)
# ======================================================================================

# choose labels that deserve heavy weighting (aggressive)
HIGH_RISK_LABELS = ['Third Party Sharing/Collection', 'International Data Transfer']
# default weight 1.0 for all labels
class_weights = [1.0] * len(label_columns)

print(f"Applying aggressive weights (10.0) to critical labels: {HIGH_RISK_LABELS}")
for critical in HIGH_RISK_LABELS:
    if critical in label_columns:
        idx = label_columns.index(critical)
        class_weights[idx] = 10.0
        print(f"  -> applied weight 10.0 to '{critical}'")
    else:
        print(f"  -> warning: '{critical}' not in label set, skipping.")

# pos_weight expects shape [num_labels]; convert to tensor (float) and keep on CPU for init
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)

class CustomTrainer(Trainer):
    def __init__(self, *args, label_pos_weight=None, **kwargs):
        super().__init__(*args, **kwargs)
        # label_pos_weight should be a 1D tensor shaped [num_labels]
        self.label_pos_weight = label_pos_weight

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        # Accept arbitrary kwargs to avoid unexpected-kwarg errors from Trainer internals
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # BCEWithLogitsLoss has 'pos_weight' to up-weight positive examples for each class
        if self.label_pos_weight is not None:
            pos_weight = self.label_pos_weight.to(logits.device)
            loss_fct = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            loss_fct = torch.nn.BCEWithLogitsLoss()

        # labels are expected as float for BCE
        loss = loss_fct(logits, labels.float().to(logits.device))

        return (loss, outputs) if return_outputs else loss

# ======================================================================================
# Training arguments, metrics, and trainer initialization
# ======================================================================================

training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,  # set small if you are on CPU / low memory
    per_device_eval_batch_size=16,
    warmup_steps=200,
    weight_decay=0.01,
    logging_strategy="epoch",
    eval_strategy="epoch",
    save_strategy="epoch",
    report_to="none",
    fp16=False,  # set True only if supported
)

def compute_metrics(p):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    sigmoid = torch.nn.Sigmoid()
    probs = sigmoid(torch.tensor(preds))
    y_pred = (probs >= 0.5).int().numpy()
    y_true = p.label_ids
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='micro')
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    return {'f1_micro': f1, 'subset_accuracy': acc}

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    label_pos_weight=class_weights_tensor  # pass pos_weight here
)

print("\n--- Training model (with weighted multi-label loss) ---")
trainer.train()

# ======================================================================================
# Save Model + Tokenizer
# ======================================================================================
model.save_pretrained("./privacy_model")
tokenizer.save_pretrained("./privacy_model")
print("\n✅ Model and tokenizer saved to ./privacy_model/")
