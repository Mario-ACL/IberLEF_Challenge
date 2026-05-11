import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.metrics import classification_report
import numpy as np

MODEL_NAME = "./models/distilbert-spanish"
DATA_PATH = "./Data/raw"

def train_pair(train_csv, test_csv, pair_name, label_col="behavior_code_1"):
    print(f"\n{'='*50}")
    print(f"Entrenando par: {pair_name}")
    print(f"{'='*50}")

    train_df = pd.read_csv(train_csv)
    test_df  = pd.read_csv(test_csv)

    train_df["input"] = train_df["client_utterance"] + " [SEP] " + train_df["clinician_utterance"]
    test_df["input"]  = test_df["client_utterance"]  + " [SEP] " + test_df["clinician_utterance"]

    labels   = sorted(train_df[label_col].unique())
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}

    train_df["label"] = train_df[label_col].map(label2id)
    test_df["label"]  = test_df[label_col].map(label2id)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["input"], truncation=True, padding="max_length", max_length=128)

    train_ds = Dataset.from_pandas(train_df[["input", "label"]]).map(tokenize, batched=True)
    test_ds  = Dataset.from_pandas(test_df[["input",  "label"]]).map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=f"./models/{pair_name}",
        num_train_epochs=5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        fp16=torch.cuda.is_available(),
        logging_steps=20,
        report_to="none",
	learning_rate=2e-5,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
    )
    trainer.train()

    preds       = trainer.predict(test_ds)
    pred_labels = preds.predictions.argmax(axis=1)
    true_labels = test_df["label"].values

    print(f"\nResultados {pair_name}:")
    print(classification_report(true_labels, pred_labels, target_names=labels))

# --- Los 3 pares ---
# Por ahora entrenamos con los datos originales (seed dataset)
# Cuando tengas los CSVs generados por Qwen, cambia train_csv por esos

train_pair("./generated_dataset_info.csv",   f"{DATA_PATH}/INFO_AUG.csv",   "GI_PE")
train_pair("./generated_dataset_quest.csv",  f"{DATA_PATH}/QUEST_AUG.csv",  "OQ_CQ")
train_pair("./generated_dataset_reflex.csv", f"{DATA_PATH}/REFLEX_AUG.csv", "SR_CR")
