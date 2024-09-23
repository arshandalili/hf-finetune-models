from transformers import (
    RobertaForSequenceClassification,
    RobertaTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)
from datasets import load_dataset, load_metric
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="FacebookAI/roberta-base")
parser.add_argument("--dataset", type=str)
parser.add_argument("--text_col", type=str, default="text")
parser.add_argument("--label_col", type=str, default="label")
parser.add_argument("--task", type=str, default="yelp")
parser.add_argument("--train_split", type=str, default="train")
parser.add_argument("--eval_split", type=str, default="test")
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--epochs", type=int, default=1)

args = parser.parse_args()

print(
    f"Fine-tuning {args.model} on {args.dataset} for {args.task} classification task."
)


model = RobertaForSequenceClassification.from_pretrained(args.model)
tokenizer = RobertaTokenizer.from_pretrained(args.model)

print(f"Name/path to the model: {model.name_or_path}")

train_ds, eval_ds = load_dataset(
    args.dataset, split=[args.train_split, args.eval_split]
)


def tokenize_function(examples):
    return tokenizer(examples[args.text_col], truncation=True)


tokenized_train_ds = train_ds.map(tokenize_function, batched=True)
tokenized_eval_ds = eval_ds.map(tokenize_function, batched=True)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

output_dir = (
    f"/path/to/fts/{args.task}"  # Change this path to the desired output directory
)

training_args = TrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=args.lr,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    num_train_epochs=args.epochs,
    weight_decay=0.01,
    save_total_limit=1,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
)


accuracy_metric = load_metric("accuracy", trust_remote_code=True)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return accuracy_metric.compute(predictions=predictions, references=labels)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train_ds,
    eval_dataset=tokenized_eval_ds,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

trainer.save_model(f"{output_dir}/best_model")
tokenizer.save_pretrained(f"{output_dir}/best_model")

print(f"Model fine-tuning complete and the best model saved to {output_dir}.")
