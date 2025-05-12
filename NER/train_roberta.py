import os
import json
import yaml
import datasets
import numpy as np
from transformers import (
    AutoTokenizer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch
from transformers import RobertaTokenizerFast

# Disable Weights & Biases logging
os.environ["WANDB_DISABLED"] = "true"

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(
                    label[word_idx] if label_all_tokens else -100
                )
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=2)

    true_labels = [
        [label_list[label] for (pred, label) in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]
    true_predictions = [
        [label_list[pred] for (pred, label) in zip(prediction, label_row) if label != -100]
        for prediction, label_row in zip(predictions, labels)
    ]

    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }

def main():
    # Load configuration
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    model_name = config.get("model_name", "roberta-base")
    num_labels = config.get("num_labels", 9)
    training_args_config = config.get("training_args", {})

    # Load dataset
    conll2003 = datasets.load_dataset("conll2003")

    # Initialize tokenizer
    global tokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    tokenizer = RobertaTokenizerFast.from_pretrained(model_name, add_prefix_space=True)
    tokenized_datasets = conll2003.map(tokenize_and_align_labels, batched=True)

    # Initialize model
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=num_labels
    )

    # Prepare training arguments
    args = TrainingArguments(**training_args_config)

    # Initialize data collator
    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Load metric
    global metric
    metric = datasets.load_metric("seqeval")

    # Get label list
    global label_list
    label_list = conll2003["train"].features["ner_tags"].feature.names

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained("ner_model")
    tokenizer.save_pretrained("tokenizer")

    # Update config.json with id2label and label2id mappings
    id2label = {str(i): label for i, label in enumerate(label_list)}
    label2id = {label: str(i) for i, label in enumerate(label_list)}

    config_path = os.path.join("ner_model", "config.json")
    with open(config_path, "r") as file:
        model_config = json.load(file)

    model_config["id2label"] = id2label
    model_config["label2id"] = label2id

    with open(config_path, "w") as file:
        json.dump(model_config, file, indent=2)

    # Evaluate the model on the test set
    print("\nEvaluating on the test set...")
    test_dataset = tokenized_datasets["test"]

    # Use the Trainer's predict method to get predictions and metrics
    test_predictions = trainer.predict(test_dataset)

    # Compute the metrics using the compute_metrics function
    test_metrics = compute_metrics(
        (test_predictions.predictions, test_predictions.label_ids)
    )

    # Print test results
    print("Test set evaluation metrics:")
    for key, value in test_metrics.items():
        print(f"{key.capitalize()}: {value:.4f}")

if __name__ == "__main__":
    main()
