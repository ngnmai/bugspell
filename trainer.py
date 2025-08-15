import argparse
import os

from datasets import load_dataset
from transformers import Seq2SeqTrainer

from src.model import model_init, parse_args
    

def preprocess_function(examples, tokenizer, max_source_length, max_target_length):
    inputs = tokenizer(
        examples["input"],
        max_length=max_source_length,
        padding="max_length",
        truncation=True
    )
    labels = tokenizer(
        examples["target"],
        max_length=max_target_length,
        padding="max_length",
        truncation=True
    )
    inputs["labels"] = labels["input_ids"]
    return inputs

def main():
    args = parse_args()

    # Load model, tokenizer, data collator, and training args
    model, tokenizer, data_collator, training_args = model_init(args)

    # Load dataset
    dataset_path = os.environ.get("DATASET_PATH", None)
    if dataset_path is None:
        raise ValueError("DATASET_PATH environment variable not set. Please make sure dataload.py is run")

    dataset = load_dataset("json", data_files=dataset_path, split="train")
    # split
    dataset = dataset.train_test_split(test_size=0.02, seed=42)
    train_dataset = dataset["train"]
    val_dataset = dataset["test"]

    # tokenizing
    train_ds = train_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ), 
        batched=True, 
        remove_columns=train_dataset.column_names)
    val_ds = val_dataset.map(
        lambda examples: preprocess_function(
            examples, tokenizer, args.max_source_length, args.max_target_length
        ), 
        batched=True, 
        remove_columns=val_dataset.column_names)


    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    trainer.train()

    # Save final model
    output_dir = os.environ.get("OUTPUT_DIR", "./output/model")
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

if __name__ == "__main__":
    main()

