import os
import json
from datasets import load_dataset, interleave_datasets
from augment_data import process_python, process_text, process_markdown


python_ds = load_dataset("bigcode/the-stack", data_dir="data/python",split="train", streaming=True)
markdown_ds = load_dataset("bigcode/the-stack", data_dir="data/markdown", split="train", streaming=True)
wiki_ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)


python_pairs = python_ds.map(process_python).select_columns(["input", "target"])
markdown_pairs = markdown_ds.map(process_markdown).select_columns(["input", "target"])
wiki_pairs = wiki_ds.map(process_text).select_columns(["input", "target"])

# check only 1 instance 

# blending the datasets 
mixed = interleave_datasets(
    [python_pairs, markdown_pairs, wiki_pairs],
    probabilities=[0.35, 0.15, 0.5],
    seed=42
)

with open("./data/spellcheck_dataset_train.jsonl", "w", encoding="utf-8") as f:
    for ex in mixed.take(1000000):  # limit if needed
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

os.environ["DATASET_PATH"] = "./data/spellcheck_dataset.jsonl"

print("✅ Saved blended dataset to spellcheck_dataset.jsonl")


# Paths
input_file = "./data/spellcheck_dataset_train.jsonl"
train_file = "./data/spellcheck_dataset_train_split.jsonl"
test_file = "./data/spellcheck_dataset_test_split.jsonl"

# Read all lines
with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Split 90% train, 10% test
split_index = int(0.8 * len(lines))
train_lines = lines[:split_index]
test_lines = lines[split_index:]

# Write train
with open(train_file, "w", encoding="utf-8") as f_train:
    f_train.writelines(train_lines)

# Write test
with open(test_file, "w", encoding="utf-8") as f_test:
    f_test.writelines(test_lines)

print(f"Split completed: {len(train_lines)} training entries and {len(test_lines)} testing entries.")

