import os
import json
from datasets import load_dataset, interleave_datasets
from augment_data import process_python, process_text, process_markdown

# Load datasets
python_ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train", streaming=True)
markdown_ds = load_dataset("bigcode/the-stack", data_dir="data/markdown", split="train", streaming=True)
wiki_ds = load_dataset("wikitext", "wikitext-103-v1", split="train", streaming=True)

# Process datasets
python_pairs = python_ds.map(process_python).select_columns(["input", "target"])
markdown_pairs = markdown_ds.map(process_markdown).select_columns(["input", "target"])
wiki_pairs = wiki_ds.map(process_text).select_columns(["input", "target"])

# Blend datasets
mixed = interleave_datasets(
    [python_pairs, markdown_pairs, wiki_pairs],
    probabilities=[0.35, 0.15, 0.5],
    seed=42
)

# Save full blended dataset
full_dataset_path = "./data/spellcheck_dataset_big.jsonl"
with open(full_dataset_path, "w", encoding="utf-8") as f:
    for ex in mixed.take(275000):  # Adjust limit if needed
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print("✅ Saved full blended dataset to spellcheck_dataset_small.jsonl")

# Read all lines
with open(full_dataset_path, "r", encoding="utf-8") as f:
    lines = f.readlines()

# Calculate split indices
total = len(lines)
train_end = int(0.7 * total)
val_end = int(0.85 * total)

# Split data
train_lines = lines[:train_end]
val_lines = lines[train_end:val_end]
test_lines = lines[val_end:]

# Save splits
with open("./data/spellcheck_ds_train_big.jsonl", "w", encoding="utf-8") as f:
    f.writelines(train_lines)

with open("./data/spellcheck_ds_val_big.jsonl", "w", encoding="utf-8") as f:
    f.writelines(val_lines)

with open("./data/spellcheck_ds_test_big.jsonl", "w", encoding="utf-8") as f:
    f.writelines(test_lines)

print(f"✅ Split completed: {len(train_lines)} train, {len(val_lines)} validation, {len(test_lines)} test entries.")
