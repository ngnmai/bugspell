import os
import json
from datasets import load_dataset, interleave_datasets
from utils.augment_data import process_python, process_text, process_markdown


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

with open("./data/spellcheck_dataset.jsonl", "w", encoding="utf-8") as f:
    for ex in mixed.take(10):  # limit if needed
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

os.environ["DATASET_PATH"] = "./data/spellcheck_dataset.jsonl"

print("✅ Saved blended dataset to spellcheck_dataset.jsonl")
