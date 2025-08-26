import torch
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

# Paths
MODEL_PATH = os.environ.get("MODEL_PATH")
TEST_FILE = os.environ.get("TEST_DS_PATH")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Load test data
with open(TEST_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)

# Metrics
bleu_metric = evaluate.load("bleu")
wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

predictions = []
references = []
exact_match_count = 0

print(f"Evaluating {len(data)} samples...")

for sample in tqdm(data, desc="Evaluating"):
    input_text = sample["input"]
    target_text = sample["target"]

    # Tokenize and generate prediction
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=256,
            num_beams=4,
            early_stopping=True
        )
    pred_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    predictions.append(pred_text)
    references.append(target_text)

    if pred_text.strip() == target_text.strip():
        exact_match_count += 1

# Compute metrics
bleu_score = bleu_metric.compute(predictions=[p.split() for p in predictions],
                                  references=[[r.split()] for r in references])
wer_score = wer_metric.compute(predictions=predictions, references=references)
cer_score = cer_metric.compute(predictions=predictions, references=references)
exact_match_acc = exact_match_count / len(predictions)

print("\n--- Evaluation Results ---")
print(f"BLEU Score: {bleu_score['bleu']:.4f}")
print(f"WER (Word Error Rate): {wer_score:.4f}")
print(f"CER (Character Error Rate): {cer_score:.4f}")
print(f"Exact Match Accuracy: {exact_match_acc:.4f}")
