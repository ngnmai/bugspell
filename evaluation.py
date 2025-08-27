import torch
import json
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import evaluate

def generate_prediction(text, tokenizer, model, device, max_length=512, num_beams=4):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            #early_stopping=True
            pad_token_id=tokenizer.pad_token_id
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model(model_path, test_file):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load test data
    with open(test_file, "r", encoding="utf-8") as f:
        try:
            data_list = json.load(f)  # JSON array
        except json.JSONDecodeError:
            # If JSONL format
            f.seek(0)
            data_list = [json.loads(line) for line in f]

    bleu_metric = evaluate.load("sacrebleu")
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")

    predictions = []
    references = []
    exact_match_count = 0

    print(f"Evaluating {len(data_list)} samples...")

    for data in data_list:
        input_text = data["input"].strip()
        target_text = data["target"].strip()

        print(input_text)
        print(target_text)

        pred_text = generate_prediction(input_text, tokenizer, model, device)

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned model on test dataset.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test JSON or JSONL file")
    
    args = parser.parse_args()
    evaluate_model(args.model_path, args.test_file)
