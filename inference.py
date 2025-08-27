import json
import torch
import argparse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

def correct_text(input_text, tokenizer, model, device, max_length=512, num_beams=5):
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            #early_stopping=True
            pad_token_id=tokenizer.pad_token_id
        )
        
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_inference(model_path, input_file, output_file):
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load input data
    with open(input_file, "r", encoding="utf-8") as f:
        try:
            data_list = json.load(f)  # For JSON array
        except json.JSONDecodeError:
            # If it's JSONL format
            f.seek(0)
            data_list = [json.loads(line) for line in f]

    results = []
    for data in data_list:
        input_text = data["input"].strip()
        # if not input_text:
        #   continue
        corrected_text = correct_text(input_text, tokenizer, model, device)
        results.append({
            "input": input_text,
            "corrected": corrected_text
            })

    # Save output
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Inference complete. Results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference for spell correction on code + English text.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the fine-tuned model directory")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input JSON or JSONL file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the corrected output JSON file")
    
    args = parser.parse_args()
    run_inference(args.model_path, args.input_file, args.output_file)
