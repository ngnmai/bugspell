import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Path to your fine-tuned model
MODEL_PATH = os.environ.get("MODEL_PATH")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

def correct_text(input_text, max_length=256, num_beams=4):
    # Tokenize input
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True, padding=True).to(device)
    
    # Generate prediction using beam search for better quality
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
    
    # Decode output
    corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return corrected_text

if __name__ == "__main__":
    print("Spell Correction Inference (Code + English)")
    print("Type 'exit' to quit.\n")
    
    while True:
        text = input("Enter text/code with errors: ")
        if text.lower() == 'exit':
            break
        corrected = correct_text(text)
        print(f"Corrected Output:\n{corrected}\n")
