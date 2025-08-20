import argparse
import os

from datasets import load_dataset
from transformers import Seq2SeqTrainer
import mlflow
import torch

from src.model import *

# Environment variables
OUTPUT_DIR = "./output"
USE_LORA = True
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
	# Compensate with the HF old version 
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Remove num_items_in_batch if present
        inputs.pop("num_items_in_batch", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR)
	parser.add_argument("--use_lora", type=bool, default=USE_LORA)
	parser.add_argument("--max_source_length", type=int, default=MAX_SOURCE_LENGTH)
	parser.add_argument("--max_target_length", type=int, default=MAX_TARGET_LENGTH)
	parser.add_argument("--num_train_epochs", type=int, default=3)
	parser.add_argument("--logging_steps", type=int, default=100)
	parser.add_argument("--eval_steps", type=int, default=500)
	parser.add_argument("--save_steps", type=int, default=500)
	return parser.parse_args()

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

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	print(f"Selected device: {device}")

	# ------------------------------------
	# Load model, tokenizer, data collator, and training args
	MODEL_NAME = "Salesforce/codet5p-2b"


	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
				 trust_remote_code=True)
	config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
	config.eos_token_id = tokenizer.eos_token_id
	config.bos_token_id = tokenizer.bos_token_id
	config.pad_token_id = tokenizer.pad_token_id
	config.decoder_start_token_id = config.bos_token_id

	model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, 
				config=config, 
				trust_remote_code=True)
	if args.use_lora:
		model = get_peft_model(model, config_lora(args))
	model.print_trainable_parameters()

	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

	training_args_ = training_args(args)

	# ------------------------------------
	model.to(device)

	# ------------------------------------

	# Load dataset
	ds_train_path = "/scratch/project_462000131/mainguye/bugspell/data/spellcheck_dataset_train_split.jsonl"
	ds_test_path = "/scratch/project_462000131/mainguye/bugspell/data/spellcheck_dataset_test_split.jsonl"

	ds_train = load_dataset("json", data_files={"train": ds_train_path})["train"]
	ds_eval = load_dataset("json", data_files={"train": ds_test_path})["train"]


	# tokenizing
	train_ds = ds_train.map(
		lambda examples: preprocess_function(
			examples, tokenizer, args.max_source_length, args.max_target_length
		), 
		batched=True
		)
	val_ds = ds_eval.map(
		lambda examples: preprocess_function(
			examples, tokenizer, args.max_source_length, args.max_target_length
		), 
		batched=True
		)

	# ------------------------------------
	print('start fine-tuning')

	trainer = CustomSeq2SeqTrainer(
		model=model,
		args=training_args_,
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

