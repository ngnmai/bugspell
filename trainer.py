import argparse
import os

from datasets import load_dataset
import evaluate
from transformers import Seq2SeqTrainer, TrainerCallback
import mlflow
import torch
import numpy as np

from src.model import *

# Environment variables
OUTPUT_DIR = "./output"
USE_LORA = True
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

# metrics
wer_metric = evaluate.load("wer") # -> word error rate: counts insertions, deletions, substitution at word level
cer_metric = evaluate.load("cer") # -> character error rate: measure how many characters differ between pred and target
bleu_metric = evaluate.load("sacrebleu") #-> translation style metric comparing n-grams of pred vs reference


class CustomSeq2SeqTrainer(Seq2SeqTrainer):
	# Compensate with the HF old version 
	def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
		# Remove num_items_in_batch if present
		inputs.pop("num_items_in_batch", None)
		outputs = model(**inputs)
		loss = outputs.loss
		return (loss, outputs) if return_outputs else loss


class MLflowLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(key, value, step=state.global_step)


def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("--train_ds_dir", type=str)
	parser.add_argument("--eval_ds_dir", type=str)
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
	# mlflow setup
	mlflow.set_tracking_uri("/scratch/project_462000131/mainguye/mlruns")  # or your remote URI
	mlflow.set_experiment("CodeT5p_Fine-Tuning")


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

	# metrics ----------------------------
	def compute_metrics(eval_pred):
		predictions, labels = eval_pred
		decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
		decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

		decoded_preds = [pred.strip() for pred in decoded_preds]
		decoded_labels = [label.strip() for label in decoded_labels]

		bleu_result = bleu_metric.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
		cer_result = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
		wer_result = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

		metrics = {
			"bleu": bleu_result["score"] if "score" in bleu_result else bleu_result["bleu"],
			"cer": cer_result,
			"wer": wer_result
		}

		# Log to MLflow
		mlflow.log_metrics(metrics)

		return metrics

	# ------------------------------------

	# Load dataset
	ds_train_path = args.train_ds_dir
	ds_test_path = args.eval_ds_dir

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
	print('Start fine-tuning')
	print('Test mlflow track with quick benchmark')
	with mlflow.start_run():
		trainer = CustomSeq2SeqTrainer(
			model=model,
			args=training_args_,
			train_dataset=train_ds,
			eval_dataset=val_ds,
			data_collator=data_collator,
			tokenizer=tokenizer,
			compute_metrics=compute_metrics, 
			callbacks=[MLflowLoggingCallback()]
		)
	
		mlflow.log_params({
			"model_name": MODEL_NAME,
			"use_lora": args.use_lora,
			"max_source_length": args.max_source_length,
			"max_target_length": args.max_target_length,
			"num_train_epochs": args.num_train_epochs,
			"logging_steps": args.logging_steps,
			"eval_steps": args.eval_steps,
			"save_steps": args.save_steps,
			"num_training_samples": len(train_ds),
			"effective_batch_size":16 # edit this later
		})

		trainer.train()

		# Save final model
		output_dir = args.output_dir
		trainer.save_model(output_dir)
		tokenizer.save_pretrained(output_dir)

		# Log final metrics
		metrics = trainer.evaluate()
		mlflow.log_metrics(metrics)

		# Optionally log model artifacts
		mlflow.log_artifacts(output_dir, artifact_path="model")

if __name__ == "__main__":
	main()

