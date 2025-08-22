import os
import argparse
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainingArguments, 
	GenerationConfig, 
	Seq2SeqTrainer, 
	TrainerCallback
)
from peft import LoraConfig, get_peft_model

def config_lora(args):
	return LoraConfig(
		r=8,
		lora_alpha=32,
		target_modules=["qkv_proj", "out_proj"],
		lora_dropout=0.1,
		bias="none",
		task_type="SEQ_2_SEQ_LM"
	)

def training_args(args):
	return Seq2SeqTrainingArguments(
		output_dir=args.output_dir,
		evaluation_strategy="steps",
		eval_steps=args.eval_steps,
		save_strategy="steps",
		save_steps=args.save_steps,
		logging_steps=args.logging_steps,
		learning_rate=3e-4 if USE_LORA else 1e-5,
		per_device_train_batch_size=4,
		per_device_eval_batch_size=4,
		num_train_epochs=args.num_train_epochs,
		gradient_accumulation_steps=4,
		predict_with_generate=True,
		load_best_model_at_end=True, 
		fp16=True,
		push_to_hub=False,
		report_to=["mlflow"]
	)

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