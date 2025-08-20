import os
import argparse
from transformers import (
	AutoTokenizer,
	AutoModelForSeq2SeqLM,
	DataCollatorForSeq2Seq,
	Seq2SeqTrainingArguments
)
from transformers import GenerationConfig
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM
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

USE_LORA = True

def training_args(args):
	return Seq2SeqTrainingArguments(
		output_dir=args.output_dir,
		evaluation_strategy="steps",
		eval_steps=args.eval_steps,
		save_strategy="steps",
		save_steps=args.save_steps,
		logging_steps=args.logging_steps,
		learning_rate=3e-4 if USE_LORA else 1e-5,
		per_device_train_batch_size=2,
		per_device_eval_batch_size=2,
		num_train_epochs=args.num_train_epochs,
		predict_with_generate=True,
		fp16=True,
		push_to_hub=False,
		report_to="none"
	)

'''
def model_init(args):
	MODEL_NAME = "Salesforce/codet5p-2b"

	# Load encoder and decoder configs
	# config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
	#config.encoder = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
	#config.decoder = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)


	tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,
				 trust_remote_code=True)
	config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
	config.eos_token_id=2
	config.bos_token_id=1
	config.pad_token_id=0
	config.decoder_start_token_id=tokenizer.convert_tokens_to_ids(['<pad>'])[0]

	model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME, 
				config=config, 
				trust_remote_code=True)
	if args.use_lora:
		model = get_peft_model(model, config_lora(args))
	model.print_trainable_parameters()

	data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
	
	return model, tokenizer, data_collator, training_args(args)
 

'''