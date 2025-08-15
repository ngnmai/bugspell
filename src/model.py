import os
import argparse
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments
)
from peft import LoraConfig, get_peft_model

# Environment variables
MODEL_NAME = "Salesforce/codet5p-2b"
OUTPUT_DIR = "./output"
USE_LORA = True
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 512

def parse_args():
    parser = argparse.ArgumentParser()
    parser.argument_default("--output_dir", type=str, default=OUTPUT_DIR)
    parser.add_argument("--use_lora", type=bool, default=USE_LORA)
    parser.add_argument("--max_source_length", type=int, default=MAX_SOURCE_LENGTH)
    parser.add_argument("--max_target_length", type=int, default=MAX_TARGET_LENGTH)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--logging_step", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=500)
    return parser.parse_args()


def config_lora(args):
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
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
        logging_steps=args.logging_step,
        learning_rate=3e-4 if USE_LORA else 1e-5,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.num_train_epochs,
        predict_with_generate=True,
        fp16=True,
        push_to_hub=False,
        report_to="none"
    )


def model_init(args):
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    if args.use_lora:
        model = get_peft_model(model, config_lora(args))
    model.print_trainable_parameters()

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    return model, tokenizer, data_collator, training_args(args)


    
