# bugspell :bug:
_STILL A WORK IN PROGRESS_

Spell checker for both codes and natural language.

This spell checked was fine-tuned based on CodeT5p-2b with a mixture of Python code, Markdown files and Natural Language (English)

## Installation and usage

### Installation
Feel free to clone the repo and create container or environment using `environment.yaml`or `requirements.txt`

### Data loader
Run this script to generate the data mixture that was used for fine-tuning and evaluating. 

```bash
python utils/dataloader.py
```

## Fine-tuning
For fine-tuning, run this script.

```bash
python trainer.py --train_ds_dir {path to your training dataset} \
                --eval_ds_dir {path to your validation dataset} \
                --output_dir {path to the directory where tuned model will be saved} \
                --use_lora True
```

## Inference
For inference, you can either input text through CLI or parse the input text file for multiple text lines. 

### CLI

```bash
python inference_cli.py
```

### File as parser argument

```bash
python inference.py --model_path $MODEL_PATH \
                    --input_file {path to your input text file} \
                    --output_file {path to your output file}
```

## Evaluation
Current there are 3 metrics being added in the evaluation script: sacrebleu, cer and wer. However, cer and wer will be better to look at to check on the model's efficiency!

```bash
python evaluation.py --model_path $MODEL_PATH \
                     --test_file "{path to the test dataset}
```

## Next todos or possible future features
- Integrate this code with a VSCode extension
- Extend this to real-time spell checker
- Add OCR for additional feature

## Other notes