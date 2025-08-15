Project goal: 
- Fine-tune a model that can do spell checking on both code types and normal language typoes
- Future implementation: Vscode extension or aoo


Tech stacks: 
- Huggingface
- torch

Base model: 
- CodeT5+: light weight and already pretrained on code languages. 

Datasets to fine-tune:
- 50% natural language
- 35% code
- 15% mixed prose/code Q&A from open platform