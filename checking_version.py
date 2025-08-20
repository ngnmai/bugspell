import sys
import subprocess
import importlib.util


def run_cmd(cmd):
    try:
        result = subprocess.check_output(cmd, shell=True, text=True).strip()
        return result
    except subprocess.CalledProcessError:
        return "Not found"


def check_module(module_name):
    spec = importlib.util.find_spec(module_name)
    return spec.origin if spec else None

print("===== Python Info =====")
print("Python Executable:", sys.executable)
print("Python Version:", sys.version)

print("\n===== pip Info =====")
print("pip Executable:", run_cmd("which pip"))
print("pip Version:", run_cmd("pip --version"))

print("\n===== Transformers Info =====")
try:
    import transformers
    print("Transformers Version:", transformers.__version__)
    print("Transformers Location:", transformers.__file__)
except ImportError:
    print("Transformers not installed.")

print("\n===== PyTorch Info =====")
try:
    import torch
    print("PyTorch Version:", torch.__version__)
    print("PyTorch Location:", torch.__file__)
    print("CUDA Available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU Name:", torch.cuda.get_device_name(0))
        print("GPU Count:", torch.cuda.device_count())
except ImportError:
    print("PyTorch not installed.")

print("\n===== PEFT Info =====")
try:
    import peft
    print("PEFT Version:", getattr(peft, "__version__", "Unknown"))
    print("PEFT Location:", peft.__file__)
except ImportError:
    print("PEFT not installed.")

print("\n===== MLflow Info =====")
try:
    import mlflow
    print("MLflow Version:", mlflow.__version__)
    print("MLflow Location:", mlflow.__file__)
except ImportError:
    print("MLflow not installed.")


from transformers import AutoModelForSeq2SeqLM

# Load the model with remote code
model = AutoModelForSeq2SeqLM.from_pretrained("Salesforce/codet5p-2b", trust_remote_code=True)

# Print all named modules
print(model)
