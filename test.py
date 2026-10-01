from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "./qwen2.5-0.5b"

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

print("Loading model...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    torch_dtype="auto"
)

print("\nMODEL LOADED SUCCESSFULLY\n")

print(model)