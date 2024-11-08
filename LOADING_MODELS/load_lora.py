import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Set the path to the folder containing the adapter_config.json and associated files
peft_model_id = "D:\\proj_rk\\federated\\lora_prune"  # Update with your actual path

# Load the PEFT configuration
config = PeftConfig.from_pretrained(peft_model_id)

# Load the tokenizer for the base model (this assumes your model is based on a pre-trained model)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token  # Set pad token if necessary

# Load the base model
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)

# Load the LoRA model
model = PeftModel.from_pretrained(base_model, peft_model_id)

# Move the model to the appropriate device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Model and tokenizer are now ready for inference
print("Model and tokenizer loaded successfully.")


pruned_model = model

# Example input text
input_text = "Once upon a time I lived in a "

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate text
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=50)

# Decode the generated tokens
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated text:", generated_text)
