import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"


def load_model(model_path):

  # Load tokenizer
  tokenizer = AutoTokenizer.from_pretrained(model_path)

  # Load model using the correct method for SafeTensors
  model = AutoModelForSequenceClassification.from_pretrained(
      model_path,
      use_safetensors=True  # Ensure this is still there for SafeTensors format
  ).to(torch.float32)


  # print(model)
  return model , tokenizer


