import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F
from check_size import get_directory_size
model_path = "C:\\Users\\vishn\\fed_up\\Quantized"
import os
import psutil 

process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / (1024 * 1024)  # in MB
print(mem_before)
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


model, tokenizer = load_model(model_path=model_path)

print(model)


# define label mapping (adjust based on your specific model's labels)
label_map = {0: "Negative", 1: "Positive"}

# texts = ["The movie was fantastic!", "The film was terrible."]
# texts = ["The movie was okay okay!", "I dind't like this film."]
texts = ['BAD', 'BAD', "BAD Movie ever", "BEST"]

# Tokenize batch of texts
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)

mem_after = process.memory_info().rss / (1024 * 1024)  # in MB
print(mem_after)
print(f"Memory used by inference: {mem_after - mem_before:.2f} MB")

# Convert logits to probabilities using softmax
probabilities = F.softmax(outputs.logits, dim=-1)
print(probabilities)



# Get the predicted class (with the highest probability)
predicted_classes = torch.argmax(probabilities, dim=-1)
print(predicted_classes)


# Map predicted class indices to label names
predicted_labels = [label_map[int(idx)] for idx in predicted_classes]
print(predicted_labels)



# Print the results
for text, label, prob in zip(texts, predicted_labels, probabilities):
    print(f"Text: '{text}' | Predicted Label: {label} | Probabilities: {prob.tolist()}")

'''
tensor([[0.5167, 0.4833],
        [0.5174, 0.4826]])
tensor([0, 0])
['Positive', 'Positive']
Text: 'The movie was okay okay!' | Predicted Label: Positive | Probabilities: [0.5167193412780762, 0.4832806885242462]
Text: 'I dind't like this film.' | Predicted Label: Positive | Probabilities: [0.5174391865730286, 0.48256081342697144]'''


'''

tensor([[0.4793, 0.5207],
        [0.4793, 0.5207],
        [0.4793, 0.5207],
        [0.4790, 0.5210]])
tensor([1, 1, 1, 1])
['Negative', 'Negative', 'Negative', 'Negative']
Text: 'BAD' | Predicted Label: Negative | Probabilities: [0.47927355766296387, 0.5207264423370361]
Text: 'BAD' | Predicted Label: Negative | Probabilities: [0.47927355766296387, 0.5207264423370361]
Text: 'BAD Movie ever' | Predicted Label: Negative | Probabilities: [0.4792587161064148, 0.5207412242889404]
Text: 'BEST MOVIE EVER' | Predicted Label: Negative | Probabilities: [0.47895047068595886, 0.5210494995117188]
'''

# Define a file path for saving the model
file_path = "temp_model.pth"

# Save the quantized model
torch.save(model.state_dict(), file_path)

# Get the file size in bytes, convert to MB
file_size = os.path.getsize(file_path) / (1024 * 1024)
print(f"Model size on disk: {file_size:.2f} MB")

# Optionally, delete the temporary file if you don't need it
# os.remove(file_path)