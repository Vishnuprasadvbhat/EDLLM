import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch.nn.functional as F

model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model using the correct method for SafeTensors
model = AutoModelForSequenceClassification.from_pretrained(
    model_path,
    use_safetensors=True  # Ensure this is still there for SafeTensors format
).to(torch.float32)


print(model)

# Define label mapping (adjust based on your specific model's labels)
label_map = {1: "Negative", 0: "Positive"}

# texts = ["The movie was fantastic!", "The film was terrible."]
texts = ["The movie was okay okay!", "I dind't like this film."]

# Tokenize batch of texts
inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

# Forward pass
with torch.no_grad():
    outputs = model(**inputs)


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


# FOR  texts = ["The movie was fantastic!", "The film was terrible."]


# tensor([[0.5011, 0.4989],
#         [0.4989, 0.5011]])
# tensor([0, 1])
# ['Positive', 'Negative']


# Text: 'The movie was fantastic!' | Predicted Label: Positive | Probabilities: [0.5011333227157593, 0.4988667070865631]
# Text: 'The film was terrible.' | Predicted Label: Negative | Probabilities: [0.49891090393066406, 0.5010890960693359]


# tensor([[0.4994, 0.5006],
#         [0.4976, 0.5024]])
# tensor([1, 1])
# ['Negative', 'Negative']
# Text: 'The movie was okay okay!' | Predicted Label: Negative | Probabilities: [0.4993659555912018, 0.500633955001831]
# Text: 'I dind't like this film.' | Predicted Label: Negative | Probabilities: [0.4976396858692169, 0.5023603439331055]