from datasets import load_dataset
from transformers import BertTokenizer, BertForSequenceClassification
import flwr as fl
from torch.utils.data import DataLoader
import torch

# Load IMDb dataset
dataset = load_dataset("imdb")


# Define BERT model with 2 output classes (positive or negative)
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
# Initialize the tokenizer for BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Preprocessing function to tokenize text data
def preprocess_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=128)

# Tokenize dataset
train_data = dataset["train"].map(preprocess_function, batched=True)
test_data = dataset["test"].map(preprocess_function, batched=True)

# Set format for PyTorch tensors
train_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

import flwr as fl
from torch.utils.data import DataLoader

class IMDbClient(fl.client.NumPyClient):
    def __init__(self, model, train_data, test_data):
        self.model = model
        self.train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    def get_parameters(self):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def train(self, epochs):
        self.model.train()
        for epoch in range(epochs):
            for batch in self.train_loader:
                inputs, labels = batch["input_ids"].to(self.device), batch["label"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask).logits
                loss = self.criterion(outputs, labels)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate(self):
        self.model.eval()
        total, correct = 0, 0
        with torch.no_grad():
            for batch in self.test_loader:
                inputs, labels = batch["input_ids"].to(self.device), batch["label"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                outputs = self.model(input_ids=inputs, attention_mask=attention_mask).logits
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        accuracy = correct / total
        return accuracy

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.train(epochs=1)
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        accuracy = self.evaluate()
        return float(accuracy), len(self.test_loader.dataset), {}

def start_federated_client():

    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    client = IMDbClient(model=model, train_data=train_data, test_data=test_data)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)



if __name__ == "__main__":
    
    start_federated_client()
