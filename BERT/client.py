import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import flwr as fl
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F

def load_imdb_dataset(split='train'):
    dataset = load_dataset("imdb", split=split)
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    return tokenized_dataset

train_dataset = load_imdb_dataset(split='train')
test_dataset = load_imdb_dataset(split='test')

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=12)


class IMDbClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader

    def get_parameters(self, config=None):  # Added config as an optional parameter
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        for epoch in range(1):  # You can increase the number of epochs
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["label"]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["label"]
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                loss_sum += loss.item() * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total if total > 0 else 0.0

        # Debug logs to trace potential issues
        print(f"Evaluation - Loss Sum: {loss_sum}, Total: {total}, Accuracy: {accuracy}")

        if total == 0:
            print("Warning: Total number of samples in evaluation is zero.")
        
        return float(loss_sum / total) if total > 0 else float('nan'), total, {"accuracy": accuracy}

def main():
    # Load and prepare the model
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Create the client instance
    client = IMDbClient(model, train_loader, test_loader)
    
    # Start Flower client
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
