import torch
from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import flwr as fl
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_imdb_dataset(split='train'):
    logging.info(f"Loading IMDb dataset for {split} split...")
    dataset = load_dataset("imdb", split=split)
    tokenizer = BertTokenizer.from_pretrained("huawei-noah/TinyBERT_General_4L_312D")
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128)
    
    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    logging.info(f"Dataset for {split} split loaded successfully.")
    return tokenized_dataset

train_dataset = load_imdb_dataset(split='train')
test_dataset = load_imdb_dataset(split='test')

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16)

class IMDbClient(fl.client.NumPyClient):
    def __init__(self, model, train_loader, test_loader, device):
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device

    def get_parameters(self, config=None):
        logging.info("Getting model parameters...")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        logging.info("Setting model parameters...")
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        logging.info("Starting model fitting...")
        self.set_parameters(parameters)
        self.model.train()
        optimizer = AdamW(self.model.parameters(), lr=5e-5)
        
        for epoch in range(1):  # You can increase the number of epochs
            logging.info(f"Epoch {epoch + 1} training...")
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
        
        logging.info("Model fitting completed.")
        return self.get_parameters(), len(self.train_loader.dataset), {}

    def evaluate(self, parameters, config):
        logging.info("Starting model evaluation...")
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                
                loss = outputs.loss
                loss_sum += loss.item() * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        accuracy = correct / total
        logging.info("Model evaluation completed.")
        return float(loss_sum / total), len(self.test_loader.dataset), {"accuracy": accuracy}

def main():
    logging.info("Loading TinyBERT model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)
    model.to(device)
    logging.info("Creating client instance...")
    client = IMDbClient(model, train_loader, test_loader, device)
    
    logging.info("Starting Flower client...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
