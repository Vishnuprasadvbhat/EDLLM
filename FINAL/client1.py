import flwr as fl
import torch
from transformers import AutoTokenizer
from datasets import load_dataset
from load_quantized import load_model
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TinyBertClient(fl.client.NumPyClient):
    def __init__(self, model, tokenizer, train_loader, test_loader, device):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.device = device
        logging.info("Client initialized with model: %s", model)

    def get_parameters(self):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        for epoch in range(1):
            for batch in self.train_loader:
                optimizer.zero_grad()
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logging.info("Epoch [%d]: Loss = %.4f", epoch, loss.item())
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        correct, total, loss_sum = 0, 0, 0.0
        criterion = torch.nn.CrossEntropyLoss()
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss_sum += loss.item() * input_ids.size(0)
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        accuracy = correct / total if total > 0 else 0.0
        return float(loss_sum / total), len(self.test_loader.dataset), {"accuracy": accuracy}

    def train_on_user_input(self, user_input):
        preds, is_valid = self.predict_and_validate(user_input)
        if is_valid:
            return
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        for epoch in range(20):
            inputs = self.tokenizer(user_input, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(self.device)
            labels = torch.tensor([int(input('Enter correct label (1 or 0): '))]).to(self.device)
            self.model.train()
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            logging.info("Epoch [%d]: Loss = %.4f", epoch, loss.item())
            time.sleep(1)  # Sleep for 1 second between epochs

    def predict_and_validate(self, user_input):
        inputs = self.tokenizer(user_input, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(self.device)
        outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        true_label = int(input('Enter the actual sentiment of the sentence (1 for Positive, 0 for Negative): '))
        return preds.item(), preds.item() == true_label

    def send_incremental_update(self):
        updated_parameters = self.get_parameters()
        # Simulate sending these parameters to the server
        logging.info("Sending incremental update to the server.")
        fl.client.send_parameters(updated_parameters)

    def run_incremental_learning(self):
        while True:  # Keep prompting for user input
            user_input = input("Enter a sentence for prediction and incremental training: ")
            self.train_on_user_input(user_input)
            self.send_incremental_update()
            time.sleep(5)  # Sleep for 5 seconds before next input

    def infer_user_input(self, user_input):
        # Ensure the model is in evaluation mode
        self.model.eval()
        with torch.no_grad():
            # Prepare input
            inputs = self.tokenizer(user_input, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(self.device)
            # Forward pass
            outputs = self.model(**inputs)
            logits = outputs.logits
            # Get predictions
            preds = torch.argmax(logits, dim=1)
            # Map prediction to label
            sentiment = 'Positive' if preds.item() == 1 else 'Negative'
            logging.info(f"User input: {user_input} | Predicted sentiment: {sentiment}")


def load_data():
    dataset = load_dataset("imdb")
    model_path = "C:\\Users\\vishn\\fed_up\\Quantized"
    _, tokenizer = load_model(model_path=model_path)

    def preprocess(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=128)
    
    train_data = dataset['train'].map(preprocess, batched=True)
    test_data = dataset['test'].map(preprocess, batched=True)
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=8)
    return train_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = load_data()
    model_path = "C:\\Users\\vishn\\fed_up\\Quantized"
    model, _ = load_model(model_path=model_path)
    model.to(device)

    client = TinyBertClient(model=model, tokenizer=_, train_loader=train_loader, test_loader=test_loader, device=device)
    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=client)

    client.run_incremental_learning()

    while True:
        user_input = input("Enter a sentence to predict sentiment (or 'exit' to quit): ")
        if user_input.lower() == 'exit':
            break
        client.infer_user_input(user_input)
