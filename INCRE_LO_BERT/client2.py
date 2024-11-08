import flwr as fl
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from load_model import load_model
import logging

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
        logging.info("Fetching model parameters.")
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        logging.info("Setting model parameters.")
        state_dict = dict(zip(self.model.state_dict().keys(), [torch.tensor(val) for val in parameters]))
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config=None):
        logging.info("Starting model training.")
        self.set_parameters(parameters)
        self.model.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
        for epoch in range(1):
            for batch in self.train_loader:
                optimizer.zero_grad()

                # Directly use batch tensors and send them to the device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["label"].to(self.device)

                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                logging.info("Epoch [%d]: Loss = %.4f", epoch, loss.item())
        
        logging.info("Model training completed.")
        return self.get_parameters(), len(self.train_loader.dataset), {"loss": loss.item()}

    def evaluate(self, parameters, config):
        logging.info("Starting model evaluation.")
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

        if total > 0:
            accuracy = correct / total
            avg_loss = loss_sum / total
            logging.info("Model evaluation completed. Accuracy: %.4f", accuracy)
            return float(avg_loss), len(self.test_loader.dataset), {"accuracy": accuracy}
        else:
            logging.warning("No samples in evaluation, returning zero metrics.")
            return 0.0, 0, {"accuracy": 0.0}  # Return zero metrics if no samples

    def predict_and_validate(self, user_input):
        logging.info("Receiving user input for prediction: %s", user_input)
        inputs = self.tokenizer(user_input, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(self.device)
        outputs = self.model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)

        is_valid = self.validate_prediction(preds)
        logging.info("Prediction made. Predicted: %d | Validation: %s", preds.item(), is_valid)
        return preds.item(), is_valid

    def validate_prediction(self, preds):
        logging.info("Validating prediction: %d", preds.item())
        true_label = int(input('Enter the Actual sentiment of sentence if Postive enter 1 if Negative Enter 0: '))
        return preds.item() == true_label

    def train_on_user_input(self, user_input):
        preds, is_valid = self.predict_and_validate(user_input)
        if is_valid:
            print("Predicted value is same as true label. Skipping incremental training.")
            return
        
        epochs = 20
        for epoch in range(epochs):
            inputs = self.tokenizer(user_input, return_tensors='pt', padding='max_length', truncation=True, max_length=128).to(self.device)
            labels = torch.tensor([preds]).to(self.device)

            self.model.train()
            optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)
            optimizer.zero_grad()
            outputs = self.model(**inputs, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            logging.info("Epoch [%d]: Loss = %.4f", epoch, loss.item())

        print("Incremental training completed with user input.")

    def run_incremental_learning(self):
        user_input = input("Enter a sentence for prediction and incremental training: ")
        self.train_on_user_input(user_input)

def load_data():
    dataset = load_dataset("imdb")
    model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"

    _, tokenizer = load_model(model_path=model_path)

    def preprocess(batch):
        return tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128)

    train_data = dataset["train"].map(preprocess, batched=True)
    test_data = dataset["test"].map(preprocess, batched=True)

    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32, shuffle=False)

    return tokenizer, train_loader, test_loader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"

    model, _ = load_model(model_path=model_path)
    model.to(device)

    tokenizer, train_loader, test_loader = load_data()

    client = TinyBertClient(model, tokenizer, train_loader, test_loader, device)

    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

    client.run_incremental_learning()
