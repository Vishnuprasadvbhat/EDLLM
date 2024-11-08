# Federated Learning and Incremental Learning Framework

This project implements a Federated Learning framework using a pruned BERT model and extends it into an Incremental Learning flow. The primary goal is to optimize Large Language Models (LLMs) for edge devices by integrating federated learning with real-time user input-based incremental training. The implementation leverages the Flower (`flwr`) framework for federated learning and incorporates advanced strategies for incremental learning. This README provides a detailed overview of the project, its purpose, technical approaches, and key code snippets for better comprehension.

## Project Overview
The primary objectives of this project include:
- **Federated Learning Implementation**: Establishing a federated learning framework using a pruned BERT model to distribute training across client nodes while keeping user data private.
- **Incremental Learning Flow**: Enabling real-time updates to the model by fine-tuning client nodes using new user input data, optimizing the model's accuracy incrementally.

---

## Federated Learning Implementation

In the initial stage, we implemented a federated learning approach using a pruned BERT model. The federated learning setup involved defining a server and multiple client nodes using the Flower framework (`flwr`). 

### **Server Code for Federated Learning**

The server is responsible for initializing the global model, handling communication between client nodes, and coordinating the aggregation of parameters.

#### Key Aspects:
- **Initialization of Global Model**: We started with a pre-trained BERT model to serve as the global model.
- **FedAvg Strategy**: The Flower framework's `FedAvg` strategy is utilized to average the parameters across client nodes after each round.

**Code Snippet for Server Implementation:**
```python
import flwr as fl
import torch
from transformers import BertForSequenceClassification

class BertServer(fl.server.strategy.FedAvg):
    def __init__(self, model, device, num_rounds=5):
        self.model = model
        self.device = device
        super().__init__(initial_parameters=self.get_initial_parameters())

    def get_initial_parameters(self):
        """Extract initial model parameters from the state_dict."""
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        """Update model parameters."""
        keys = list(self.model.state_dict().keys())
        state_dict = {keys[i]: torch.tensor(parameters[i]) for i in range(len(keys))}
        self.model.load_state_dict(state_dict, strict=True)

# Initialize and start the server
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
model.to(device)
strategy = BertServer(model=model, device=device, num_rounds=5)

fl.server.start_server(server_address="0.0.0.0:8080", strategy=strategy)
```

### **Client Code for Federated Learning**

Each client is responsible for receiving the model's initial parameters, performing local training, and sending the updated parameters back to the server. The Flower framework helps to abstract and automate much of this communication.

#### Key Aspects:
- **Data Loading**: We used the IMDb dataset for training and evaluation.
- **Local Training**: Each client trains locally on its respective dataset split.
- **Model Updates**: The updated model parameters are shared with the server after each round.

**Code Snippet for Client Implementation:**
```python
import torch
# from transformers import BertTokenizer, BertForSequenceClassification
from datasets import load_dataset
import flwr as fl
from torch.utils.data import DataLoader
from torch.optim import AdamW
import logging
from load_model import load_model

# Set up logging
logging.basicConfig(level=logging.INFO)

def load_imdb_dataset(split='train'):
    logging.info(f"Loading IMDb dataset for {split} split...")
    dataset = load_dataset("imdb", split=split)
    
    model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"

    _ , tokenizer = load_model(model_path)
    
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
    model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"
    model, _ = load_model(model_path)
    model.to(device)
    logging.info("Creating client instance...")
    client = IMDbClient(model, train_loader, test_loader, device)
    
    logging.info("Starting Flower client...")
    fl.client.start_numpy_client(server_address="localhost:8080", client=client)

if __name__ == "__main__":
    main()
```

### **Stopping Server Gracefully**

To stop the server gracefully when running rounds of federated learning, we explored using signal handling for `SIGINT` (keyboard interrupts). The final solution included capturing the interrupt signal to shut down the server without abrupt termination.

---

## Incremental Learning Flow

After federated learning, we implemented an incremental learning flow to enable real-time user input-based updates at client nodes. This approach utilized the existing client model from federated learning rounds and incorporated new data for fine-tuning.

### **Incremental Training Approach**

**Core Idea**: Reuse the client model from federated learning without initializing a separate model. The incremental learning flow involves directly updating the model with new user input data to improve accuracy and relevance.

#### Key Steps:
1. **Load the Existing Client Model**: Instead of initializing a new model, load the model that was used in federated learning.
2. **Real-Time Input Processing**: Capture real-time input data from the user or application.
3. **Fine-Tuning**: Use the new data to perform in-place updates on the client model's parameters.
4. **Synchronize Parameters**: Ensure that the parameters updated during incremental learning are shared back with the server in subsequent federated learning rounds.

**Code Snippet for Incremental Training:**
```python
def incremental_train(self, new_data_loader):
    """Fine-tune the existing client model using new user input data."""
    self.model.train()
    optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-5)

    for batch in new_data_loader:
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["label"].to(self.device)
        
        optimizer.zero_grad()
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    
    logging.info("Incremental training completed.")
```

### **Seamless Integration with Federated Learning**

We ensured that the incremental learning approach is integrated seamlessly with the federated learning process by synchronizing the updated client parameters with the server. This maintains consistency in the model and enhances the relevance of training for the task-specific domain.

---

## Challenges and Key Considerations

1. **Data Privacy**: The federated learning framework ensures that user data is kept private and only model updates (gradients) are shared.
2. **Resource Constraints**: The pruned BERT model and TinyBERT were chosen to optimize resource consumption for edge devices.
3. **Real-Time Validation**: During incremental learning, external validation mechanisms are employed to verify updates.

---

## Conclusion

This project leverages federated learning and incremental training to optimize the performance of LLMs while maintaining data privacy. By combining these approaches, we effectively balance global model generalization with local adaptation, enhancing both the model’s efficiency and relevance.

Feel free to explore the code files for more in-depth understanding and reach out for further clarification!

---

