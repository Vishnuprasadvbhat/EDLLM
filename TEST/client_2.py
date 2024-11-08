import flwr as fl
import torch
from transformers import BertForSequenceClassification
from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']

# Load BERT model
model_name = "bert-base-uncased"
global_model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

# Define strategy for federated learning
class CustomFedAvg(fl.server.strategy.FedAvg):
    def aggregate_fit(self, server_round, results, failures):
        # Call the parent class's aggregate_fit method
        parameters_aggregated = super().aggregate_fit(server_round, results, failures)

        # Perform model fine-tuning or any additional optimization if needed
        return parameters_aggregated

def get_parameters(model):
    """Get model parameters as a list of numpy arrays."""
    return [param.detach().numpy() for param in model.parameters()]

def set_parameters(model, parameters):
    """Set model parameters from a list of numpy arrays."""
    for param, new_param in zip(model.parameters(), parameters):
        param.data = torch.tensor(new_param)

# Start server
if __name__ == "__main__":
    
    try:
      # Define a custom strategy with minimum client requirements
      strategy = CustomFedAvg(
          min_fit_clients=2,  # Minimum number of clients to be included in each training round
          min_available_clients=2,  # Minimum number of clients that need to be connected
      )

      # Start Flower server with custom strategy
      fl.server.start_server(
          server_address="0.0.0.0:8080",
          strategy=strategy,

      )
    except KeyboardInterrupt:
        print('Server Shutting Down')