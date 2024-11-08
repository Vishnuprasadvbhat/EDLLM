import flwr as fl
import torch
from transformers import AutoModelForSequenceClassification
from datasets import load_dataset

# Load the IMDb dataset
dataset = load_dataset("imdb")
train_data = dataset['train']
test_data = dataset['test']

# Load TinyBERT model
model_name = "huawei-noah/TinyBERT_General_4L_312D"
global_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

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
    # Define a custom strategy with specific number of rounds
    strategy = CustomFedAvg(
        min_fit_clients=2,  # Minimum number of clients to be included in each training round
        min_available_clients=2,  # Minimum number of clients that need to be connected
    )

    # Create a server configuration with number of federated learning rounds
    server_config = fl.server.ServerConfig(num_rounds=5)

    # Start Flower server with custom strategy and server configuration
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=server_config,
        strategy=strategy,
    )
