# server_tinybert.py

import flwr as fl
import torch
from transformers import BertForSequenceClassification

class TinyBertServer(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy to initialize the global model with pre-trained TinyBERT."""

    def __init__(self, model, **kwargs):
        super().__init__(**kwargs)
        self.model = model

    def initialize_parameters(self, client_manager):
        """Initialize global model parameters with pre-trained TinyBERT."""
        # Create a Parameters object from the model's state_dict
        weights = [val.cpu().numpy() for val in self.model.state_dict().values()]
        return fl.common.ndarrays_to_parameters(weights)

def run_server(num_rounds=3, server_address="localhost:8080"):
    """Start a Flower server with a custom FedAvg strategy using pre-trained TinyBERT."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Load TinyBERT model on the server side
    model = BertForSequenceClassification.from_pretrained("huawei-noah/TinyBERT_General_4L_312D", num_labels=2)
    model.to(device)

    # Define the strategy and pass the model for parameter initialization
    strategy = TinyBertServer(
        model=model,
        min_fit_clients=2,  # Minimum number of clients used for training
        min_available_clients=2  # Minimum number of clients needed to start training
    )

    # Start Flower server with the custom strategy
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds)
    )

if __name__ == "__main__":
    try:
        run_server(num_rounds=1)  # Adjust the number of rounds as needed
    except KeyboardInterrupt:
        print('Shutting the Server Down')

# Time for 1 round is 49 mins 