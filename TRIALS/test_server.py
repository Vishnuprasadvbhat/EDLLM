import flwr as fl
import torch
import signal
import threading
import sys
from load_model import load_model

class TinyBertServer(fl.server.strategy.FedAvg):
    def __init__(self, model, device, num_rounds=5):
        self.model = model
        self.device = device
        self.num_rounds = num_rounds
        initial_parameters = fl.common.ndarrays_to_parameters(self.get_initial_parameters())
        super().__init__(
            min_fit_clients=2,  # Set minimum number of clients to fit on
            min_available_clients=2,  # Minimum number of clients needed to start training
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=self.fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=self.evaluate_metrics_aggregation_fn
        )

    def get_initial_parameters(self):
        """Get initial parameters from the pre-trained global model."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit_metrics_aggregation_fn(self, metrics):
        """Aggregate fit metrics across clients."""
        aggregated_metrics = {}
        for metric in metrics[0].keys():
            aggregated_metrics[metric] = sum([m[metric] for m in metrics]) / len(metrics)
        return aggregated_metrics

    def evaluate_metrics_aggregation_fn(self, metrics):
        """Aggregate evaluation metrics across clients."""
        aggregated_metrics = {}
        for metric in metrics[0].keys():
            aggregated_metrics[metric] = sum([m[metric] for m in metrics]) / len(metrics)
        return aggregated_metrics

    def get_parameters(self):
        """Get current model parameters."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        """Set model parameters."""
        keys = list(self.model.state_dict().keys())
        state_dict = {keys[i]: torch.tensor(parameters[i]) for i in range(len(keys))}
        self.model.load_state_dict(state_dict, strict=True)

def run_server_thread(strategy, server_address="0.0.0.0:8080"):
    """Function to start the server in a separate thread."""
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        grpc_max_message_length=1024 * 1024 * 1024,
    )


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "C:\\Users\\vishn\\fed_up\\LORA_BERT"

    # Load the pre-trained model
    model, _ = load_model(model_path=model_path)
    model.to(device)


    # Create server strategy instance with the correct number of rounds
    server_strategy = TinyBertServer(model=model, device=device, num_rounds=1)

    # Start the server in a background thread
    server_thread = threading.Thread(target=run_server_thread, args=(server_strategy,))
    server_thread.start()

    
    try:
        # Main thread waits for the server thread to complete or interrupt
        while server_thread.is_alive():
            server_thread.join(1)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected, shutting down server...")
        sys.exit(0)