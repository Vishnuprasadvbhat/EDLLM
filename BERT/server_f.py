def run_server(num_rounds=3, server_address="localhost:8080"):
    """Start a Flower server with a simple FedAvg strategy."""
    import flwr as fl
    import torch
    from transformers import BertForSequenceClassification
    
    # Load pretrained model (global model initialization)
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)
    
    # Function to get the initial parameters from the global model
    def get_initial_parameters():
        return [val.cpu().numpy() for val in model.state_dict().values()]

    # Define a simple FedAvg strategy with initialized model weights
    strategy = fl.server.strategy.FedAvg(
        min_fit_clients=2,  # Minimum number of clients used for training
        min_available_clients=2,  # Minimum number of clients needed to start training
        initial_parameters=fl.common.ndarrays_to_parameters(get_initial_parameters())  # Set initial model parameters
    )

    # Start Flower server
    fl.server.start_server(
        server_address=server_address,
        strategy=strategy,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        grpc_max_message_length= 1024 *1024 * 1024 
    )

if __name__ == "__main__":
    try:
        run_server(num_rounds=1)
    except KeyboardInterrupt:
        print('Shutting the Server Down')
