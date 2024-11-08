import datasets
import flwr as f
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAdam, FedAdagrad, FedAvg, FedProx
from transformers import AutoTokenizer, BertForSequenceClassification

from load_data import partition_data
# print(client_train_data[0])


def load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)

    return model, tokenizer

