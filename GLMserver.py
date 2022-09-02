import flwr as fl

strategy = strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,  # Sample 10% of available clients for the next round
    min_fit_clients=5,  # Minimum number of clients to be sampled for the next round
    min_available_clients=5,  # Minimum number of clients that need to be connected to the server before a training round can start
)

if __name__ == "__main__":
    fl.server.start_server(config=fl.server.ServerConfig(num_rounds=5),strategy=strategy)