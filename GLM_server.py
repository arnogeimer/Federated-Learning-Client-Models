import flwr as fl

strategy = fl.server.strategy.FedAvg(
    fraction_fit=1,
    fraction_evaluate=1,
    min_fit_clients=5,
    min_evaluate_clients=5,
    min_available_clients=5,
)

fl.server.start_server(config=fl.server.ServerConfig(num_rounds=100),strategy=strategy)