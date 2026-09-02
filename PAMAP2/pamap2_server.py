import flwr as fl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pamap2_client import HARClient

sujeitos = [101, 102, 103, 104, 105, 106, 107, 108] 

def client_fn(cid: str):
    subject_id = sujeitos[int(cid)]
    DATA_PATH = "/content/drive/MyDrive/Datasets/PAMAP2/pamap2_federated.npz"
    dataset_fed = np.load(DATA_PATH, allow_pickle=True)
    client_data = dataset_fed[f'client_{subject_id}'].item()
    
    # O .to_client() remove aquele aviso de depreciação do log
    return HARClient(
        cid=cid, 
        X_train=client_data['X_train'], 
        y_train=client_data['y_train'], 
        X_test=client_data['X_test'], 
        y_test=client_data['y_test']
    ).to_client()

def fit_config(server_round: int):
    contexto = "base"
    if 10 <= server_round < 20:
        contexto = "ruido"
    elif 20 <= server_round < 30:
        contexto = "escala"
    elif 30 <= server_round <= 40:
        contexto = "falha_sensor"
        
    print(f"\n--- Iniciando Rodada {server_round} | Contexto Ativo: {contexto} ---\n")
    return {"local_epochs": 3, "contexto": contexto}

# 1. Função que ensina o servidor a calcular a média de acurácia dos clientes
def weighted_average(metrics):
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]
    return {"accuracy": sum(accuracies) / sum(examples)}

# 2. Adição da função de agregação na Estratégia
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, 
    fraction_evaluate=1.0,
    min_fit_clients=len(sujeitos),
    min_evaluate_clients=len(sujeitos),
    min_available_clients=len(sujeitos),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=fit_config,
    evaluate_metrics_aggregation_fn=weighted_average # NOVO
)

if __name__ == "__main__":
    # 3. Salvar o resultado da simulação em uma variável "history"
    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(sujeitos),
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=strategy,
    )

    # 4. Extração e exportação dos resultados (CSV e Gráfico)
    rounds, losses = zip(*history.losses_distributed)
    _, accuracies = zip(*history.metrics_distributed["accuracy"])

    df = pd.DataFrame({"Round": rounds, "Loss": losses, "Accuracy": accuracies})
    df.to_csv("pamap2_resultados.csv", index=False)
    print("\n[+] Resultados numéricos salvos em pamap2_resultados.csv")

    plt.figure(figsize=(12, 6))
    plt.plot(rounds, accuracies, marker='o', linestyle='-', color='b', label='Acurácia')
    plt.title("Impacto do Concept Drift na Acurácia Global (PAMAP2 - FedAvg)")
    plt.xlabel("Rodadas de Comunicação Federada")
    plt.ylabel("Acurácia Global")
    
    # Linhas verticais marcando onde cada mobilidade começa
    plt.axvline(x=10, color='red', linestyle='--', label='Início: Ruído')
    plt.axvline(x=20, color='green', linestyle='--', label='Início: Escala')
    plt.axvline(x=30, color='orange', linestyle='--', label='Início: Falha de Sensor')
    
    plt.legend()
    plt.grid(True)
    plt.savefig("pamap2_acuracia.png")
    print("[+] Gráfico visual salvo como pamap2_acuracia.png")