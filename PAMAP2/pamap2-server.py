import flwr as fl
import numpy as np
from client import HARClient

# 1. Carregamento dos dados federados engarrafados
DATA_PATH = "/content/drive/MyDrive/Datasets/PAMAP2/pamap2_federated.npz"
dataset_fed = np.load(DATA_PATH, allow_pickle=True)

# Os 8 sujeitos válidos do PAMAP2 que você pré-processou
sujeitos = [101, 102, 103, 104, 105, 106, 107, 108] 

# 2. Criação dinâmica dos clientes para a simulação
def client_fn(cid: str) -> fl.client.Client:
    subject_id = sujeitos[int(cid)]
    client_data = dataset_fed[f'client_{subject_id}'].item()
    
    return HARClient(
        cid=cid, 
        X_train=client_data['X_train'], 
        y_train=client_data['y_train'], 
        X_test=client_data['X_test'], 
        y_test=client_data['y_test']
    )

# 3. O Relógio do Concept Drift
def fit_config(server_round: int):
    """Aciona a mobilidade em blocos de rodadas."""
    contexto = "base"
    
    if 10 <= server_round < 20:
        contexto = "ruido"
    elif 20 <= server_round < 30:
        contexto = "escala"
    elif 30 <= server_round <= 40:
        contexto = "falha_sensor"
        
    print(f"\n--- Iniciando Rodada {server_round} | Contexto Ativo: {contexto} ---\n")
    return {"local_epochs": 3, "contexto": contexto}

# 4. Estratégia do Aprendizado Federado
strategy = fl.server.strategy.FedAvg(
    fraction_fit=1.0, 
    fraction_evaluate=1.0,
    min_fit_clients=len(sujeitos),
    min_evaluate_clients=len(sujeitos),
    min_available_clients=len(sujeitos),
    on_fit_config_fn=fit_config,
    on_evaluate_config_fn=fit_config
)

# 5. Execução
if __name__ == "__main__":
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=len(sujeitos),
        config=fl.server.ServerConfig(num_rounds=40),
        strategy=strategy,
    )