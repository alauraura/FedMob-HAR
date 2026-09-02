import flwr as fl
import torch
from collections import OrderedDict
from torch.utils.data import DataLoader
from pamap2_models import CNN1D_HAR
from pamap2_dataset import HARDatasetFederado
from pamap2_engine import train_model, evaluate_model

class HARClient(fl.client.NumPyClient):
    def __init__(self, cid, X_train, y_train, X_test, y_test):
        self.cid = cid
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        
        # Instancia a arquitetura e preparativos do PyTorch
        self.model = CNN1D_HAR(num_channels=40, num_classes=12)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        
        # O Maestro (Servidor) dita qual é o contexto (mobilidade) desta rodada
        contexto = config.get("contexto", "base")
        epochs = config.get("local_epochs", 3)

        dataset_train = HARDatasetFederado(self.X_train, self.y_train, contexto=contexto)
        train_loader = DataLoader(dataset_train, batch_size=32, shuffle=True)
        
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        train_model(self.model, train_loader, self.criterion, optimizer, epochs=epochs, device=self.device)
        
        return self.get_parameters(config={}), len(self.X_train), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        
        # Avalia usando o mesmo contexto para medir se a acurácia despencou
        contexto = config.get("contexto", "base")
        dataset_test = HARDatasetFederado(self.X_test, self.y_test, contexto=contexto)
        test_loader = DataLoader(dataset_test, batch_size=32, shuffle=False)
        
        loss, accuracy = evaluate_model(self.model, test_loader, self.criterion, device=self.device)
        
        return loss, len(self.X_test), {"accuracy": accuracy}