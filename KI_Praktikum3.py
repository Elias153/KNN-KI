import os
import time
import glob
import torch
import pandas as pd
import plotly.express as px
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset

# Aktivierungsfunktion
def get_activation_function(name):
    return {
        "relu": F.relu,
        "sigmoid": torch.sigmoid,
        "tanh": torch.tanh
    }.get(name, lambda x: x)

# Modell
class FullyConnectedNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation='relu', dropout=False):
        super().__init__()
        self.activation = get_activation_function(activation)
        self.dropout = nn.Dropout(0.5) if dropout else nn.Identity()

        layers = []
        prev_size = input_size
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, output_size))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
            x = self.dropout(x)
        return self.layers[-1](x)

    def evaluate_metrics(self, loader, device):
        self.eval()
        total_loss, correct = 0.0, 0
        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)
                output = self(data)
                total_loss += criterion(output, target).item() * data.size(0)
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()

        avg_loss = total_loss / len(loader.dataset)
        accuracy = correct / len(loader.dataset)
        return avg_loss, accuracy

# Training
def train_model(config, train_loader, val_loader, test_loader):
    input_size, output_size = 784, 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FullyConnectedNN(input_size, output_size, config["hidden_layers"],
                             activation=config["activation"], dropout=config["dropout"]).to(device)

    optimizer_class = optim.Adam if config["optimizer"] == "adam" else optim.SGD
    optimizer = optimizer_class(model.parameters(), lr=config["lr"],
                                weight_decay=(1e-4 if config["L2"] else 0))
    criterion = nn.CrossEntropyLoss()

    history = []
    start = time.perf_counter()

    try:
        for epoch in range(config["epochs"]):
            model.train()
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                data = data.view(data.size(0), -1)

                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()
                optimizer.step()

            train_loss, _ = model.evaluate_metrics(train_loader, device)
            val_loss, val_acc = model.evaluate_metrics(val_loader, device)

            history.append({"Epoch": epoch+1, "Train Loss": train_loss,
                            "Validation Loss": val_loss, "Accuracy": val_acc})

            if (epoch + 1) % 50 == 0:
                print(f"[{epoch+1}] Train: {train_loss:.4f}, Val: {val_loss:.4f}, Acc: {val_acc:.2%}")

    except KeyboardInterrupt:
        print("⏹️ Manuell abgebrochen – Zwischenergebnisse werden gespeichert...")

    total_time = time.perf_counter() - start
    test_loss, test_acc = model.evaluate_metrics(test_loader, device)

    # Speichern
    os.makedirs("results", exist_ok=True)
    name = f"[{config['hidden_layers'][0]}]_{config['optimizer']}_dropout_{config['dropout']}_L2_{config['L2']}"
    path = f"results/{name}.csv"
    pd.DataFrame(history).to_csv(path, index=False)

    print(f"✅ Modell gespeichert unter: {path}")
    return total_time, test_acc, name

# Daten
def load_data(subset_size=2000, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    indices = torch.randperm(len(full_train))
    train_idx = indices[:subset_size]
    val_idx = indices[subset_size:]

    train = Subset(full_train, train_idx)
    val = Subset(full_train, val_idx)

    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(val, batch_size=batch_size),
        DataLoader(test_set, batch_size=batch_size)
    )

# Plot-Funktion
def plot_all_results():
    files = glob.glob("results/*.csv")
    if not files:
        print("⚠️ Keine CSV-Dateien gefunden. Erst Training ausführen!")
        return

    all_df = []
    for file in files:
        df = pd.read_csv(file)
        df["Experiment"] = os.path.basename(file).replace(".csv", "")
        all_df.append(df)

    df = pd.concat(all_df)
    df_melted = df.melt(id_vars=["Epoch", "Experiment"], value_vars=["Train Loss", "Validation Loss"],
                        var_name="Metric", value_name="Value")

    fig = px.line(df_melted, x="Epoch", y="Value", color="Experiment", line_dash="Metric",
                  title="Vergleich: Training vs. Validierung", log_y=True)
    fig.show()

# --- Hauptablauf ---
if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data()

    configs = [
        {"hidden_layers": [512], "activation": "relu", "dropout": True,  "L2": True,  "optimizer": "SGD", "lr": 0.005, "epochs": 50},
        {"hidden_layers": [512], "activation": "relu", "dropout": True,  "L2": False, "optimizer": "SGD", "lr": 0.005, "epochs": 50},
        {"hidden_layers": [512], "activation": "relu", "dropout": False, "L2": True,  "optimizer": "SGD", "lr": 0.005, "epochs": 50},
        {"hidden_layers": [512], "activation": "relu", "dropout": False, "L2": False, "optimizer": "SGD", "lr": 0.005, "epochs": 50},
    ]

    for config in configs:
        print(f"\n🚀 Starte Training: {config}")
        duration, accuracy, name = train_model(config, train_loader, val_loader, test_loader)
        print(f"✔ {name}: Dauer={duration:.2f}s, Test-Genauigkeit={accuracy:.2%}")

    # Plots anzeigen
    plot_all_results()

