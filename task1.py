import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Importa i modelli definiti in "models.py"
from models import MLP, NODEClassifier

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Impostazioni di iperparametri comuni
input_dim = 28 * 28  # immagini MNIST
output_dim = 10      # 10 classi
hidden_dim = 200     # iperparametro per MLP e per il blocco NODE
num_layers = 3       # numero di layers per MLP e per il blocco NODE
latent_dim = 200     # dimensione dello spazio latente per NODEClassifier
integration_times = torch.tensor([0.0, 1.0]).to(device)  # per il modello NODE

# Iperparametri di training
num_epochs = 10
batch_size = 128
learning_rate = 1e-4

# ================================
# Dataset wrapper per MNIST
# ================================

class MNISTDataset(torch.utils.data.Dataset):
    def __init__(self, subset):
        self.subset = subset

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        # Estrarre l'elemento: alcuni dataset salvati con torch.save possono avere struttura diversa.
        sample = self.subset[idx]
        if isinstance(sample, (list, tuple)):
            img, label = sample
        else:
            # Se l'elemento ha attributi .data e .target (come in torchvision)
            img, label = sample[0], sample[1]
        # Flatten l'immagine (28x28 -> 784)
        img = img.view(-1)
        return img, label

def load_mnist_datasets():
    # Carica i file salvati in precedenza
    train_subset = torch.load(os.path.join('./', 'mnist_train.pt'), weights_only=False)
    test_subset = torch.load(os.path.join('./', 'mnist_test.pt'), weights_only=False)
    train_dataset = MNISTDataset(train_subset)
    test_dataset = MNISTDataset(test_subset)
    return train_dataset, test_dataset

# ================================
# Funzioni di training e validazione
# ================================

def train_one_epoch(model, optimizer, criterion, dataloader):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # forward
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        # Calcola accuracy
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# ================================
# Evaluation sul test set
# ================================

def test_evaluation(model, test_dataset):
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct = 0
    total = 0
    running_loss = 0.0

    start_infer = time.time()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += inputs.size(0)
    infer_time = time.time() - start_infer
    test_loss = running_loss / total
    test_acc = correct / total
    return test_loss, test_acc, infer_time

# ================================
# Main
# ================================

def main():
    # Carica i dataset MNIST salvati in precedenza
    train_dataset, test_dataset = load_mnist_datasets()

    # Definiamo i modelli da testare: MLP e NODE (NODEClassifier per la classificazione)
    models_to_test = [
        {"name": "MLP", "class": MLP},
        {"name": "NODE", "class": NODEClassifier}
    ]

    # Allenamento finale sul training set completo e valutazione sul test set
    final_results = {}
    for m in models_to_test:
        print(f"\n=== Allenamento finale del modello {m['name']} sul training set completo ===")
        if m["name"] == "MLP":
            model = m["class"](input_dim=input_dim, output_dim=output_dim, hidden_dim=hidden_dim, num_layers=num_layers)
        elif m["name"] == "NODE":
            model = m["class"](input_dim=input_dim, latent_dim=latent_dim,
                                hidden_dim=hidden_dim, num_layers=num_layers,
                                integration_times=integration_times, output_dim=output_dim)
        else:
            raise ValueError("Modello non riconosciuto")
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        start_train = time.time()
        for epoch in range(num_epochs):
            train_loss, train_acc = train_one_epoch(model, optimizer, criterion, train_loader)
            print(f"Final Training Epoch {epoch+1}/{num_epochs} \t -> loss: {train_loss:.4f}, acc: {train_acc:.4f}")
        final_train_time = time.time() - start_train

        # Valutazione sul test set
        test_loss, test_acc, infer_time = test_evaluation(model, test_dataset)
        num_params = count_parameters(model)
        print(f"\nModello {m['name']} -> Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        print(f"Numero di parametri: {num_params}")
        print(f"Tempo di training finale: {final_train_time:.2f} sec")
        print(f"Tempo di inferenza sul test set: {infer_time:.2f} sec")
        final_results[m["name"]] = {
            "test_loss": test_loss,
            "test_acc": test_acc,
            "num_params": num_params,
            "final_train_time": final_train_time,
            "infer_time": infer_time
        }

    # Riassunto finale dei risultati
    print("\n===== RISULTATI FINALI =====")
    for model_name, res in final_results.items():
        print(f"Modello: {model_name}")
        print(f"  Test Loss: {res['test_loss']:.4f}")
        print(f"  Test Acc: {res['test_acc']:.4f}")
        print(f"  Parametri: {res['num_params']}")
        print(f"  Tempo training finale: {res['final_train_time']:.2f} sec")
        print(f"  Tempo inferenza: {res['infer_time']:.2f} sec")
        print("--------------------------------------------------")

if __name__ == '__main__':
    main()