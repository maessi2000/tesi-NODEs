import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Importa i modelli definiti in models.py
from models import MLP, NODE

# Parametri per il sistema FHN:
input_dim = 2    # lo stato (FitzHugh-Nagumo ha 2 variabili)
output_dim = 2   # target: segmento (step, batch, 2)

# Iperparametri per le reti (MLP e NODE devono avere input_dim == output_dim)
hidden_dim = 200
num_layers = 3

# Iperparametri di training
num_epochs = 10      # numero di epoche per il training finale 
batch_size = 128
learning_rate = 1e-4
batch_time = 20      # lunghezza in step del segmento da campionare

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================================
# Dataset per il sistema FHN
# ================================

class FHN_Dataset(torch.utils.data.Dataset):
    def __init__(self, dataset_list):
        """
        dataset_list è una lista di dizionari, ciascuno con chiavi:
          - 'y0': condizione iniziale (tensor di shape (2,))
          - 't': vettore dei tempi (tensor di shape (steps,))
          - 'trajectory': traiettoria simulata (tensor di shape (steps, 2))
          - 'noise_level': livello di rumore usato (opzionale)
        """
        self.data = dataset_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Restituisce il sample (dizionario)
        return self.data[idx]

def load_fhn_datasets():
    train_data = torch.load(os.path.join('./', 'fitzhugh_train.pt'), weights_only=False)
    test_data = torch.load(os.path.join('./', 'fitzhugh_test.pt'), weights_only=False)
    train_dataset = FHN_Dataset(train_data)
    test_dataset = FHN_Dataset(test_data)
    return train_dataset, test_dataset

# ================================
# Funzione di campionamento: get_batch_segment
# ================================

def get_batch_segment(dataset, batch_time, batch_size):
    """
    Campiona un batch di segmenti dalle traiettorie del dataset.
    
    Se 'dataset' è un Subset, estrae il dataset originale e gli indici validi.
    
    Per ogni campione:
      - Sceglie casualmente un sample dal dataset (il sample è un dizionario con chiave 'trajectory').
      - Sceglie un indice s tale che s + batch_time sia minore della lunghezza della traiettoria.
      - L'input (batch_y0) sarà lo stato in posizione s.
      - Il target (batch_y) sarà la sequenza:
            y[s], y[s+1], ..., y[s+batch_time-1]
    
    Restituisce:
      - batch_y0: tensor di shape (batch_size, 2)
      - batch_t: vettore dei tempi relativi (tensor di shape (batch_time,))
      - batch_y: tensor di shape (batch_time, batch_size, 2)
    """
    # Se il dataset è un Subset, estraiamo il dataset originale e gli indici validi
    if isinstance(dataset, torch.utils.data.Subset):
        underlying_data = dataset.dataset.data
        valid_indices = dataset.indices
    else:
        underlying_data = dataset.data
        valid_indices = np.arange(len(underlying_data))
    
    traj_length = underlying_data[0]['trajectory'].shape[0]
    batch_y0_list = []
    batch_y_list = []
    for _ in range(batch_size):
        chosen_index = np.random.choice(valid_indices)
        sample = underlying_data[chosen_index]  # sample è un dizionario
        max_start = traj_length - batch_time
        s = np.random.randint(0, max_start)
        traj = sample['trajectory']  # tensore di forma (traj_length, 2)
        segment = traj[s: s + batch_time]  # (batch_time, 2)
        batch_y0_list.append(traj[s])
        batch_y_list.append(segment)
    batch_y0 = torch.stack(batch_y0_list, dim=0)  # (batch_size, 2)
    batch_y = torch.stack(batch_y_list, dim=1)      # (batch_time, batch_size, 2)
    batch_t = torch.linspace(0, 1, batch_time)      # (batch_time,)
    return batch_y0, batch_t, batch_y

# ================================
# Funzione per simulare in modo iterativo (per il modello MLP)
# ================================

def simulate_segment(model, y0, t_span):
    ys = [y0]
    current = y0
    for _ in range(len(t_span)-1):
        current = model(current)
        ys.append(current)
    return torch.stack(ys, dim=0)

# ================================
# Funzioni di training e validazione
# ================================

def train_one_epoch(model, optimizer, criterion, dataset, num_iterations=100, model_type="NODE"):
    model.train()
    running_loss = 0.0
    total = 0
    for _ in range(num_iterations):
        batch_y0, batch_t, batch_y = get_batch_segment(dataset, batch_time, batch_size)
        batch_y0 = batch_y0.to(device)
        batch_t = batch_t.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        if model_type == "NODE":
            # Per il modello NODE integriamo usando odeint sul campo 'func'
            pred_y = odeint(model.func, batch_y0, batch_t, method='rk4')
        else:  # MLP
            pred_y = simulate_segment(model, batch_y0, batch_t)
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
        total += batch_size
    epoch_loss = running_loss / total
    return epoch_loss

def test_evaluation(model, dataset, num_iterations=50, model_type="NODE"):
    criterion = nn.MSELoss()
    model.eval()
    running_loss = 0.0
    total = 0
    start_infer = time.time()
    with torch.no_grad():
        for _ in range(num_iterations):
            batch_y0, batch_t, batch_y = get_batch_segment(dataset, batch_time, batch_size)
            batch_y0 = batch_y0.to(device)
            batch_t = batch_t.to(device)
            batch_y = batch_y.to(device)
            if model_type == "NODE":
                pred_y = odeint(model.func, batch_y0, batch_t, method='rk4')
            else:
                pred_y = simulate_segment(model, batch_y0, batch_t)
            loss = criterion(pred_y, batch_y)
            running_loss += loss.item() * batch_size
            total += batch_size
    infer_time = time.time() - start_infer
    test_loss = running_loss / total
    return test_loss, infer_time

#############################################
# Plot dei risultati: traiettorie
#############################################

def plot_results(trained_models, num_samples=1, t_sim_steps=100):
    t_span = torch.linspace(0, 50, t_sim_steps).to(device)
    raw_test = torch.load(os.path.join('./', 'fitzhugh_test.pt'), weights_only=False)
    
    for _ in range(num_samples):
        idx = np.random.randint(0, len(raw_test))
        sample = raw_test[idx]
        y0 = sample['y0'].unsqueeze(0).to(device)  # (1, 2)
        ground_truth = sample['trajectory'].cpu().numpy()  # (steps, 2)
        # Per il plotting usiamo le funzioni di simulazione già definite
        pred_MLP = simulate_segment(trained_models["MLP"], y0, t_span).squeeze(1).detach().cpu().numpy()  # (t_sim_steps, 2)
        pred_NODE = odeint(trained_models["NODE"].func, y0, t_span, method='rk4').squeeze(1).detach().cpu().numpy()
        
        plt.figure(figsize=(8,6))
        plt.plot(ground_truth[:, 0], ground_truth[:, 1], 'k-', linewidth=2, label="Ground Truth")
        plt.scatter(ground_truth[0, 0], ground_truth[0, 1], c='k', marker='o', s=40, label="Start GT")
        plt.plot(pred_MLP[:, 0], pred_MLP[:, 1], 'b--', linewidth=2, label="MLP Prediction")
        plt.scatter(pred_MLP[0, 0], pred_MLP[0, 1], c='b', marker='o', s=40, label="Start MLP")
        plt.plot(pred_NODE[:, 0], pred_NODE[:, 1], 'r-.', linewidth=2, label="NODE Prediction")
        plt.scatter(pred_NODE[0, 0], pred_NODE[0, 1], c='r', marker='o', s=40, label="Start NODE")
        plt.xlabel("Variabile 1")
        plt.ylabel("Variabile 2")
        plt.title(f"Traiettoria di esempio (sample idx: {idx})")
        plt.legend()
        plt.grid(True)
        plt.show()

# ================================
# Training Finale e Main
# ================================

def main():
    train_dataset, test_dataset = load_fhn_datasets()

    models_to_test = [
        {"name": "MLP", "class": MLP},
        {"name": "NODE", "class": NODE}
    ]

    """ results = {}
    for m in models_to_test:
        val_loss, train_time = kfold_training(m["class"], m["name"], train_dataset)
        results[m["name"]] = {"val_loss": val_loss, "kfold_training_time": train_time} """

    trained_models = {}
    final_results = {}
    for m in models_to_test:
        print(f"\n=== Allenamento finale del modello {m['name']} sul training set completo ===")
        if m["name"] == "MLP":
            model = m["class"](input_dim=input_dim, output_dim=output_dim,
                                hidden_dim=hidden_dim, num_layers=num_layers)
        elif m["name"] == "NODE":
            model = m["class"](input_dim=input_dim, hidden_dim=hidden_dim,
                                num_layers=num_layers, integration_times=torch.linspace(0, 1, batch_time).to(device))
        else:
            raise ValueError("Modello non riconosciuto")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()
        start_train = time.time()
        for epoch in range(num_epochs):
            epoch_loss = train_one_epoch(model, optimizer, criterion, train_dataset, num_iterations=100, model_type=m["name"])
            print(f"Final Training Epoch {epoch+1}/{num_epochs} \t -> loss: {epoch_loss:.10f}")
        final_train_time = time.time() - start_train

        test_loss, infer_time = test_evaluation(model, test_dataset, num_iterations=50, model_type=m["name"])
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nModello {m['name']} -> Test Loss: {test_loss:.6f}")
        print(f"Numero di parametri: {num_params}")
        print(f"Tempo di training finale: {final_train_time:.2f} sec")
        print(f"Tempo di inferenza sul test set: {infer_time:.2f} sec")
        final_results[m["name"]] = {
            "test_loss": test_loss,
            "num_params": num_params,
            "final_train_time": final_train_time,
            "infer_time": infer_time
        }
        trained_models[m["name"]] = model

    print("\n===== RISULTATI FINALI =====")
    for model_name, res in final_results.items():
        print(f"Modello: {model_name}")
        print(f"  Test Loss: {res['test_loss']:.6f}")
        print(f"  Parametri: {res['num_params']}")
        print(f"  Tempo training finale: {res['final_train_time']:.2f} sec")
        print(f"  Tempo inferenza: {res['infer_time']:.2f} sec")
        print("--------------------------------------------------")
    return trained_models, test_dataset

if __name__ == '__main__':
    trained_models, test_dataset = main()
    plot_results(trained_models, num_samples=5, t_sim_steps=500)