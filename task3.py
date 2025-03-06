import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchdiffeq import odeint

# Importa i modelli definiti in "models.py": MLP, NODE, HNN, LNN
from models import MLP, NODE, HNN, LNN

#############################################
# Parametri e iperparametri
#############################################

input_dim = 2                   # Lo stato: ad esempio [x, v] o [q, p]
output_dim = input_dim          
hidden_dim = 200
num_layers = 3
batch_time = 20                 
batch_size = 128
num_epochs = 10
learning_rate = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#############################################
# Caricamento del dataset conservativo
#############################################

def load_conservative_datasets():
    train_data = torch.load(os.path.join('./', 'conservative_train.pt'), weights_only=False)
    test_data  = torch.load(os.path.join('./', 'conservative_test.pt'), weights_only=False)

    # Definiamo un dataset che restituisce il sample (dizionario)
    dataset = ConservativeDataset(train_data)  # per training
    test_dataset = ConservativeDataset(test_data)

    return dataset, test_dataset

class ConservativeDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        """
        Ogni sample è un dizionario con chiavi:
          - 'y0': condizione iniziale (tensor shape (2,))
          - 't': vettore dei tempi (tensor shape (steps,))
          - 'trajectory': traiettoria simulata (tensor shape (steps, 2))
        """
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

#############################################
# Funzione di campionamento a segmenti
#############################################

def get_batch_segment(dataset, batch_time, batch_size):
    # Se dataset è un Subset, estraiamo il dataset originale e gli indici validi
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
        sample = underlying_data[chosen_index]              # sample è un dict
        max_start = traj_length - batch_time
        s = np.random.randint(0, max_start)
        traj = sample['trajectory']                         # tensor shape (traj_length, 2)
        segment = traj[s: s + batch_time]                   # (batch_time, 2)
        batch_y0_list.append(traj[s])
        batch_y_list.append(segment)
    batch_y0 = torch.stack(batch_y0_list, dim=0)            # (batch_size, 2)
    batch_y = torch.stack(batch_y_list, dim=1)              # (batch_time, batch_size, 2)
    batch_t = torch.linspace(0, 1, batch_time)              # (batch_time,)
    return batch_y0, batch_t, batch_y

def get_random_batch(dataset, num_points, batch_size):
    """
    Campiona un batch di punti dalla traiettoria di ciascun sample.
    Per ogni sample viene scelto un insieme comune di 'num_points' indici (ordinati)
    da cui estrarre lo stato e il corrispondente istante temporale.
    
    Restituisce:
      - batch_y0: tensor (batch_size, 2) -> stato iniziale (primo dei punti campionati)
      - batch_t: tensor (num_points, batch_size) -> vettori dei tempi associati ai punti
      - batch_y: tensor (num_points, batch_size, 2) -> punti campionati della traiettoria
    """
    # Gestione dei casi in cui il dataset sia un Subset o l'intero dataset
    if isinstance(dataset, torch.utils.data.Subset):
        underlying_data = dataset.dataset.data
        valid_indices = dataset.indices
    else:
        underlying_data = dataset.data
        valid_indices = np.arange(len(underlying_data))
    
    # Supponiamo che tutte le traiettorie abbiano la stessa lunghezza
    traj_length = underlying_data[0]['trajectory'].shape[0]
    # Campiona un insieme comune di indici casuali (ordinati) dalla traiettoria
    indices = np.sort(np.random.choice(traj_length, num_points, replace=False))
    
    batch_y0_list = []
    batch_y_list = []
    batch_t_list = []
    for _ in range(batch_size):
        chosen_index = np.random.choice(valid_indices)
        sample = underlying_data[chosen_index]
        traj = sample['trajectory']                     # tensor shape (traj_length, 2)
        t_vec = sample['t']                             # tensor shape (traj_length,)
        batch_y0_list.append(traj[indices[0]])
        batch_y_list.append(traj[indices])
        batch_t_list.append(t_vec[indices])
    batch_y0 = torch.stack(batch_y0_list, dim=0)        # (batch_size, 2)
    batch_y = torch.stack(batch_y_list, dim=1)          # (num_points, batch_size, 2)
    batch_t = torch.stack(batch_t_list, dim=1)          # (num_points, batch_size)
    
    return batch_y0, batch_t, batch_y

#############################################
# Funzioni di simulazione
#############################################

def simulate_MLP(model, y0, t_span):
    """
    Simula iterativamente la traiettoria predetta da un modello MLP.
    Restituisce un tensore di shape (len(t_span), batch, 2).
    """
    model.eval()
    ys = [y0]
    current = y0
    for _ in range(len(t_span)-1):
        current = model(current)
        ys.append(current)
    return torch.stack(ys, dim=0)

def simulate_NODE(model, y0, t_span):
    model.eval()   
    sol = odeint(model.func, y0, t_span, method='rk4')
    return sol  # shape: (len(t_span), batch, 2)

def simulate_ODEmodel(model, y0, t_span):
    model.eval()
    with torch.enable_grad():
        sol = odeint(lambda t, y: model(t, y), y0, t_span, method='rk4')
    return sol

#############################################
# Funzioni di training/validazione/test
#############################################

def train_one_epoch(model, optimizer, criterion, dataset, num_iterations=50, model_type="NODE"):
    running_loss = 0.0
    total = 0
    for _ in range(num_iterations):
        batch_y0, batch_t, batch_y = get_batch_segment(dataset, batch_time, batch_size)
        batch_y0 = batch_y0.to(device).requires_grad_(True)
        batch_t = batch_t.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        if model_type == "MLP":
            pred_y = simulate_MLP(model, batch_y0, batch_t)
        elif model_type == "NODE":
            pred_y = simulate_NODE(model, batch_y0, batch_t)
        elif model_type in ["HNN", "LNN"]:
            pred_y = simulate_ODEmodel(model, batch_y0, batch_t)
        else:
            raise ValueError("Modello non riconosciuto per la simulazione")
        loss = criterion(pred_y, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * batch_size
        total += batch_size
    return running_loss / total

def test_evaluation(model, dataset, num_iterations=10, model_type="NODE"):
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
            if model_type == "MLP":
                pred_y = simulate_MLP(model, batch_y0, batch_t)
            elif model_type == "NODE":
                pred_y = simulate_NODE(model, batch_y0, batch_t)
            elif model_type in ["HNN", "LNN"]:
                pred_y = simulate_ODEmodel(model, batch_y0, batch_t)
            else:
                raise ValueError("Modello non riconosciuto")
            loss = criterion(pred_y, batch_y)
            running_loss += loss.item() * batch_size
            total += batch_size
    infer_time = time.time() - start_infer
    return running_loss / total, infer_time

#############################################
# Funzione per calcolare l'energia dalla traiettoria e errore sulla conservazione
#############################################

def compute_energy(states):
    return 0.5 * (states[:, 0]**2 + states[:, 1]**2)

def compute_energy_error(E):
    E_diff = torch.diff(E)                      # Derivata discreta dell'energia
    
    # Penalizzazione basata sulla variazione 
    error = torch.mean(torch.abs(E_diff))
    
    return error

#############################################
# Plot dei risultati: traiettorie e energia
#############################################

def plot_results(trained_models, num_samples, t_sim_steps):
    # Crea un vettore di tempi lungo l'intervallo [0, 100] con t_sim_steps punti
    t_span = torch.linspace(0, 100, t_sim_steps).to(device)
    raw_test = torch.load(os.path.join('./', 'conservative_test.pt'), weights_only=False)
    
    for _ in range(num_samples):
        idx = np.random.randint(0, len(raw_test))
        sample = raw_test[idx]
        y0 = sample['y0'].unsqueeze(0).to(device)       # (1, 2)
        ground_truth = sample['trajectory'].cpu()         # (num_steps, 2)
        
        # NOTA: per la ground truth usiamo la parte disponibile (che potrebbe essere più corta di t_sim_steps)
        gt_steps = ground_truth.shape[0]
        
        predictions = {}
        for name, model in trained_models.items():
            if name in ["MLP", "NODE"]:
                if name == "MLP":
                    pred = simulate_MLP(model, y0, t_span)
                else:
                    pred = simulate_NODE(model, y0, t_span)
            elif name in ["HNN", "LNN"]:
                pred = simulate_ODEmodel(model, y0, t_span)
            else:
                continue
            # Salviamo la predizione completa (t_sim_steps punti)
            predictions[name] = pred.squeeze(1).cpu()
        
        # Iniziamo a plottare
        plt.figure(figsize=(14,6))
        
        # --- Plot delle traiettorie nello spazio delle fasi ---
        plt.subplot(1,2,1)
        plt.plot(ground_truth[:,0], ground_truth[:,1], 'k-', linewidth=2, label="Ground Truth")
        plt.scatter(ground_truth[0,0], ground_truth[0,1], c='k', marker='o', s=50, label="Ground Truth Start")
        colors = {"MLP":'b', "NODE":'r', "HNN":'g', "LNN":'m'}
        linestyles = {"MLP":'--', "NODE":'-.', "HNN":'--', "LNN":'-.'}
        for name, pred in predictions.items():
            plt.plot(pred[:t_sim_steps, 0].detach().numpy(), pred[:t_sim_steps, 1].detach().numpy(), 
                     color=colors[name], linestyle=linestyles[name], linewidth=2, label=f"{name} Prediction")
            plt.scatter(pred[0,0].detach().numpy(), pred[0,1].detach().numpy(), color=colors[name], marker='o',
                         s=40, label=f"Start of {name} Prediction")
        plt.xlabel("x")
        plt.ylabel("v")
        plt.title("Traiettorie nello spazio delle fasi")
        plt.legend()
        plt.grid(True)
        
        # --- Plot dell'energia lungo la traiettoria ---
        plt.subplot(1,2,2)
        # Calcoliamo e plottiamo l'energia della ground truth
        gt_energy = compute_energy(ground_truth)
        plt.plot(np.linspace(0, 1, gt_steps), gt_energy.numpy(), 'k-', linewidth=2, label="Ground Truth Energy")
        for name, pred in predictions.items():
            if name in ["HNN", "LNN"]:
                # Assicuriamoci che l'input per get_energy sia su device
                energy_input = pred[:t_sim_steps].to(device)
                E = trained_models[name].get_energy(energy_input)
                # Convertiamo il risultato su CPU per il plotting e il calcolo dell'errore
                E = E.cpu()
            else:
                E = compute_energy(pred[:t_sim_steps])
            plt.plot(np.linspace(0, 1, t_sim_steps), E.detach().numpy(), 
                     color=colors[name], linestyle=linestyles[name], linewidth=2, label=f"{name} Energy")
        plt.xlabel("Tempo (normalizzato)")
        plt.ylabel("Energia")
        plt.ylim((-10, 10))
        plt.title("Conservazione dell'energia")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()
        
        # --- Calcolo e stampa dell'errore medio dell'energia (MAE) ---
        valid_steps = min(gt_steps, t_sim_steps)
        for name, pred in predictions.items():
            if name in ["HNN", "LNN"]:
                energy_input = pred[:t_sim_steps].to(device)
                E_pred = trained_models[name].get_energy(energy_input).cpu()
            else:
                E_pred = compute_energy(pred[:t_sim_steps])
            
            error = compute_energy_error(E_pred[:valid_steps])
            error = torch.nan_to_num(error, nan=0.0)
            print(f"Errore sulla conservazione Energia per {name}: {error.item():.6f}")
        print("-------------------------------------------")

#############################################
# Training Finale e Main
#############################################

def main():
    train_dataset, test_dataset = load_conservative_datasets()
    trained_models = {}

    # I modelli da testare: MLP, NODE, HNN, LNN
    model_names = ["LNN", "MLP", "NODE", "HNN"]
    model_classes = {"MLP": MLP, "NODE": NODE, "HNN": HNN, "LNN": LNN}
    # model_names = ["MLP", "HNN"]
    # model_classes = {"MLP": MLP, "HNN": HNN}

    final_results = {}
    for name in model_names:
        print(f"\n=== Allenamento finale del modello {name} sul training set completo ===")
        if name == "MLP":
            model = model_classes[name](input_dim=input_dim, output_dim=output_dim,
                                         hidden_dim=hidden_dim, num_layers=num_layers)
        elif name == "NODE":
            model = model_classes[name](input_dim=input_dim, hidden_dim=hidden_dim,
                                         num_layers=num_layers, integration_times=torch.linspace(0,1,batch_time).to(device))
        elif name in ["HNN", "LNN"]:
            model = model_classes[name](input_dim, hidden_dim, num_layers)
        else:
            raise ValueError("Modello non riconosciuto")
        
        model = model.to(device)
        criterion = nn.MSELoss()
        start_train = time.time()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        model.train()
        for epoch in range(num_epochs):
            epoch_loss = train_one_epoch(model, optimizer, criterion, train_dataset, num_iterations=100, model_type=name)
            print(f"Final Training Epoch {epoch+1}/{num_epochs} \t -> loss: {epoch_loss:.8f}")

        final_train_time = time.time() - start_train
        test_loss, infer_time = test_evaluation(model, test_dataset, num_iterations=50, model_type=name)
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\nModello {name} \t -> Test Loss: {test_loss:.6f}")
        print(f"Numero di parametri: {num_params}")
        print(f"Tempo di training finale: {final_train_time:.2f} sec")
        print(f"Tempo di inferenza sul test set: {infer_time:.2f} sec")

        final_results[name] = {"test_loss": test_loss,
                                "num_params": num_params,
                                "final_train_time": final_train_time,
                                "infer_time": infer_time}
        
        trained_models[name] = model
        
    print("\n===== RISULTATI FINALI =====")
    for name, res in final_results.items():
        print(f"Modello: {name}")
        print(f"  Test Loss: {res['test_loss']:.6f}")
        print(f"  Parametri: {res['num_params']}")
        print(f"  Tempo training finale: {res['final_train_time']:.2f} sec")
        print(f"  Tempo inferenza: {res['infer_time']:.2f} sec")
        print("-------------------------------------------")
    
    return trained_models, test_dataset

if __name__ == '__main__':
    trained_models, test_dataset = main()
    plot_results(trained_models, num_samples=3, t_sim_steps=1000)