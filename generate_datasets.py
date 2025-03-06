import os
import torch
from torch.utils.data import random_split, ConcatDataset
import torchvision
import torchvision.transforms as transforms
from torchdiffeq import odeint
import random
import matplotlib.pyplot as plt

def generate_mnist_dataset():
    transform = transforms.ToTensor()
    # Scarica i dataset originali
    train_dataset = torchvision.datasets.MNIST(root='./', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.MNIST(root='./', train=False, download=True, transform=transform)
    
    # Concateniamo i due dataset
    full_dataset = ConcatDataset([train_dataset, test_dataset])
    total = len(full_dataset)
    train_size = int(0.8 * total)
    test_size = total - train_size

    # Suddividiamo in training e testing
    train_subset, test_subset = random_split(full_dataset, [train_size, test_size])

    # Salvataggio su file (usiamo torch.save, che preserva anche le strutture dei dataset)
    torch.save(train_subset, os.path.join('./', 'mnist_train.pt'))
    torch.save(test_subset, os.path.join('./', 'mnist_test.pt'))
    print(f"MNIST dataset salvato: {train_size} campioni per il training e {test_size} per il testing.")

def generate_fitzhugh_dataset(num_samples_train=500, num_samples_test=125, T=20.0, steps=500, noise_var=0.01):  
    def fitzhugh_ode(t, y):
        # y è un tensore di shape (2,)
        v, w = y[0], y[1]
        I = 0.5  # corrente esterna
        dv = v - (v**3)/3 - w + I
        # Parametri tipici per FitzHugh-Nagumo
        dw = 0.08 * (v + 0.7 - 0.8 * w)
        return torch.stack([dv, dw])

    # Vettore dei tempi per la simulazione
    t_span = torch.linspace(0., T, steps)

    def generate_sample():
        # Condizione iniziale: campionata uniformemente nell'intervallo [-2, 2] per entrambi i componenti
        y0 = torch.empty(2).uniform_(-2, 2)
        traj = odeint(fitzhugh_ode, y0, t_span, method='rk4')
        # Generazione del rumore con torch.empty().normal_()
        noise = torch.empty_like(traj).normal_(mean=0, std=torch.tensor(noise_var))
        traj_noisy = traj + noise
        return {'y0': y0, 't': t_span, 'trajectory': traj_noisy, 'noise_var': noise_var}

    # Generazione del dataset di training
    train_samples = [generate_sample() for _ in range(num_samples_train)]

    # Generazione del dataset di testing
    test_samples = [generate_sample() for _ in range(num_samples_test)]

    # Salvataggio su file
    torch.save(train_samples, os.path.join('./', 'fitzhugh_train.pt'))
    torch.save(test_samples, os.path.join('./', 'fitzhugh_test.pt'))
    print(f"Dataset FitzHugh-Nagumo salvato: {num_samples_train} campioni per il training e {num_samples_test} per il testing.")

def generate_conservative_dataset(num_samples_train=500, num_samples_test=125, T=10.0, steps=500, noise_var=0.01):
    def harmonic_oscillator(t, y):
        x, v = y[0], y[1]
        dx = v
        dv = -x
        return torch.stack([dx, dv])

    t_span = torch.linspace(0., T, steps)

    def generate_sample():
        y0 = torch.empty(2).uniform_(-2, 2)
        traj = odeint(harmonic_oscillator, y0, t_span, method='rk4')
        noise = torch.empty_like(traj).normal_(mean=0, std=torch.tensor(noise_var))
        traj_noisy = traj + noise
        return {'y0': y0, 't': t_span, 'trajectory': traj_noisy}

    train_samples = [generate_sample() for _ in range(num_samples_train)]
    test_samples = [generate_sample() for _ in range(num_samples_test)]

    torch.save(train_samples, os.path.join('./', 'conservative_train.pt'))
    torch.save(test_samples, os.path.join('./', 'conservative_test.pt'))
    print(f"Dataset sistema conservativo salvato: {num_samples_train} campioni per il training e {num_samples_test} per il testing.")

def plot_trajectories(dataset_path, num_trajectories=1):
    # Caricamento del dataset
    dataset = torch.load(dataset_path, weights_only=True)

    # Estrazione casuale di num_trajectories traiettorie
    sampled_data = random.sample(dataset, num_trajectories)

    plt.figure(figsize=(10, 5))

    for i, sample in enumerate(sampled_data):
        t = sample['t']
        trajectory = sample['trajectory']

        # Plot delle due variabili (v e w) nel tempo
        plt.subplot(1, 2, 1)
        plt.plot(t, trajectory[:, 0], label=f'Traj {i+1}')
        plt.xlabel("Time")
        plt.ylabel("var1")
        plt.title("Variable 1 over time")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(t, trajectory[:, 1], label=f'Traj {i+1}')
        plt.xlabel("Time")
        plt.ylabel("var2")
        plt.title("Variable 2 over time")
        plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    print("Generazione dei dataset in corso...")
    generate_mnist_dataset()
    generate_fitzhugh_dataset()
    generate_conservative_dataset()
    print("Tutti i dataset sono stati generati e salvati correttamente.")
    # plot_trajectories('./conservative_test.pt', 3)