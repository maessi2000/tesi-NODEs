import torch
import torch.nn as nn
from torchdiffeq import odeint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###############################################################################
# MLP: Multi-Layer Perceptron
###############################################################################

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, num_layers):
        super(MLP, self).__init__()
        
        layers = []
        # Primo layer
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.Tanh())
        
        # Layers intermedi
        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        
        # Layer finale
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

###############################################################################
# Neural ODE (NODE base)
###############################################################################

class ODEFunc(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ODEFunc, self).__init__()
        self.net = MLP(input_dim, input_dim, hidden_dim, num_layers)
    
    def forward(self, t, x):
        return self.net(x)

class NODE(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, integration_times=torch.tensor([0.0, 1.0])):
        super(NODE, self).__init__()
        self.func = ODEFunc(input_dim, hidden_dim, num_layers)
        self.integration_times = integration_times

    def forward(self, x):
        # x ha forma (batch, input_dim)
        # odeint restituisce una soluzione di forma (len(integration_times), batch, input_dim)
        sol = odeint(self.func, x, self.integration_times, method='rk4')
        # Restituiamo lo stato finale (al tempo t_final)
        return sol[-1]

###############################################################################
# NODEClassifier: combinazione di encoder, NODE e decoder per classificazione
###############################################################################

class NODEClassifier(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_dim, num_layers, integration_times, output_dim):
        super(NODEClassifier, self).__init__()
        # Encoder: mappa da input_dim allo spazio latente
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, latent_dim),
            nn.Tanh()
        )
        # Blocco NODE: opera in uno spazio latente di dimensione "latent_dim"
        self.node_block = NODE(input_dim=latent_dim, hidden_dim=hidden_dim, num_layers=num_layers, integration_times=integration_times)
        # Decoder: mappa dallo spazio latente all'output di classificazione
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, output_dim)
        )
    
    def forward(self, x):
        # x: (batch, input_dim)
        x_enc = self.encoder(x)       # (batch, latent_dim)
        x_node = self.node_block(x_enc)  # Evoluzione nello spazio latente (stessa dimensione)
        logits = self.decoder(x_node)    # (batch, output_dim)
        return logits

###############################################################################
# Hamiltonian Neural Network (HNN)
###############################################################################
    
class HNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(HNN, self).__init__()
        # La rete apprende l'Hamiltoniana, funzione scalare
        self.hamiltonian_net = MLP(input_dim=input_dim, output_dim=1,
                                   hidden_dim=hidden_dim, num_layers=num_layers)
        # Costruisce la matrice symplettica J (si assume input_dim pari, cioè 2n)
        self.J = self._build_symplectic_matrix(input_dim)

    def forward(self, t, x):
        # Assicuriamo il tracciamento del gradiente
        x = x.clone().detach().requires_grad_(True)
        # Calcola l'Hamiltoniana (scala) per ogni sample
        H = self.hamiltonian_net(x)
        # Calcola il gradiente ∇ₓH(x)
        gradH = torch.autograd.grad(H.sum(), x, create_graph=True)[0]
        # Applica la struttura symplettica: dx/dt = Jᵀ·gradH (ovvero, per ogni sample, J gradH)
        return gradH @ self.J.T

    def _build_symplectic_matrix(self, dim):
        n = dim // 2
        J = torch.zeros(dim, dim)
        for i in range(n):
            J[i, i+n] = 1
            J[i+n, i] = -1
        return J.to(device)
    
    def get_energy(self, x):
        H = self.hamiltonian_net(x)
        return H

###############################################################################
# Lagrangian Neural Network (LNN)
###############################################################################

class LNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(LNN, self).__init__()
        # La rete apprende la Lagrangiana (funzione scalare)
        self.lagrangian_net = MLP(input_dim=input_dim, output_dim=1,
                                  hidden_dim=hidden_dim, num_layers=num_layers)
        self.eps = 0.1

    def forward(self, t, x):
        # Assicuriamo il tracciamento dei gradienti
        x = x.requires_grad_(True)  # (batch, 2)

        # Definiamo una funzione che, per un singolo sample (di forma (2,)), restituisce la Lagrangiana scalare.
        # Nota: lag_fn riceve un tensore 1D e lo trasforma in (1,2) per compatibilità con la rete.
        def lag_fn(x_sample):
            # x_sample: shape (2,)
            return self.lagrangian_net(x_sample.unsqueeze(0))[0, 0]

        grad_fn = torch.func.grad(lag_fn)
        grad_all = torch.vmap(grad_fn)(x)
        dL_dq = grad_all[:, 0:1]  # shape (batch, 1)
        dL_dv = grad_all[:, 1:2]  # shape (batch, 1)

        hess_fn = torch.func.hessian(lag_fn)
        H_full = torch.vmap(hess_fn)(x)
        # Estraiamo gli elementi necessari:
        d2L_dqv = H_full[:, 1, 0].unsqueeze(-1)  # shape (batch, 1)
        d2L_dv2 = H_full[:, 1, 1].unsqueeze(-1)  # shape (batch, 1)

        # Estraiamo la velocità v dallo stato
        v = x[:, 1:2]  # shape (batch, 1)

        # Risoluzione delle equazioni di Eulero-Lagrange invertite:
        a = (dL_dq - d2L_dqv * v) / (d2L_dv2 + self.eps)  # shape (batch, 1)

        # Lo stato incrementale è la coppia (v, a)
        return torch.cat([v, a], dim=1)  # (batch, 2)

    def get_energy(self, x):
        x = x.requires_grad_(True)
        L = self.lagrangian_net(x)
        
        # Momento generalizzato p = dL/dv
        grad_x = torch.autograd.grad(L.sum(), x, create_graph=True, allow_unused=True)[0]
        p = grad_x[:, 1:2]

        # Energia hamiltoniana H = p*v - L
        E = p * x[:, 1:2] - L
        return E