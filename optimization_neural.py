"""
Module d'optimisation avec réseau de neurones pour apprendre l'interpolation
entre deux distributions au lieu d'un simple vecteur alpha.

Au lieu d'apprendre alpha et beta tels que: interpolation = alpha * x + (1-alpha) * y
On apprend un réseau f^theta(x,y) qui produit pour chaque gène i un résultat entre x_i et y_i
"""

import torch
import torch.nn as nn
import numpy as np
from geomloss import SamplesLoss
from joblib import Parallel, delayed


class InterpolationNetwork(nn.Module):
    """
    Réseau de neurones qui apprend à interpoler entre x et y.
    
    Architecture:
    - Input: concaténation de x et y (2 * n_genes)
    - Hidden layers avec ReLU
    - Output: coefficients alpha par gène (n_genes) passés par sigmoid
    - Résultat final: alpha * x + (1 - alpha) * y (contraint entre x et y)
    """
    
    def __init__(self, n_genes, hidden_dims=[64, 32]):
        """
        Parameters:
        -----------
        n_genes : int
            Nombre de gènes
        hidden_dims : list of int
            Dimensions des couches cachées
        """
        super(InterpolationNetwork, self).__init__()
        
        self.n_genes = n_genes
        
        # Construction du réseau
        layers = []
        input_dim = 2 * n_genes  # Concaténation de x et y
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.1))  # Régularisation
            input_dim = hidden_dim
        
        # Couche de sortie: un coefficient alpha par gène
        layers.append(nn.Linear(input_dim, n_genes))
        layers.append(nn.Sigmoid())  # alpha entre 0 et 1
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x, y):
        """
        Forward pass du réseau.
        
        Parameters:
        -----------
        x : torch.Tensor, shape (batch_size, n_genes)
            Points de la distribution initiale
        y : torch.Tensor, shape (batch_size, n_genes)
            Points de la distribution finale
            
        Returns:
        --------
        interpolated : torch.Tensor, shape (batch_size, n_genes)
            Interpolation contrainte entre x et y
        alpha : torch.Tensor, shape (batch_size, n_genes)
            Coefficients d'interpolation appris
        """
        # Concaténation de x et y
        xy = torch.cat([x, y], dim=1)  # (batch_size, 2*n_genes)
        
        # Calcul des alphas via le réseau
        alpha = self.network(xy)  # (batch_size, n_genes)
        
        # Interpolation contrainte: alpha * x + (1 - alpha) * y
        interpolated = alpha * x + (1 - alpha) * y
        
        return interpolated, alpha


def get_indices_coupling(data_t1, data_t3, coupling, n_samples):
    """
    Échantillonne des couples (x, y) selon le couplage.
    """
    n_cells_t1 = data_t1.shape[0]
    n_cells_t3 = data_t3.shape[0]
    i_indices = []
    j_indices = []
    np.random.seed(42)
    
    for index_x in range(n_cells_t1):
        probs = coupling[index_x, :]
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(n_cells_t3) / n_cells_t3
        sampled_indices = np.random.choice(np.arange(n_cells_t3), size=n_samples, p=probs)
        for index_y in sampled_indices:
            i_indices.append(index_x)
            j_indices.append(index_y)

    return data_t1[i_indices], data_t3[j_indices]


def optimize_neural_network(
    data_t1, data_t3, B_ref, rho_ref,
    n_samples=10000, n_iterations=1000, lr=0.001, blur=0.01,
    hidden_dims=[64, 32], verbose=True, batch_size=None
):
    """
    Optimise un réseau de neurones pour l'interpolation de McCann.
    
    Parameters:
    -----------
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Données au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Données au temps t3
    B_ref : array-like, shape (n_cells_t1, n_cells_t3)
        Matrice de couplage de référence
    rho_ref : array-like, shape (n_ref_cells, n_genes)
        Distribution de référence à approximer
    n_samples : int
        Nombre de couples à échantillonner du couplage
    n_iterations : int
        Nombre d'itérations d'entraînement
    lr : float
        Learning rate
    blur : float
        Paramètre de régularisation Sinkhorn
    hidden_dims : list of int
        Dimensions des couches cachées du réseau
    verbose : bool
        Affichage détaillé
    batch_size : int or None
        Si spécifié, utilise du mini-batch training
        
    Returns:
    --------
    network : InterpolationNetwork
        Réseau entraîné
    training_history : dict
        Historique de l'entraînement (losses)
    """
    n_genes = data_t1.shape[1]
    
    # Conversion en tensors PyTorch
    data_t1_torch = torch.tensor(data_t1, dtype=torch.float32)
    data_t3_torch = torch.tensor(data_t3, dtype=torch.float32)
    rho_ref_torch = torch.tensor(rho_ref, dtype=torch.float32)
    
    # Échantillonnage des couples selon le couplage
    x_samples, y_samples = get_indices_coupling(data_t1_torch, data_t3_torch, B_ref, n_samples)
    x_samples = torch.tensor(x_samples, dtype=torch.float32)
    y_samples = torch.tensor(y_samples, dtype=torch.float32)
    
    if verbose:
        print(f"\nEntraînement du réseau de neurones")
        print(f"  Architecture: {2*n_genes} -> {' -> '.join(map(str, hidden_dims))} -> {n_genes}")
        print(f"  Échantillons: {x_samples.shape[0]}")
        print(f"  Référence: {rho_ref_torch.shape[0]} cellules")
    
    # Initialisation du réseau
    network = InterpolationNetwork(n_genes, hidden_dims=hidden_dims)
    optimizer = torch.optim.Adam(network.parameters(), lr=lr)
    
    # Loss de Wasserstein multidimensionnelle
    loss_fn = SamplesLoss("sinkhorn", blur=blur, diameter=10.0, scaling=0.95, debias=True)
    
    training_history = {'losses': [], 'alpha_means': [], 'alpha_stds': []}
    
    # Entraînement
    for iteration in range(n_iterations):
        # Mini-batch training si spécifié
        if batch_size is not None and batch_size < x_samples.shape[0]:
            indices = torch.randperm(x_samples.shape[0])[:batch_size]
            x_batch = x_samples[indices]
            y_batch = y_samples[indices]
        else:
            x_batch = x_samples
            y_batch = y_samples
        
        optimizer.zero_grad()
        
        # Forward pass
        interpolated, alpha = network(x_batch, y_batch)
        
        # Calcul de la distance de Wasserstein
        loss = loss_fn(interpolated, rho_ref_torch)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
        
        # Sauvegarde de l'historique
        training_history['losses'].append(loss.item())
        training_history['alpha_means'].append(alpha.mean().item())
        training_history['alpha_stds'].append(alpha.std().item())
        
        if verbose and (iteration + 1) % max(1, n_iterations // 10) == 0:
            print(f"  Iter {iteration+1}/{n_iterations} | "
                  f"Loss: {loss.item():.6f} | "
                  f"Alpha mean: {alpha.mean().item():.4f} ± {alpha.std().item():.4f}")
    
    if verbose:
        print(f"\nEntraînement terminé!")
        print(f"  Loss finale: {training_history['losses'][-1]:.6f}")
        print(f"  Alpha final: {training_history['alpha_means'][-1]:.4f} ± {training_history['alpha_stds'][-1]:.4f}")
    
    return network, training_history


def neural_mccann_interpolation(network, data_t1, data_t3, coupling, n_samples=10000):
    """
    Applique l'interpolation de McCann avec le réseau de neurones entraîné.
    
    Parameters:
    -----------
    network : InterpolationNetwork
        Réseau entraîné
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Données au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Données au temps t3
    coupling : array-like, shape (n_cells_t1, n_cells_t3)
        Matrice de couplage
    n_samples : int
        Nombre de couples à échantillonner
        
    Returns:
    --------
    interpolated : array, shape (n_samples, n_genes)
        Distribution interpolée
    alpha_samples : array, shape (n_samples, n_genes)
        Coefficients alpha utilisés pour chaque échantillon
    """
    # Échantillonnage des couples
    x_samples, y_samples = get_indices_coupling(data_t1, data_t3, coupling, n_samples)
    x_samples = torch.tensor(x_samples, dtype=torch.float32)
    y_samples = torch.tensor(y_samples, dtype=torch.float32)
    
    # Forward pass sans gradient
    network.eval()
    with torch.no_grad():
        interpolated, alpha = network(x_samples, y_samples)
    
    return interpolated.numpy(), alpha.numpy()


def mccann_interpolation(data_t1, data_t3, coupling, alpha, beta, n_samples=10000):
    """
    Version classique de l'interpolation de McCann (gardée pour compatibilité).
    
    Parameters:
    -----------
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Données au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Données au temps t3
    coupling : array-like
        Matrice de couplage
    alpha : array-like, shape (n_genes,) or scalar
        Paramètres alpha pour chaque gène
    beta : array-like, shape (n_genes,) or scalar
        Paramètres beta pour chaque gène
    n_samples : int
        Nombre de couples à échantillonner
        
    Returns:
    --------
    interpolated : array, shape (n_samples, n_genes)
        Distribution interpolée
    """
    x_samples, y_samples = get_indices_coupling(data_t1, data_t3, coupling, n_samples)
    
    # Conversion en numpy si nécessaire
    if isinstance(alpha, (int, float)):
        alpha = np.array([alpha])
    if isinstance(beta, (int, float)):
        beta = np.array([beta])
    
    # Interpolation: alpha * x + beta * y
    interpolated = alpha[np.newaxis, :] * x_samples + beta[np.newaxis, :] * y_samples
    
    return interpolated


# ========== Fonctions de compatibilité avec l'ancienne API ==========

def optimize_alpha_beta_complete(
    data_t1, data_t3, B_ref, rho_ref,
    n_samples=10000, n_iterations=1000, lr=0.01, blur=0.01,
    verbose=True, mode='neural', constrain=True, n_jobs=-1,
    hidden_dims=[64, 32], batch_size=None
):
    """
    Fonction wrapper pour compatibilité avec l'ancien code.
    
    Maintenant supporte un nouveau mode 'neural' en plus de 'per_gene' et 'global'.
    
    Parameters:
    -----------
    mode : str
        - 'neural': utilise un réseau de neurones (NOUVEAU!)
        - 'per_gene': un couple (alpha, beta) par gène optimisé indépendamment (ancien)
        - 'global': un couple (alpha, beta) par gène optimisé conjointement (ancien)
    hidden_dims : list of int
        Dimensions des couches cachées (seulement pour mode='neural')
    batch_size : int or None
        Taille des mini-batches (seulement pour mode='neural')
    
    Returns:
    --------
    Si mode='neural':
        network : InterpolationNetwork
            Réseau entraîné
        training_history : dict
            Historique de l'entraînement
    Sinon:
        alpha_np : array, shape (n_genes,)
        beta_np : array, shape (n_genes,)
    """
    if mode == 'neural':
        if verbose:
            print(f"\nMode: NEURAL NETWORK - Apprentissage d'un réseau f^θ(x,y)")
        
        network, history = optimize_neural_network(
            data_t1=data_t1,
            data_t3=data_t3,
            B_ref=B_ref,
            rho_ref=rho_ref,
            n_samples=n_samples,
            n_iterations=n_iterations,
            lr=lr,
            blur=blur,
            hidden_dims=hidden_dims,
            verbose=verbose,
            batch_size=batch_size
        )
        return network, history
    
    elif mode in ['per_gene', 'global']:
        # Ancienne implémentation (importée du fichier original)
        from optimization import optimize_alpha_beta_complete as old_optimize
        return old_optimize(
            data_t1=data_t1,
            data_t3=data_t3,
            B_ref=B_ref,
            rho_ref=rho_ref,
            n_samples=n_samples,
            n_iterations=n_iterations,
            lr=lr,
            blur=blur,
            verbose=verbose,
            mode=mode,
            constrain=constrain,
            n_jobs=n_jobs
        )
    else:
        raise ValueError(f"Mode inconnu: {mode}. Utilisez 'neural', 'per_gene' ou 'global'.")
