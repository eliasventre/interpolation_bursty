"""
Module d'optimisation pour trouver les paramètres alpha et beta optimaux
pour l'interpolation de McCann entre deux distributions.
"""

import torch
import numpy as np
from geomloss import SamplesLoss
from joblib import Parallel, delayed

def get_indices_coupling(data_t1, data_t3, coupling, n_samples):

    # On échantillonne n_samples couples
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


def optimize_alpha_beta_complete(
    data_t1, data_t3, B_ref, rho_ref, alpha_init, beta_init,
    n_samples=10000, n_iterations=1000, lr=0.01, blur=.01,
    verbose=True, mode='per_gene', constrain=True, n_jobs=-1
):
    """
    ...
    mode : str, 'per_gene' ou 'global'
        - 'per_gene': un couple (alpha, beta) différent par gène optimisé indépendamment (Sinkhorn 1D)
        - 'global': un couple (alpha_g, beta_g) par gène, optimisé conjointement
                    en minimisant une divergence de Sinkhorn multivariée (tous les gènes d'un coup)
    ...
    """
    n_genes = data_t1.shape[1]
    
    # Conversion en tensors PyTorch
    data_t1_torch = torch.tensor(data_t1, dtype=torch.float32)
    data_t3_torch = torch.tensor(data_t3, dtype=torch.float32)
    rho_ref_torch = torch.tensor(rho_ref, dtype=torch.float32)  # (n_ref_cells, n_genes)
    
    # On échantillonne n_samples couples
    # x_samples, y_samples : (n_samples, n_genes)
    x_samples, y_samples = get_indices_coupling(data_t1_torch, data_t3_torch, B_ref, n_samples)
    
    # Optimisation conjointe de (alpha_g, beta_g) pour tous les gènes
    if verbose:
        print(f"Mode: GLOBAL - Un alpha_g (et beta_g) par gène, optimisés conjointement (Sinkhorn multivarié)")
        if constrain:
            print("Contraintes: alpha_g + beta_g = 1 pour chaque gène")
        else:
            print("Pas de contraintes sur alpha et beta")
    
    # Paramètres bruts par gène : shape (n_genes,)
    alpha_raw = torch.nn.Parameter(torch.tensor(alpha_init, dtype=torch.float32))
    if not constrain:
        beta_raw = torch.nn.Parameter(torch.tensor(beta_init, dtype=torch.float32))
        optimizer = torch.optim.Adam([alpha_raw, beta_raw], lr=lr)
    else:
        optimizer = torch.optim.Adam([alpha_raw], lr=lr)
    
    # Loss de Wasserstein multidimensionnelle (tous les gènes ensemble)
    loss_fn = SamplesLoss("sinkhorn", blur=blur, scaling=0.95, debias=True)
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        if constrain:
            alpha = torch.clip(alpha_raw, 0, 1)              # (n_genes,)
            beta  = 1.0 - alpha
        else:
            alpha = torch.clip(alpha_raw, 0, 1)
            beta  = torch.clip(beta_raw, 0, 1)
        
        # Broadcasting sur les samples : (n_samples, n_genes)
        # x_samples, y_samples : (n_samples, n_genes)
        interpolated = alpha.unsqueeze(0) * x_samples + beta.unsqueeze(0) * y_samples
        
        loss = loss_fn(interpolated, rho_ref_torch)
        loss.backward()
        optimizer.step()
        
        if verbose and (iteration + 1) % 50 == 0:
            print(f"[GLOBAL] Iter {iteration+1}/{n_iterations}, Loss: {loss.item():.6f}, "
                f"alpha_mean: {alpha.mean().item():.4f}, beta_mean: {beta.mean().item():.4f}")
    
    # Extraction des valeurs finales
    if constrain:
        alpha_final = torch.clip(alpha_raw, 0, 1).detach().cpu().numpy()  # (n_genes,)
        beta_final  = 1.0 - alpha_final
    else:
        alpha_final = torch.clip(alpha_raw, 0, 1).detach().cpu().numpy()
        beta_final  = torch.clip(beta_raw, 0, 1).detach().cpu().numpy()
    
    alpha_np = alpha_final
    beta_np  = beta_final
    
    if verbose:
        print(f"\n[GLOBAL] Optimisation terminée.")
        print(f"Alpha - min: {alpha_np.min():.4f}, max: {alpha_np.max():.4f}, mean: {alpha_np.mean():.4f}")
        print(f"Beta  - min: {beta_np.min():.4f}, max: {beta_np.max():.4f}, mean: {beta_np.mean():.4f}")

    
    return alpha_np, beta_np



def mccann_interpolation(data_t1, data_t3, coupling, alpha, beta, n_samples=10000):
    """
    Applique l'interpolation de McCann avec les paramètres alpha et beta donnés.
    
    Parameters:
    -----------
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Données au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Données au temps t3
    coupling_indices : array-like
        Indices du couplage ou matrice de probabilités
    alpha : array-like, shape (n_genes,)
        Paramètres alpha pour chaque gène
    beta : array-like, shape (n_genes,)
        Paramètres beta pour chaque gène
    n_samples : int
        Nombre de couples à échantillonner
        
    Returns:
    --------
    interpolated : array, shape (n_samples, n_genes)
        Distribution interpolée au temps intermédiaire
    """
    
    x_samples, y_samples = get_indices_coupling(data_t1, data_t3, coupling, n_samples)
    
    # Interpolation: alpha * x + beta * y
    # Broadcasting pour appliquer alpha et beta par gène
    interpolated = alpha[np.newaxis, :] * x_samples + beta[np.newaxis, :] * y_samples
    
    return interpolated
