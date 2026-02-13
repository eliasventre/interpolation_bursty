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
    for index_x in range(n_cells_t1):
        probs = coupling[index_x, :]
        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs = np.ones(n_cells_t3) / n_cells_t3
        sampled_indices = np.random.choice(n_cells_t3, size=n_samples, p=probs)
        # sampled_indices = np.argsort(probs)[::-1][:n_samples]
        for index_y in sampled_indices:
            i_indices.append(index_x)
            j_indices.append(index_y)

    return data_t1[i_indices], data_t3[j_indices]

def get_indices_coupling_full(data_t1, data_t3, coupling, n_samples_full):

    # Matrice de probabilités
    coupling_flat = coupling.flatten()
    n_samples = int(coupling.shape[0] * n_samples_full)
    probs = coupling_flat / coupling_flat.sum()
    sampled_flat_indices = np.random.choice(
        len(coupling_flat), size=n_samples, p=probs, replace=True
    )
    i_indices = sampled_flat_indices // coupling.shape[1]
    j_indices = sampled_flat_indices % coupling.shape[1]

    # Extraction des cellules
    x_samples = data_t1[i_indices]
    y_samples = data_t3[j_indices]

    return x_samples, y_samples


def optimize_single_gene(gene_idx, x_gene, y_gene, rho_gene, n_iterations, lr, blur, constrain, verbose_gene=False):
    """
    Optimise alpha et beta pour un seul gène.
    
    Parameters:
    -----------
    gene_idx : int
        Indice du gène
    x_gene : torch.Tensor, shape (n_samples, 1)
        Échantillons de t1 pour ce gène
    y_gene : torch.Tensor, shape (n_samples, 1)
        Échantillons de t3 pour ce gène
    rho_gene : torch.Tensor, shape (n_ref_cells, 1)
        Distribution de référence pour ce gène
    n_iterations : int
        Nombre d'itérations
    lr : float
        Learning rate
    blur : float
        Paramètre de régularisation Sinkhorn
    constrain : bool
        Si True, contraint alpha + beta = 1
    verbose_gene : bool
        Affichage détaillé pour ce gène (désactivé par défaut en parallèle)
    
    Returns:
    --------
    gene_idx : int
        Indice du gène (pour réordonner si besoin)
    alpha_final : float
        Valeur optimale de alpha
    beta_final : float
        Valeur optimale de beta
    """
    # Initialisation des paramètres pour ce gène
    alpha_raw = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
    if not constrain:
        beta_raw = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
        optimizer = torch.optim.Adam([alpha_raw, beta_raw], lr=lr)
    else:
        optimizer = torch.optim.Adam([alpha_raw], lr=lr)
    
    # Loss de Wasserstein 1D
    loss_fn = SamplesLoss("sinkhorn", blur=blur, diameter=10.0, scaling=0.95, debias=True)
    
    for iteration in range(n_iterations):
        optimizer.zero_grad()
        
        # Application de la contrainte
        if constrain:
            alpha = torch.sigmoid(alpha_raw)
            beta = 1 - alpha
        else:
            alpha = torch.sigmoid(alpha_raw)
            beta = torch.sigmoid(beta_raw)
        
        # Construction de la distribution interpolée pour ce gène
        interpolated = alpha * x_gene + beta * y_gene
        
        # Calcul de la distance de Wasserstein 1D
        loss = loss_fn(interpolated, rho_gene)
        
        # Backpropagation
        loss.backward()
        optimizer.step()
    
    # Extraction des valeurs finales pour ce gène
    if constrain:
        alpha_final = torch.sigmoid(alpha_raw).item()
        beta_final = 1 - alpha_final
    else:
        alpha_final = torch.sigmoid(alpha_raw).item()
        beta_final = torch.sigmoid(beta_raw).item()
    
    if verbose_gene:
        print(f"alpha: {alpha_final:.4f}, beta: {beta_final:.4f}")
    
    return gene_idx, alpha_final, beta_final


def optimize_alpha_beta_complete(
    data_t1, data_t3, B_ref, rho_ref, 
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
    
    if mode == 'per_gene':
        # Optimisation gène par gène avec Sinkhorn 1D en parallèle
        if verbose:
            print(f"Mode: PER_GENE - Optimisation indépendante pour {n_genes} gènes (Sinkhorn 1D)")
            if constrain:
                print("Contraintes: alpha + beta = 1")
            else:
                print("Pas de contraintes sur alpha et beta")
            print(f"Parallélisation avec n_jobs={n_jobs}")
        
        gene_data = []
        for gene_idx in range(n_genes):
            x_gene = x_samples[:, gene_idx].reshape(-1, 1)
            y_gene = y_samples[:, gene_idx].reshape(-1, 1)
            rho_gene = rho_ref_torch[:, gene_idx].reshape(-1, 1)
            gene_data.append((gene_idx, x_gene, y_gene, rho_gene))
        
        results = Parallel(n_jobs=n_jobs, verbose=10 if verbose else 0)(
            delayed(optimize_single_gene)(
                gene_idx, x_gene, y_gene, rho_gene, 
                n_iterations, lr, blur, constrain, verbose_gene=False
            )
            for gene_idx, x_gene, y_gene, rho_gene in gene_data
        )
        
        alpha_np = np.array([r[1] for r in results])
        beta_np  = np.array([r[2] for r in results])
        
        if verbose:
            print(f"\nOptimisation terminée pour tous les gènes (PER_GENE).")
            print(f"Alpha - min: {alpha_np.min():.4f}, max: {alpha_np.max():.4f}, mean: {alpha_np.mean():.4f}")
            print(f"Beta  - min: {beta_np.min():.4f}, max: {beta_np.max():.4f}, mean: {beta_np.mean():.4f}")
    
    elif mode == 'global':
        # Optimisation conjointe de (alpha_g, beta_g) pour tous les gènes
        if verbose:
            print(f"Mode: GLOBAL - Un alpha_g (et beta_g) par gène, optimisés conjointement (Sinkhorn multivarié)")
            if constrain:
                print("Contraintes: alpha_g + beta_g = 1 pour chaque gène")
            else:
                print("Pas de contraintes sur alpha et beta")
        
        # Paramètres bruts par gène : shape (n_genes,)
        alpha_raw = torch.nn.Parameter(torch.zeros(n_genes, dtype=torch.float32))
        if not constrain:
            beta_raw = torch.nn.Parameter(torch.zeros(n_genes, dtype=torch.float32))
            optimizer = torch.optim.Adam([alpha_raw, beta_raw], lr=lr)
        else:
            optimizer = torch.optim.Adam([alpha_raw], lr=lr)
        
        # Loss de Wasserstein multidimensionnelle (tous les gènes ensemble)
        loss_fn = SamplesLoss("sinkhorn", blur=blur, diameter=10.0, scaling=0.95, debias=True)
        
        for iteration in range(n_iterations):
            optimizer.zero_grad()
            
            if constrain:
                alpha = torch.sigmoid(alpha_raw)              # (n_genes,)
                beta  = 1.0 - alpha
            else:
                alpha = torch.sigmoid(alpha_raw)
                beta  = torch.sigmoid(beta_raw)
            
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
            alpha_final = torch.sigmoid(alpha_raw).detach().cpu().numpy()  # (n_genes,)
            beta_final  = 1.0 - alpha_final
        else:
            alpha_final = torch.sigmoid(alpha_raw).detach().cpu().numpy()
            beta_final  = torch.sigmoid(beta_raw).detach().cpu().numpy()
        
        alpha_np = alpha_final
        beta_np  = beta_final
        
        if verbose:
            print(f"\n[GLOBAL] Optimisation terminée.")
            print(f"Alpha - min: {alpha_np.min():.4f}, max: {alpha_np.max():.4f}, mean: {alpha_np.mean():.4f}")
            print(f"Beta  - min: {beta_np.min():.4f}, max: {beta_np.max():.4f}, mean: {beta_np.mean():.4f}")
    
    else:
        raise ValueError(f"Mode inconnu: {mode}. Utilisez 'per_gene' ou 'global'.")
    
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
