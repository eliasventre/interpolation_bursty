"""
Module de transport optimal entropique pour calculer le couplage OT classique
et l'interpolation de McCann associée.
"""

import numpy as np
import ot


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
        sampled_indices = np.random.choice(np.arange(n_cells_t3), size=n_samples, p=probs)
        for index_y in sampled_indices:
            i_indices.append(index_x)
            j_indices.append(index_y)

    return data_t1[i_indices], data_t3[j_indices]


def normalize_cpm(data, scale_factor=1e4):
    """
    La méthode la plus standard en single-cell
    """
    library_size = (1e-16 + data.sum(axis=1, keepdims=True))
    data_normalized = (data / library_size) * scale_factor
    return data # np.nan_to_num(data_normalized)


def compute_entropic_ot_coupling(data_t1=np.zeros(10), data_t3=np.zeros(10), M = np.zeros((10, 10)), epsilon=0, numItermax=100000):
    """
    Calcule le couplage de transport optimal entropique entre deux distributions.
    
    Parameters:
    -----------
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Distribution au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Distribution au temps t3
    epsilon : float
        Coefficient de régularisation entropique
    numItermax : int
        Nombre maximum d'itérations pour Sinkhorn
        
    Returns:
    --------
    coupling : array, shape (n_cells_t1, n_cells_t3)
        Matrice de couplage optimal (plan de transport)
    """
    if np.sum(M) != 0:
        n_cells_t1 = M.shape[0]
        n_cells_t3 = M.shape[1]

    else:
        n_cells_t1 = data_t1.shape[0]
        n_cells_t3 = data_t3.shape[0]
        M = ot.dist(np.log1p(normalize_cpm(data_t1)), np.log1p(normalize_cpm(data_t3)), metric='sqeuclidean')
    
    # Distributions uniformes (mesures empiriques)
    a = np.ones(n_cells_t1)
    b = np.ones(n_cells_t3) * n_cells_t1 / n_cells_t3
    
    # Transport optimal entropique (algorithme de Sinkhorn)
    if epsilon > 0:
        coupling = ot.sinkhorn(a, b, M, reg=epsilon, numItermax=numItermax)
    else:
        coupling = ot.emd(a, b, M, numItermax=numItermax)
    
    return coupling


def compute_ot_distance(data_1, data_2, numItermax=100000):
    """
    Calcule le couplage de transport optimal entropique entre deux distributions.
    
    Parameters:
    -----------
    data_t1 : array-like, shape (n_cells_t1, n_genes)
        Distribution au temps t1
    data_t3 : array-like, shape (n_cells_t3, n_genes)
        Distribution au temps t3
    epsilon : float
        Coefficient de régularisation entropique
    numItermax : int
        Nombre maximum d'itérations pour Sinkhorn
        
    Returns:
    --------
    coupling : array, shape (n_cells_t1, n_cells_t3)
        Matrice de couplage optimal (plan de transport)
    """
    
    n_cells_1 = data_1.shape[0]
    n_cells_2 = data_2.shape[0]
    
    # Distributions uniformes (mesures empiriques)
    a = np.ones(n_cells_1) / n_cells_1
    b = np.ones(n_cells_2) / n_cells_2
    
    # Matrice de coûts (distance euclidienne au carré)
    M = ot.dist(data_1, data_2, metric='sqeuclidean')
    
    # Transport optimal entropique (algorithme de Sinkhorn)
    dist = ot.emd2(a, b, M, numItermax=numItermax)
    
    return dist

