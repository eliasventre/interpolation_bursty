"""
Module de transport optimal entropique pour calculer le couplage OT classique
et l'interpolation de McCann associée.
"""

import numpy as np
import ot


def compute_entropic_ot_coupling(data_t1, data_t3, epsilon=0.01, numItermax=10000):
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
    
    n_cells_t1 = data_t1.shape[0]
    n_cells_t3 = data_t3.shape[0]
    
    # Distributions uniformes (mesures empiriques)
    a = np.ones(n_cells_t1) / n_cells_t1
    b = np.ones(n_cells_t3) / n_cells_t3
    
    # Matrice de coûts (distance euclidienne au carré)
    M = ot.dist(np.log(1 + data_t1), np.log(1 + data_t3), metric='euclidean')
    M = M ** 2  # Distance quadratique pour Wasserstein-2
    
    # Transport optimal entropique (algorithme de Sinkhorn)
    coupling = ot.sinkhorn(a, b, M, reg=epsilon, numItermax=numItermax)
    
    return coupling


def compute_ot_distance(data_1, data_2, numItermax=10000):
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
    M = ot.dist(data_1, data_2, metric='euclidean')
    M = M ** 2  # Distance quadratique pour Wasserstein-2
    
    # Transport optimal entropique (algorithme de Sinkhorn)
    dist = ot.emd2(a, b, M, numItermax=numItermax)
    
    return dist

