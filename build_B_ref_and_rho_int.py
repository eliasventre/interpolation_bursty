import numpy as np
import scipy.stats as st
import sys; sys.path += ['../']
import ot #https://pythonot.github.io/index.html
from harissa import NetworkModel 
from optimal_transport import compute_entropic_ot_coupling
import pandas as pd
from joblib import Parallel, delayed
import os
from pathlib import Path
from visualization_utils import plot_exp_vs_sim
from sklearn.decomposition import PCA
import umap.umap_ as umap

n_simu = 10000 # number to simulate to compute Gaussian Kernel
n_samples = 100 # number to take into account to build rho_est

##################Trier par timegap#############################
from collections import defaultdict
groupes = defaultdict(list)

####################CARDA RESULT###################################

gene_name=np.array(pd.read_csv(str('Semrau/Data/panel_genes.txt'), header = None))
nb_gene=len(gene_name) - 1

D= np.array(pd.read_csv(str('Semrau/Rates/degradation_rates.txt'), header = None,sep="\t"))
basal=np.load(str('Semrau/cardamom/basal.npy'))
theta=np.load(str('Semrau/cardamom/inter.npy'))
A = np.zeros((3, nb_gene+1))
kmin = np.load('Semrau/cardamom/kmin.npy')
kmax = np.load('Semrau/cardamom/kmax.npy')
bet = np.load( 'Semrau/cardamom/bet.npy')
data_bool = np.load( 'Semrau/cardamom/data_bool.npy')

####################MODEL WITH X GENES#########################

# Paramètres de transformation selon le fichier de référence
r = 2.5  # technical parameter to transfer the basal regulation in the diagonal of the interaction matrix
fi = 7   # multiplicative coefficient of the interaction matrix

def GRN(nb_gene, theta, basal, D, kmin, kmax, bet, data_bool):
    """
    Construit le modèle exactement comme dans le fichier de référence
    """
    model = NetworkModel(nb_gene)
    model.a = np.zeros((3, nb_gene+1))
    model.a[0, :] = kmin
    model.a[1, :] = kmax
    model.a[2, :] = bet
    model.data_bool = data_bool
    model.d = D.T 
    
    # Application des transformations comme dans build_data
    basal = fi * basal
    inter = fi * theta
    
    # Build the interaction matrix. For technical reasons, we transfer the basal regulation in the diagonal of the matrix
    model.inter = inter.copy()
    model.inter[:, :] = inter[:, :] + (1 - (r/nb_gene)) * np.diag(basal)
    model.inter[1:, 1:] /= (1 - .6 * (r/nb_gene))
    model.inter -= np.diag(np.diag(model.inter)) * .6 * (r/nb_gene)
    model.basal = (r/nb_gene) * basal
    
    return model  


def simu(n, model, M0_cells, data_bool_cells, time_end):
    """
    Lance n simulations à partir de différentes cellules initiales
    M0_cells: matrice (n x nb_gene) des états initiaux SANS normalisation
    data_bool_cells: matrice (n x nb_gene) des booléens pour chaque cellule
    time_end: temps final de simulation
    """
    rho_end = np.zeros((n, M0_cells.shape[1]-1), dtype=float)
    for i in range(n):  # for each cell
        # M0_cells[i] contient déjà le stimulus en position 0 si nécessaire
        sim = model.simulate(time_end, M0 = M0_cells[i], P0 = data_bool_cells[i], use_numba=True)
        rho_end[i, :] = np.random.poisson(sim.m[-1])
    return rho_end


####################PROCESSES#################################

def process_single_cell(i, mu_n_i, data_bool_i, n_simu, model, t1, t2, t3, nu_n):
    """
    Traite une seule cellule initiale - à paralléliser
    mu_n_i: état initial de la cellule i (déjà avec stimulus en position 0)
    data_bool_i: booléen de la cellule i
    """
    # Créer n_simu copies de cette cellule initiale et son booléen
    M0_cells = np.tile(mu_n_i, (n_simu, 1))
    data_bool_cells = np.tile(data_bool_i, (n_simu, 1))
    
    # Simuler les distributions aux temps t2 et t3
    nu_ = simu(n_simu, model, M0_cells, data_bool_cells, t3-t1)
    rho_ = simu(n_samples, model, M0_cells, data_bool_cells, t2-t1)
    
    # Ajout de bruit pour stabilité numérique
    nu_ += 1e-6 * np.random.randn(n_simu, nu_.shape[1])
    rho_ += 1e-6 * np.random.randn(n_samples, rho_.shape[1])

    # Calcul du kernel sur la distribution
    kernel_nu = st.gaussian_kde(nu_.T)

    # Évaluation sur nu_n 
    B_row = np.array([kernel_nu(nu_n[k])[0] for k in range(np.size(nu_n, 0))])
    if B_row.sum() > 0:
        B_row = B_row / B_row.sum()
    
    return i, B_row, rho_

    
def PDMP_ref_interpolation_parallel(mu, nu, data_bool_mu, n_simu, model, t1, t2, t3, n_jobs=-1):
    """
    Version parallélisée avec joblib
    mu_n: données au temps t1 NON normalisées (avec stimulus en colonne 0)
    nu_n: données au temps t3 normalisées UNIQUEMENT pour l'évaluation du kernel
    data_bool_mu: booléens pour les cellules au temps t1
    """
    n1 = np.size(mu, 0)

    print(f"Traitement parallèle de {n1} cellules avec {n_jobs} workers...")
    
    # Parallélisation sur les cellules (boucle principale)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_cell)(
            i, mu[i], data_bool_mu[i], n_simu, model, t1, t2, t3, nu
        ) for i in range(n1)
    )
    
    # Reconstruction de la matrice B_t1_t3
    B_t1_t3 = np.zeros((n1, np.size(nu, 0)))
    all_rho = np.zeros((int(n1 * n_samples), np.size(nu, 1)))
    
    for i, B_row, rho in results:
        B_t1_t3[i, :] = B_row
        all_rho[n_samples * i: n_samples * (i+1), :] = rho[:, :]
    
    return B_t1_t3, all_rho


def sinkhorn(a, b, K): 
    a = a.reshape((K.shape[0], 1))
    b = b.reshape((K.shape[1], 1))
    v = np.ones((K.shape[1], 1), dtype='float')
    u = np.ones((K.shape[0], 1), dtype='float')
    P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
    n_iter = 0
    while n_iter < 1e5:
        u = a/np.maximum((K @ v), 1e-32) # avoid divided by zero
        v = b/np.maximum((K.T @ u), 1e-32)
        P_new = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        n_iter += 1
        if np.linalg.norm(P - P_new) < 1e-16:
            break
        P[:, :] = P_new[:, :]
    return P


def process_timepoint_trio(t1, t2, t3, data, n_simu, model, data_bool, time, n_jobs=-1):
    """
    Traite un trio de timepoints
    """
    print(f"\n{'='*60}")
    print(f"Traitement du trio: {t1}-{t2}-{t3}")
    print(f"{'='*60}\n")
    
    # Créer le dossier de sortie
    output_dir = f"{t1}_{t2}_{t3}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Calcul parallélisé
    PDMP_ref, rho = PDMP_ref_interpolation_parallel(
        data[time == t1], data[time == t3, 1:], data_bool[time == t1], n_simu, model, t1, t2, t3 * (1 + 5 * (t3 > 95)), n_jobs=n_jobs
    )
    
    PDMP_ref += 1e-6 # Eviter les problemes numeriques

    # Sauvegarde des résultats
    np.savetxt(f'{output_dir}/PDMP_ref_{t1}_{t3}.txt', PDMP_ref)
    np.savetxt(f'{output_dir}/rho_est_t{t2}.txt', rho)
    
    # Calcul du couplage OT
    a = np.ones(PDMP_ref.shape[0]) 
    b = np.ones(PDMP_ref.shape[1]) * PDMP_ref.shape[0] / PDMP_ref.shape[1]
    PDMP_sch = sinkhorn(a, b, PDMP_ref)
    np.savetxt(f'{output_dir}/PDMP_sch_{t1}_{t3}.txt', PDMP_sch)
    
    print(f"✓ Trio {t1}-{t2}-{t3} terminé. Résultats sauvegardés dans '{output_dir}/'")
    
    return {
        'trio': f"{t1}-{t2}-{t3}",
        'PDMP_ref': PDMP_ref,
        'PDMP_sch': PDMP_sch,
        'rho': rho
    }


#######################MAIN###############################

if __name__ == "__main__":
    
    # Configuration
    n_jobs = -1  # -1 utilise tous les cœurs disponibles
    method = 'umap'
    
    # Créer le modèle avec les transformations correctes
    model = GRN(nb_gene, theta, basal, D, kmin, kmax, bet, data_bool)
    
    # Définir tous les trios de timepoints
    timepoint_trios = [
        (0, 6, 12),
        (6, 12, 24),
        (12, 24, 36),
        (24, 36, 48),
        (36, 48, 60),
        (48, 60, 72),
        (60, 72, 96)
    ]
    
    print(f"\n{'#'*60}")
    print(f"# DÉMARRAGE DES SIMULATIONS PARALLÉLISÉES")
    print(f"# Nombre de simulations par cellule: {n_simu}")
    print(f"# Nombre de workers: {n_jobs if n_jobs > 0 else 'tous les cœurs'}")
    print(f"# Nombre de trios à traiter: {len(timepoint_trios)}")
    print(f"{'#'*60}\n")  

    # Build the timepoints
    data_real = np.loadtxt('Semrau/Data/panel_real.txt', dtype=float, delimiter='\t')[1:, 1:].T
    data_real[:, 0] = np.loadtxt('Semrau/Data/panel_real.txt', dtype=float, delimiter='\t')[0, 1:]
    t_real = list(set(data_real[:, 0]))
    t_real.sort()
    t = np.array(t_real, dtype=int)
    my_k = [np.sum(data_real[:, 0] == times) for times in t_real]
    C = int(np.sum(my_k))
    k = [int(np.sum(my_k[:i])) for i in range(0, len(t_real))] + [C]
    time = np.zeros(C)
    for i in range(0, len(t)): time[k[i]:k[i + 1]] = t[i]
    
    # Traiter chaque trio
    all_results = []
    for t1, t2, t3 in timepoint_trios:
        try:
            result = process_timepoint_trio(
                t1, t2, t3, 
                data_real, 
                n_simu, 
                model, 
                data_bool,
                time,
                n_jobs=n_jobs
            )
            all_results.append(result)
        except Exception as e:
            print(f"✗ Erreur lors du traitement du trio {t1}-{t2}-{t3}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'#'*60}")
    print(f"# SIMULATIONS TERMINÉES")
    print(f"# Trios traités avec succès: {len(all_results)}/{len(timepoint_trios)}")
    print(f"{'#'*60}\n")
    
    # Analyse PCA/UMAP globale exp vs simulé
    if len(all_results) > 0:

        X_exp, exp_time_labels = data_real[:, 1:], time.copy()
        sim_time_labels = time[(time > np.min(time)) & (time < np.max(time))]
        time_unique = np.sort(np.unique(sim_time_labels))
        X_sim = np.vstack([result['rho'][np.random.choice(result['rho'].shape[0], int(np.sum(time == time_unique[i])), replace=False), :] for i, result in enumerate(all_results)])

        # 3) Apprendre PCA/UMAP sur concaténation (exp_all + sim)
        X_all_for_embed = np.vstack([X_exp, X_sim])

        if method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            X_all = pca.fit_transform(X_all_for_embed)

        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_all = reducer.fit_transform(X_all_for_embed)

        n_exp_all = data_real.shape[0]
        embeddings = {
            f'X_exp_{method}': X_all[:n_exp_all],
            f'X_sim_{method}': X_all[n_exp_all:]
        }

        # 4) Plot avec la même colormap viridis et les labels temps correspondants
        plot_exp_vs_sim(
            embeddings,
            exp_time_labels=time,   # tous les temps
            sim_time_labels=sim_time_labels,       # t2 seulement
            output_path="exp_vs_sim.png",
            method=method
        )
    else:
        print("Aucun résultat de simulation disponible, PCA/UMAP non calculés.")