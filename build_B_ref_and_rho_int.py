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
from visualization_utils import build_exp_and_sim_matrices, plot_exp_vs_sim, build_all_exp_matrix
from sklearn.decomposition import PCA
import umap.umap_ as umap

data=pd.read_csv('Semrau/Data/panel_real.txt',sep='\t')
n_simu = 10000

##################Trier par timegap#############################
from collections import defaultdict
groupes = defaultdict(list)

for col in data.columns:
    base_name = str(col).split('.')[0]
    groupes[base_name].append(col)

data_t = {nom: data[cols] for nom, cols in groupes.items()}

for nom, cols in groupes.items():
    data_t[f"data{nom}"] = data[cols]

# Charger tous les timepoints disponibles
data_t0 = np.array(data_t["data0"])[1:].T #to avoid stimulus
data_t6 = np.array(data_t["data6"])[1:].T
data_t12 = np.array(data_t["data12"])[1:].T
data_t24 = np.array(data_t["data24"])[1:].T
data_t36 = np.array(data_t["data36"])[1:].T
data_t48 = np.array(data_t["data48"])[1:].T
data_t60 = np.array(data_t["data60"])[1:].T
data_t72 = np.array(data_t["data72"])[1:].T
data_t96 = np.array(data_t["data96"])[1:].T

# Dictionnaire pour accès facile
timepoint_data = {
    0: data_t0,
    6: data_t6,
    12: data_t12,
    24: data_t24,
    36: data_t36,
    48: data_t48,
    60: data_t60,
    72: data_t72,
    96: data_t96
}

####################CARDA RESULT###################################

gene_name=np.array(pd.read_csv(str('Semrau/Data/panel_genes.txt'), header = None))
nb_gene=41 #len(mu_n.T)

D= np.array(pd.read_csv(str('Semrau/Rates/degradation_rates.txt'), header = None,sep="\t"))
B_raw=np.load(str('Semrau/cardamom/basal.npy'))
Theta_raw=np.load(str('Semrau/cardamom/inter.npy'))
A = np.zeros((3, nb_gene+1))
kmin = np.load('Semrau/cardamom/kmin.npy')
kmax = np.load('Semrau/cardamom/kmax.npy')
bet = np.load( 'Semrau/cardamom/bet.npy')
data_bool = np.load( 'Semrau/cardamom/data_bool.npy')

####################MODEL WITH X GENES#########################

# Paramètres de transformation selon le fichier de référence
r = 2.5  # technical parameter to transfer the basal regulation in the diagonal of the interaction matrix
fi = 7   # multiplicative coefficient of the interaction matrix

def GRN(nb_gene, Theta_raw, B_raw, D, kmin, kmax, bet, data_bool):
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
    basal = fi * B_raw
    inter = fi * Theta_raw
    
    # Build the interaction matrix. For technical reasons, we transfer the basal regulation in the diagonal of the matrix
    model.inter = inter.copy()
    model.inter[:, :] = inter[:, :] + (1 - r/nb_gene) * np.diag(basal)
    model.inter[1:, 1:] /= (1 - .6 * r/nb_gene)
    model.inter -= np.diag(np.diag(model.inter)) * .6 * r/nb_gene
    model.basal = r/nb_gene * basal
    
    return model  

def simu_single_cell(model, M0, P0, time_end):
    """
    Simule une seule cellule selon la méthode du fichier de référence
    M0: vecteur initial des expressions (SANS normalisation)
    data_bool_cell: vecteur booléen pour cette cellule
    time_end: temps final de simulation
    """
    sim = model.simulate(time_end, M0=M0, P0=P0, use_numba=True)
    return sim

def simu(n, model, M0_cells, data_bool_cells, time_int, time_end):
    """
    Lance n simulations à partir de différentes cellules initiales
    M0_cells: matrice (n x nb_gene) des états initiaux SANS normalisation
    data_bool_cells: matrice (n x nb_gene) des booléens pour chaque cellule
    time_end: temps final de simulation
    """
    S_end = []
    S_int = []
    for i in range(n):  # for each cell
        # M0_cells[i] contient déjà le stimulus en position 0 si nécessaire
        sim = simu_single_cell(model, M0_cells[i], data_bool_cells[i], time_int)
        S_int.append(sim)
        stimulus_col = np.ones(1)  # Stimulus = 1 pour t > 0
        M0_int = np.hstack([stimulus_col,  sim.m[:, :][-1]])
        P0_int = np.hstack([stimulus_col,  sim.p[:, :][-1]])
        sim = simu_single_cell(model, M0_int, P0_int, time_end-time_int)
        S_end.append(sim)

    return S_int, S_end

def extract_distributions(S):
    """
    Extrait les distributions aux timepoints t2 et t3 (indices dans la simulation)
    SANS normalisation
    """
    rho = [] 
    for sim in S:
        cell_t = np.random.poisson(sim.m[-1])
        rho.append(cell_t)
    return np.array(rho, dtype=float)

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
    sims_t2, sims_t3 = simu(n_simu, model, M0_cells, data_bool_cells, t2-t1, t3-t1)
    
    # Extraire les distributions aux temps t2 et t3 SANS normalisation
    rho_ = extract_distributions(sims_t2)
    nu_ = extract_distributions(sims_t3)
    
    # Ajout de bruit pour stabilité numérique
    nu_ += 1e-5 * np.random.randn(*nu_.shape)
    rho_ += 1e-5 * np.random.randn(*rho_.shape)
    
    # Calcul du kernel sur la distribution
    maxi = 1.0 # np.maximum(np.max(nu_n, axis=0, keepdims=True), 1)
    kernel_nu = st.gaussian_kde((nu_ / maxi).T)
    
    # Évaluation sur nu_n 
    B_row = np.array([kernel_nu(nu_n[k]/maxi)[0] for k in range(len(nu_n))])
    if B_row.sum() > 0:
        B_row = B_row / B_row.sum()
    
    return i, B_row, rho_

    
def PDMP_ref_interpolation_parallel(mu_n, nu_n, data_bool_mu, n_simu, model, t1, t2, t3, n_jobs=-1):
    """
    Version parallélisée avec joblib
    mu_n: données au temps t1 NON normalisées (avec stimulus en colonne 0)
    nu_n: données au temps t3 normalisées UNIQUEMENT pour l'évaluation du kernel
    data_bool_mu: booléens pour les cellules au temps t1
    """
    n1 = len(mu_n)
    n2 = len(nu_n)
    
    print(f"Traitement parallèle de {n1} cellules avec {n_jobs} workers...")
    
    # Parallélisation sur les cellules (boucle principale)
    results = Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_single_cell)(
            i, mu_n[i], data_bool_mu[i], n_simu, model, t1, t2, t3, nu_n
        ) for i in range(n1)
    )
    
    # Reconstruction de la matrice B_t1_t3
    B_t1_t3 = np.zeros((int(n1), int(n2)))
    all_rho = []
    
    for i, B_row, rho in results:
        B_t1_t3[i, :] = B_row
        all_rho.append(rho)
    
    # Kernel global sur rho
    rho_global = np.vstack(all_rho)
    
    return B_t1_t3, rho_global


def prepare_data(data_t1, data_t3):
    """
    Prépare les données pour la simulation
    - Ajoute le stimulus en colonne 0
    - Calcule le max pour la normalisation du kernel uniquement
    - Retourne mu (NON normalisé), nu (normalisé pour kernel), et maxi
    """
    
    # mu_n: données à t1 AVEC stimulus, NON normalisées
    stimulus_col = np.ones((data_t1.shape[0], 1))  # Stimulus = 1 pour t > 0
    mu_n = np.hstack([stimulus_col, data_t1])
    
    # nu_n: données à t3 
    nu_n = data_t3.copy()
    
    return mu_n, nu_n


def sinkhorn(a, b, K): 
    a = a.reshape((K.shape[0], 1))
    b = b.reshape((K.shape[1], 1))
    v = np.ones((K.shape[1], 1), dtype='float')
    n_iter = 0
    while n_iter < 1e4:
        u = a/np.maximum((K @ v), 1e-32) # avoid divided by zero
        v = b/np.maximum((K.T @ u), 1e-32)
        P = np.diag(u.flatten()) @ K @ np.diag(v.flatten())
        n_iter += 1
    return P


def process_timepoint_trio(t1, t2, t3, timepoint_data, n_simu, model, data_bool, n_jobs=-1):
    """
    Traite un trio de timepoints
    """
    print(f"\n{'='*60}")
    print(f"Traitement du trio: {t1}-{t2}-{t3}")
    print(f"{'='*60}\n")
    
    # Créer le dossier de sortie
    output_dir = f"{t1}_{t2}_{t3}"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Préparation des données
    mu_n, nu_n = prepare_data(timepoint_data[t1], timepoint_data[t3])
    
    # Récupérer les data_bool correspondants aux cellules de t1
    # On suppose que data_bool a autant de lignes que de cellules dans le dataset complet
    # Il faut adapter selon la structure exacte de tes données
    data_bool_mu = data_bool[:len(mu_n), :]
    
    # Calcul parallélisé
    PDMP_ref, rho = PDMP_ref_interpolation_parallel(
        mu_n, nu_n, data_bool_mu, n_simu, model, t1, t2, t3, n_jobs=n_jobs
    )
    
    # Sauvegarde des résultats
    np.savetxt(f'{output_dir}/PDMP_ref_{t1}_{t3}.txt', PDMP_ref)
    np.savetxt(f'{output_dir}/rho_est_t{t2}.txt', rho.T)
    
    # Calcul du couplage OT
    a = np.ones(PDMP_ref.shape[0]) 
    b = np.ones(PDMP_ref.shape[1]) * PDMP_ref.shape[0] / PDMP_ref.shape[1]
    PDMP_sch = compute_entropic_ot_coupling(M=-np.log(1e-92 + PDMP_ref), epsilon=1.0)
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
    model = GRN(nb_gene, Theta_raw, B_raw, D, kmin, kmax, bet, data_bool)
    
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
    
    # Traiter chaque trio
    all_results = []
    for t1, t2, t3 in timepoint_trios:
        try:
            result = process_timepoint_trio(
                t1, t2, t3, 
                timepoint_data, 
                n_simu, 
                model, 
                data_bool,
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
        # 1) Construire toutes les données exp
        X_exp_all, exp_time_labels_all = build_all_exp_matrix(timepoint_data, max_cells_per_time=None)

        # 2) Construire uniquement les données simulées à t2 (intermédiaires)
        X_exp_dummy, X_sim, _, sim_time_labels = build_exp_and_sim_matrices(
            timepoint_trios, timepoint_data, all_results, max_cells_per_trio=200, log1p=False
        )

        # 3) Apprendre PCA/UMAP sur concaténation (exp_all + sim)
        X_all_for_embed = np.vstack([X_exp_all, X_sim])

        if method == 'pca':
            pca = PCA(n_components=2, random_state=42)
            X_all = pca.fit_transform(X_all_for_embed)

        if method == 'umap':
            reducer = umap.UMAP(n_components=2, random_state=42)
            X_all = reducer.fit_transform(X_all_for_embed)

        n_exp_all = X_exp_all.shape[0]
        embeddings = {
            f'X_exp_{method}': X_all[:n_exp_all],
            f'X_sim_{method}': X_all[n_exp_all:]
        }

        # 4) Plot avec la même colormap viridis et les labels temps correspondants
        plot_exp_vs_sim(
            embeddings,
            exp_time_labels=exp_time_labels_all,   # tous les temps
            sim_time_labels=sim_time_labels,       # t2 seulement
            output_path="outputs/exp_vs_sim.png",
            method=method
        )
    else:
        print("Aucun résultat de simulation disponible, PCA/UMAP non calculés.")