from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import os
from matplotlib import cm
from matplotlib import gridspec
import matplotlib as mpl



######################################################################
# UTILITAIRES POUR PCA / UMAP ET PLOTS
######################################################################


def build_all_exp_matrix(timepoint_data, max_cells_per_time=None):
    """
    Construit X_exp_all et labels temps pour tous les timepoints disponibles.
    """
    exp_list = []
    time_labels = []

    for t, X_t in timepoint_data.items():
        X = X_t
        n = X.shape[0]

        if max_cells_per_time is not None and n > max_cells_per_time:
            idx = np.random.choice(n, size=max_cells_per_time, replace=False)
            X = X[idx]
            n = X.shape[0]

        exp_list.append(X)
        time_labels.append(np.full(n, t))

    X_exp_all = np.vstack(exp_list)
    time_labels_all = np.concatenate(time_labels)
    return X_exp_all, time_labels_all



def build_exp_and_sim_matrices(timepoint_trios, timepoint_data, all_results, max_cells_per_trio=1000, log1p=True):
    exp_list = []
    sim_list = []
    exp_time_labels = []
    sim_time_labels = []

    trio_to_result = {res['trio']: res for res in all_results}

    for (t1, t2, t3) in timepoint_trios:
        trio_key = f"{t1}-{t2}-{t3}"
        if trio_key not in trio_to_result:
            print(f"Attention: pas de résultats pour {trio_key}, ignoré dans l'analyse PCA/UMAP.")
            continue

        res = trio_to_result[trio_key]

        # Données expérimentales à TOUS les timepoints:
        # ici tu voulais "tous les timepoints", donc on va prendre toutes les data_t? concaténées
        # mais on garde le label du temps correspondant.
        # Pour commencer, on reste sur t2 pour la comparaison directe, mais on va ensuite étendre.

        X_exp_t2 = timepoint_data[t2]  # (n_cells_t2, n_genes)
        n_exp_t2 = X_exp_t2.shape[0]

        # Simulations à t2 : (n_genes, n_sim_cells)
        rho = res['rho']
        X_sim_t2_full = rho.copy()  # (n_sim_cells, n_genes)
        n_sim_t2 = X_sim_t2_full.shape[0]

        # Sous-échantillonnage éventuel
        if max_cells_per_trio is not None:
            if n_exp_t2 > max_cells_per_trio:
                idx_exp = np.random.choice(n_exp_t2, size=max_cells_per_trio, replace=False)
                X_exp_t2 = X_exp_t2[idx_exp]
                n_exp_t2 = X_exp_t2.shape[0]
            if n_sim_t2 > max_cells_per_trio:
                idx_sim = np.random.choice(n_sim_t2, size=max_cells_per_trio, replace=False)
                X_sim_t2_full = X_sim_t2_full[idx_sim]
                n_sim_t2 = X_sim_t2_full.shape[0]

        if log1p:
            X_exp_t2 = np.log1p(X_exp_t2)
            X_sim_t2_full = np.log1p(X_sim_t2_full)

        exp_list.append(X_exp_t2)
        sim_list.append(X_sim_t2_full)
        exp_time_labels.append(np.full(n_exp_t2, t2))
        sim_time_labels.append(np.full(n_sim_t2, t2))

    if len(exp_list) == 0 or len(sim_list) == 0:
        raise ValueError("Aucune donnée expérimentale ou simulée disponible pour PCA/UMAP.")

    X_exp = np.vstack(exp_list)
    X_sim = np.vstack(sim_list)
    exp_time_labels = np.concatenate(exp_time_labels)
    sim_time_labels = np.concatenate(sim_time_labels)

    return X_exp, X_sim, exp_time_labels, sim_time_labels


def compute_pca_umap_embeddings(X_exp, X_sim, n_components_pca=2, n_components_umap=2, random_state=42):
    """
    Apprend PCA et UMAP sur la concaténation (exp + simu), puis renvoie les embeddings séparés.
    """
    # Concaténer les données
    X_all = np.vstack([X_exp, X_sim])

    # PCA
    pca = PCA(n_components=n_components_pca, random_state=random_state)
    X_all_pca = pca.fit_transform(X_all)

    # UMAP
    reducer = umap.UMAP(n_components=n_components_umap, random_state=random_state)
    X_all_umap = reducer.fit_transform(X_all)

    # Séparer exp et simu
    n_exp = X_exp.shape[0]
    X_exp_pca = X_all_pca[:n_exp]
    X_sim_pca = X_all_pca[n_exp:]
    X_exp_umap = X_all_umap[:n_exp]
    X_sim_umap = X_all_umap[n_exp:]

    return {
        'X_exp_pca': X_exp_pca,
        'X_sim_pca': X_sim_pca,
        'X_exp_umap': X_exp_umap,
        'X_sim_umap': X_sim_umap
    }

def scatter_with_time(ax, X, time_labels, title, cmap, t_min, t_max):
    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    colors = cmap(norm(time_labels))
    ax.scatter(X[:, 0], X[:, 1], s=5, alpha=0.8, c=colors, edgecolors='none')
    ax.set_title(title, fontsize=13)
    ax.tick_params(labelsize=10)


def plot_exp_vs_sim(embeddings, exp_time_labels, sim_time_labels, output_path, method='umap'):

    if method == 'pca':
        X_exp = embeddings['X_exp_pca']
        X_sim = embeddings['X_sim_pca']
    if method == 'umap':
        X_exp = embeddings['X_exp_umap']
        X_sim = embeddings['X_sim_umap']

    # Temps uniques pour une colormap continue
    all_times = np.concatenate([exp_time_labels, sim_time_labels])
    t_min, t_max = all_times.min(), all_times.max()
    norm_times = lambda t: (t - t_min) / (t_max - t_min + 1e-9)  # normalisation 0-1
    cmap = cm.get_cmap('viridis')

    fig = plt.figure(figsize=(12, 5))

    # 2 colonnes pour les nuages, 1 colonne fine pour la colorbar
    gs = gridspec.GridSpec(
        1, 3,
        width_ratios=[1, 1, 0.05],  # la dernière est très étroite
        wspace=0.15
    )

    ax_exp = fig.add_subplot(gs[0, 0])
    ax_sim = fig.add_subplot(gs[0, 1])
    ax_cb  = fig.add_subplot(gs[0, 2])

    # UMAP / PCA à gauche et à droite
    scatter_with_time(ax_exp, X_exp, exp_time_labels, "Données expérimentales", cmap, t_min, t_max)
    scatter_with_time(ax_sim, X_sim, sim_time_labels, "Données simulées (intermédiaires)", cmap, t_min, t_max)

    # Colorbar verticale compacte à droite
    norm = mpl.colors.Normalize(vmin=t_min, vmax=t_max)
    sm = mpl.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = fig.colorbar(sm, cax=ax_cb)
    cbar.set_label("Temps (h)", fontsize=11)
    cbar.ax.tick_params(labelsize=9)


    plt.tight_layout()
    Path(os.path.dirname(output_path)).mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Figure exp vs simulée sauvegardée : {output_path}")
