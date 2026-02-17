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
