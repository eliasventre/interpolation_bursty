"""
Script principal pour l'analyse de validation du modèle Bursty
via interpolation de McCann sur TOUS les intervalles temporels.

Ce script:
1. Analyse chaque intervalle temporel (0_6_12, 12_24_36, etc.)
2. Optimise alpha et beta pour chaque intervalle
3. Compare Bursty vs OT pour chaque intervalle
4. Génère une analyse globale avec boxplots par gène
5. PONDÉRATION: Utilisée UNIQUEMENT pour les métriques globales agrégées
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from optimization import optimize_alpha_beta_complete, mccann_interpolation
from optimal_transport import compute_ot_distance, compute_entropic_ot_coupling

# Paramètres globaux
n_samples_optim = 6
n_samples_interpol = 12
lr = 0.01
num_iter = 0
blur = 1.0


# Configuration des intervalles temporels à analyser
TIME_INTERVALS = [
    {'folder': '0_6_12', 't1': 0, 't2': 6, 't3': 12},
    {'folder': '6_12_24', 't1': 6, 't2': 12, 't3': 24},
    {'folder': '12_24_36', 't1': 12, 't2': 24, 't3': 36},
    {'folder': '24_36_48', 't1': 24, 't2': 36, 't3': 48},
    {'folder': '36_48_60', 't1': 36, 't2': 48, 't3': 60},
    {'folder': '48_60_72', 't1': 48, 't2': 60, 't3': 72},
    {'folder': '60_72_96', 't1': 60, 't2': 72, 't3': 96}
]


def load_data(folder='data', time_init=0, time_int=0, time_final=0):
    """
    Charge toutes les données nécessaires pour un intervalle temporel donné.
    
    Returns:
    --------
    dict contenant:
        - data_t1, data_t2, data_t3: données aux différents temps
        - B_ref, B_sch: couplages de référence et de Schrödinger
        - rho_est: distribution estimée de référence
        - genes_names: noms des gènes
    """
    
    # Chargement des fichiers
    genes_names = pd.read_csv('panel_genes.txt', sep='\t')
    B_ref = pd.read_csv(f'{folder}/PDMP_ref_{time_init}_{time_final}.txt', sep=' ', header=None)
    B_sch = pd.read_csv(f'{folder}/PDMP_sch_{time_init}_{time_final}.txt', sep=' ', header=None)
    rho_est = pd.read_csv(f'{folder}/rho_est_t{time_int}.txt', sep=' ', header=None)
    
    # Conversion en arrays numpy
    B_ref = np.array(B_ref)
    B_sch = np.array(B_sch)
    rho_est = np.array(rho_est)
    
    # Trier par timegap
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
    
    return {
        'data_t1': data_real[time == time_init, 1:],
        'data_t2': data_real[time == time_int, 1:],
        'data_t3': data_real[time == time_final, 1:],
        'B_ref': B_ref,
        'B_sch': B_sch,
        'rho_est': rho_est,
        'genes_names': genes_names
    }


def analyze_single_interval(interval_config, verbose=True):
    """
    Analyse un seul intervalle temporel.
    
    Parameters:
    -----------
    interval_config : dict
        Configuration de l'intervalle avec 'folder', 't1', 't2', 't3'
    verbose : bool
        Affichage détaillé
        
    Returns:
    --------
    dict contenant tous les résultats de l'analyse pour cet intervalle
    """
    
    folder = interval_config['folder']
    t1, t2, t3 = interval_config['t1'], interval_config['t2'], interval_config['t3']
    
    if verbose:
        print("\n" + "="*70)
        print(f"ANALYSE DE L'INTERVALLE {folder} ({t1}h → {t2}h → {t3}h)")
        print("="*70)
    
    # Chargement des données
    data = load_data(folder, t1, t2, t3)
    
    data_t1 = data['data_t1'] 
    data_t2 = data['data_t2'] 
    data_t3 = data['data_t3'] 
    B_ref = data['B_ref']
    B_sch = data['B_sch']
    rho_est = data['rho_est']
    genes_names = data['genes_names']
    
    n_genes = data_t1.shape[1]
    
    if verbose:
        print(f"Nombre de gènes: {n_genes}")
        print(f"Cellules: t1={data_t1.shape[0]}, t2={data_t2.shape[0]}, t3={data_t3.shape[0]}")
    
    # Optimisation
    if verbose:
        print(f"\nOptimisation...")
    
    alpha_opt, beta_opt = optimize_alpha_beta_complete(
        data_t1=data_t1,
        data_t3=data_t3,
        B_ref=B_ref,
        rho_ref=rho_est[np.random.choice(rho_est.shape[0], size=n_samples_optim * B_ref.shape[0]), :],
        alpha_init =((t2 - t1) / (t3 - t1)) *  np.ones(n_genes),
        beta_init = ((t3 - t2) / (t3 - t1)) *  np.ones(n_genes),
        n_samples=n_samples_optim,
        n_iterations=num_iter,
        lr=lr,
        blur=blur,
        verbose=verbose,
        constrain=True
    )
    
    # Interpolation Bursty
    rho_bursty = mccann_interpolation(
        data_t1=data_t1,
        data_t3=data_t3,
        coupling=B_sch,
        alpha=alpha_opt,
        beta=beta_opt,
        n_samples=n_samples_interpol
    )
    
    # Couplage OT classique
    OT_coupling = compute_entropic_ot_coupling(
        data_t1=data_t1,
        data_t3=data_t3,
        epsilon=0, 
    )
    
    # Interpolation OT
    rho_OT = mccann_interpolation(
        data_t1=data_t1,
        data_t3=data_t3,
        coupling=OT_coupling,
        alpha=np.array([(t2 - t1) / (t3 - t1)]),
        beta=np.array([(t3 - t2) / (t3 - t1)]),
        n_samples=n_samples_interpol
    )
    
    # Calcul des EMD par gène
    EMD_Bursty = np.zeros(n_genes)
    EMD_OT = np.zeros(n_genes)
    OT_difficulty = np.zeros(n_genes)  # Distance OT entre t1 et t3 (difficulté)
    
    for gene_idx in range(n_genes):
        EMD_Bursty[gene_idx] = compute_ot_distance(
            rho_bursty[:, gene_idx:gene_idx+1], 
            data_t2[:, gene_idx:gene_idx+1]
        )
        EMD_OT[gene_idx] = compute_ot_distance(
            rho_OT[:, gene_idx:gene_idx+1], 
            data_t2[:, gene_idx:gene_idx+1]
        )
        # Distance OT entre t1 et t3 pour ce gène (difficulté)
        OT_difficulty[gene_idx] = compute_ot_distance(
            data_t1[:, gene_idx:gene_idx+1],
            data_t3[:, gene_idx:gene_idx+1]
        )
    
    Delta_EMD = EMD_OT - EMD_Bursty
    
    if verbose:
        print(f"\nRésultats:")
        print(f"  Delta_EMD moyen: {Delta_EMD.mean():.6f}")
        print(f"  Bursty meilleur: {(Delta_EMD > 0).sum()}/{n_genes} gènes")
    
    # Récupération des noms de gènes
    if isinstance(genes_names, pd.DataFrame):
        gene_labels = genes_names.iloc[:, 0].values if len(genes_names.columns) > 0 else [f"Gene{i}" for i in range(n_genes)]
    else:
        gene_labels = [f"Gene{i}" for i in range(n_genes)]
    
    return {
        'folder': folder,
        'interval': f"{t1}-{t2}-{t3}",
        'gene_labels': gene_labels,
        'alpha': alpha_opt,
        'beta': beta_opt,
        'EMD_Bursty': EMD_Bursty,
        'EMD_OT': EMD_OT,
        'Delta_EMD': Delta_EMD,
        'OT_difficulty': OT_difficulty,
        'n_genes': n_genes
    }


def generate_interval_plots(results, output_folder='outputs'):
    """
    Génère les plots individuels pour chaque intervalle.
    PAS DE PONDÉRATION - Juste les EMD bruts.
    """
    
    Path(output_folder).mkdir(exist_ok=True)
    folder = results['folder']
    gene_labels = results['gene_labels']
    Delta_EMD = results['Delta_EMD']
    EMD_Bursty = results['EMD_Bursty']
    EMD_OT = results['EMD_OT']
    n_genes = results['n_genes']
    
    # Figure 1: Delta_EMD brut (SANS pondération)
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['green' if d > 0 else 'red' for d in Delta_EMD]
    ax.bar(range(n_genes), Delta_EMD, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('$\\Delta_{EMD} = EMD_{OT} - EMD_{Bursty}$', fontsize=12)
    ax.set_title(f'Delta EMD pour {folder}\n(Vert = Bursty meilleur, Rouge = OT meilleur)', 
                 fontsize=13, fontweight='bold')
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/delta_emd_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Figure 2: Comparaison EMD
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_genes)
    width = 0.35
    ax.bar(x - width/2, EMD_Bursty, width, label='Bursty', color='blue', alpha=0.7, 
           edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, EMD_OT, width, label='OT', color='orange', alpha=0.7, 
           edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('EMD', fontsize=12)
    ax.set_title(f'Comparaison des EMD pour {folder}', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/emd_comparison_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_global_analysis(all_results, output_folder='outputs'):
    """
    Génère l'analyse globale avec pondération UNIQUEMENT pour les métriques agrégées.
    
    Visualisations:
    1. Boxplots Delta_EMD (SANS pondération)
    2. Heatmap Delta_EMD (SANS pondération)
    3. Heatmap EMD_Bursty (SANS pondération)
    4. Heatmap EMD_OT (SANS pondération)
    5. Score global par gène (AVEC pondération par difficulté)
    6. Performance par intervalle (AVEC pondération par difficulté)
    """
    
    Path(output_folder).mkdir(exist_ok=True)
    
    # Récupération des données
    gene_labels = all_results[0]['gene_labels']
    n_genes = all_results[0]['n_genes']
    n_intervals = len(all_results)
    interval_labels = [r['interval'] for r in all_results]
    
    # Matrices SANS pondération
    Delta_EMD_matrix = np.array([r['Delta_EMD'] for r in all_results])
    EMD_Bursty_matrix = np.array([r['EMD_Bursty'] for r in all_results])
    EMD_OT_matrix = np.array([r['EMD_OT'] for r in all_results])
    OT_difficulty_matrix = np.array([r['OT_difficulty'] for r in all_results])
    
    print(f"\n{'='*70}")
    print("GÉNÉRATION DES ANALYSES GLOBALES")
    print(f"{'='*70}\n")
    
    # ============================================================================
    # FIGURE 1: BOXPLOTS Delta_EMD (SANS pondération)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    boxplot_data = [Delta_EMD_matrix[:, gene_idx] for gene_idx in range(n_genes)]
    
    bp = ax.boxplot(boxplot_data, 
                     positions=range(n_genes),
                     widths=0.6,
                     patch_artist=True,
                     showfliers=True,
                     boxprops=dict(facecolor='lightblue', edgecolor='black', linewidth=1),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(color='black', linewidth=1),
                     capprops=dict(color='black', linewidth=1),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
    
    # Coloration selon médiane
    for i, box in enumerate(bp['boxes']):
        median_val = np.median(Delta_EMD_matrix[:, i])
        if median_val > 0:
            box.set_facecolor('lightgreen')
        else:
            box.set_facecolor('lightcoral')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=1)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('$\\Delta_{EMD}$ (non pondéré)', fontsize=13)
    ax.set_title(f'Distribution de $\\Delta_{{EMD}}$ par gène sur {n_intervals} intervalles\n'
                 f'(Vert = médiane positive, Rouge = médiane négative)', 
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_delta_emd_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Boxplots Delta_EMD sauvegardés")
    
    # ============================================================================
    # FIGURE 2: HEATMAP Delta_EMD (SANS pondération)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(Delta_EMD_matrix, aspect='auto', cmap='RdYlGn', 
                   vmin=-np.abs(Delta_EMD_matrix).max(), 
                   vmax=np.abs(Delta_EMD_matrix).max())
    
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_intervals))
    ax.set_yticklabels(interval_labels, fontsize=11)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('Intervalles temporels', fontsize=13)
    ax.set_title('Heatmap des $\\Delta_{EMD}$ (tous intervalles, non pondéré)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta_{EMD}$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_delta_emd_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Heatmap Delta_EMD sauvegardée")
    
    # ============================================================================
    # FIGURE 3: HEATMAP EMD_Bursty (SANS pondération)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(EMD_Bursty_matrix, aspect='auto', cmap='YlOrRd', 
                   vmin=0, vmax=np.max([EMD_Bursty_matrix.max(), EMD_OT_matrix.max()]))
    
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_intervals))
    ax.set_yticklabels(interval_labels, fontsize=11)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('Intervalles temporels', fontsize=13)
    ax.set_title('Heatmap des EMD Bursty (tous intervalles)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('EMD Bursty', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_emd_bursty_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Heatmap EMD_Bursty sauvegardée")
    
    # ============================================================================
    # FIGURE 4: HEATMAP EMD_OT (SANS pondération)
    # ============================================================================
    fig, ax = plt.subplots(figsize=(16, 6))
    
    im = ax.imshow(EMD_OT_matrix, aspect='auto', cmap='YlOrRd',
                   vmin=0, vmax=np.max([EMD_Bursty_matrix.max(), EMD_OT_matrix.max()]))
    
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_intervals))
    ax.set_yticklabels(interval_labels, fontsize=11)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('Intervalles temporels', fontsize=13)
    ax.set_title('Heatmap des EMD OT (tous intervalles)', 
                 fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('EMD OT', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_emd_ot_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Heatmap EMD_OT sauvegardée")
    
    # ============================================================================
    # FIGURE 5: SCORE GLOBAL PAR GÈNE (AVEC pondération par difficulté)
    # ============================================================================
    # Score = Σ_intervalles (Delta_EMD × Difficulté) / Σ_intervalles (Difficulté)
    numerator = (Delta_EMD_matrix * OT_difficulty_matrix).sum(axis=0)
    denominator = OT_difficulty_matrix.sum(axis=0)
    global_score_per_gene = numerator / (denominator + 1e-16)
    
    fig, ax = plt.subplots(figsize=(16, 6))
    colors = ['green' if s > 0 else 'red' for s in global_score_per_gene]
    ax.bar(range(n_genes), global_score_per_gene, color=colors, edgecolor='black', linewidth=1)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('Score global (pondéré par difficulté)', fontsize=13)
    ax.set_title('Score global par gène (moyenne pondérée sur tous les intervalles)\n'
                 'Vert = Bursty globalement meilleur, Rouge = OT globalement meilleur',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_score_per_gene_weighted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Score global par gène (pondéré) sauvegardé")
    
    # ============================================================================
    # FIGURE 6: PERFORMANCE PAR INTERVALLE (AVEC pondération par difficulté)
    # ============================================================================
    # Pour chaque intervalle: Σ_gènes (Delta_EMD × Difficulté) / Σ_gènes (Difficulté)
    weighted_performance_by_interval = []
    for i in range(n_intervals):
        numerator = (Delta_EMD_matrix[i] * OT_difficulty_matrix[i]).sum()
        denominator = OT_difficulty_matrix[i].sum()
        weighted_avg = numerator / (denominator + 1e-16)
        weighted_performance_by_interval.append(weighted_avg)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_bar = ['green' if p > 0 else 'red' for p in weighted_performance_by_interval]
    
    ax.bar(range(n_intervals), weighted_performance_by_interval, color=colors_bar, 
           edgecolor='black', linewidth=1.5)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=2, label='Équivalence (0)')
    ax.set_xlabel('Intervalles temporels', fontsize=13)
    ax.set_ylabel('Performance (pondérée par difficulté)', fontsize=13)
    ax.set_title('Performance globale de Bursty vs OT par intervalle\n'
                 '(moyenne pondérée par difficulté des gènes)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_intervals))
    ax.set_xticklabels(interval_labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for i, p in enumerate(weighted_performance_by_interval):
        ax.text(i, p + 0.01 * np.sign(p) if p != 0 else 0.01, 
                f'{p:.4f}', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_performance_by_interval_weighted.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Performance par intervalle (pondérée) sauvegardée")
    
    # ============================================================================
    # STATISTIQUES GLOBALES
    # ============================================================================
    
    # Statistiques non pondérées
    Delta_EMD_mean = Delta_EMD_matrix.mean(axis=0)
    Delta_EMD_std = Delta_EMD_matrix.std(axis=0)
    Delta_EMD_median = np.median(Delta_EMD_matrix, axis=0)
    
    # Statistiques par gène
    global_stats = pd.DataFrame({
        'Gene': gene_labels,
        'Delta_EMD_mean': Delta_EMD_mean,
        'Delta_EMD_std': Delta_EMD_std,
        'Delta_EMD_median': Delta_EMD_median,
        'Global_score_weighted': global_score_per_gene,
        'Bursty_better_count': (Delta_EMD_matrix > 0).sum(axis=0),
        'OT_better_count': (Delta_EMD_matrix < 0).sum(axis=0),
        'Avg_difficulty': OT_difficulty_matrix.mean(axis=0)
    })
    
    global_stats = global_stats.sort_values('Global_score_weighted', ascending=False)
    global_stats.to_csv(f'{output_folder}/global_statistics.csv', index=False)
    print(f"  ✓ Statistiques globales sauvegardées")
    
    # Statistiques par intervalle
    interval_stats = pd.DataFrame({
        'Interval': interval_labels,
        'Delta_EMD_mean': Delta_EMD_matrix.mean(axis=1),
        'Weighted_performance': weighted_performance_by_interval,
        'Total_difficulty': OT_difficulty_matrix.sum(axis=1),
        'Bursty_better_genes': (Delta_EMD_matrix > 0).sum(axis=1),
        'OT_better_genes': (Delta_EMD_matrix < 0).sum(axis=1)
    })
    
    interval_stats.to_csv(f'{output_folder}/interval_statistics.csv', index=False)
    print(f"  ✓ Statistiques par intervalle sauvegardées")


def main():
    """
    Pipeline principal d'analyse pour tous les intervalles temporels.
    """
    
    print("="*70)
    print("VALIDATION DU MODÈLE BURSTY - ANALYSE MULTI-INTERVALLES")
    print("="*70)
    print(f"\nNombre d'intervalles à analyser: {len(TIME_INTERVALS)}")
    for interval in TIME_INTERVALS:
        print(f"  - {interval['folder']}: {interval['t1']}h → {interval['t2']}h → {interval['t3']}h")
    print()
    
    # Analyse de chaque intervalle
    all_results = []
    
    for i, interval_config in enumerate(TIME_INTERVALS, 1):
        print(f"\n{'='*70}")
        print(f"INTERVALLE {i}/{len(TIME_INTERVALS)}: {interval_config['folder']}")
        print(f"{'='*70}")
        
        try:
            results = analyze_single_interval(interval_config, verbose=True)
            all_results.append(results)
            
            # Génération des plots individuels
            print(f"\nGénération des visualisations pour {interval_config['folder']}...")
            generate_interval_plots(results, output_folder='outputs')
            
            # Sauvegarde CSV
            df_interval = pd.DataFrame({
                'Gene': results['gene_labels'],
                'Alpha': results['alpha'],
                'Beta': results['beta'],
                'EMD_Bursty': results['EMD_Bursty'],
                'EMD_OT': results['EMD_OT'],
                'Delta_EMD': results['Delta_EMD'],
                'OT_difficulty': results['OT_difficulty']
            })
            df_interval.to_csv(f"outputs/results_{interval_config['folder']}.csv", index=False)
            
            print(f"✓ Intervalle {interval_config['folder']} terminé avec succès")
            
        except Exception as e:
            print(f"✗ ERREUR lors de l'analyse de {interval_config['folder']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Génération de l'analyse globale
    if len(all_results) > 0:
        generate_global_analysis(all_results, output_folder='outputs')
        
        # Résumé final
        print(f"\n{'='*70}")
        print("ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"{'='*70}\n")
        
        print(f"RÉSUMÉ GLOBAL:")
        print(f"  - Intervalles analysés: {len(all_results)}/{len(TIME_INTERVALS)}")
        print(f"  - Nombre de gènes: {all_results[0]['n_genes']}")
        print()
        
        print("Performance par intervalle:")
        print("  (1) Moyenne simple Delta_EMD | (2) Moyenne pondérée par difficulté")
        for results in all_results:
            simple_avg = results['Delta_EMD'].mean()
            weighted_avg = (results['Delta_EMD'] * results['OT_difficulty']).sum() / (1e-16 + results['OT_difficulty'].sum())
            n_better = (results['Delta_EMD'] > 0).sum()
            n_total = results['n_genes']
            print(f"  {results['folder']}: (1) {simple_avg:+.6f} | (2) {weighted_avg:+.6f} | "
                  f"Bursty meilleur: {n_better}/{n_total} gènes")
        
        print(f"\nTous les résultats sont sauvegardés dans le dossier 'outputs/'")
    
    else:
        print("\n✗ Aucune analyse n'a réussi. Vérifiez vos données et les messages d'erreur.")
    
    return all_results


if __name__ == "__main__":
    all_results = main()