"""
Script principal pour l'analyse de validation du modèle Bursty
via interpolation de McCann sur TOUS les intervalles temporels.

Ce script:
1. Analyse chaque intervalle temporel (0_6_12, 12_24_36, etc.)
2. Optimise alpha et beta pour chaque intervalle
3. Compare Bursty vs OT pour chaque intervalle
4. Génère une analyse globale avec boxplots par gène
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path

from optimization import optimize_alpha_beta_complete, mccann_interpolation
from optimal_transport import compute_entropic_ot_coupling, compute_ot_distance

# Paramètres globaux
n_samples = 1
num_iter = 100
blur = 1.0
mode = 'global'


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
    df = pd.read_csv('panel_real.txt', sep='\t')
    genes_names = pd.read_csv('panel_genes.txt', sep='\t')
    B_ref = pd.read_csv(f'{folder}/PDMP_ref_{time_init}_{time_final}.txt', sep=' ', header=None)
    B_sch = pd.read_csv(f'{folder}/PDMP_sch_{time_init}_{time_final}.txt', sep=' ', header=None)
    rho_est = pd.read_csv(f'{folder}/rho_est_t{time_int}.txt', sep=' ', header=None)
    
    # Conversion en arrays numpy
    B_ref = np.array(B_ref)
    B_sch = np.array(B_sch)
    rho_est = np.array(rho_est)
    
    # Trier par timegap
    groupes = defaultdict(list)
    
    for col in df.columns:
        base_name = str(col).split('.')[0]
        groupes[base_name].append(col)
    
    data_t = {nom: df[cols] for nom, cols in groupes.items()}
    
    for nom, cols in groupes.items():
        data_t[f"data{nom}"] = df[cols]
    
    # Extraction des données aux temps clés (en évitant le stimulus)
    data_t1 = np.array(data_t[f'data{time_init}'])[1:]
    data_t2 = np.array(data_t[f'data{time_int}'])[1:]
    data_t3 = np.array(data_t[f'data{time_final}'])[1:]
    
    return {
        'data_t1': data_t1,
        'data_t2': data_t2,
        'data_t3': data_t3,
        'B_ref': B_ref,
        'B_sch': B_sch,
        'rho_est': data_t2,
        'genes_names': genes_names
    }


def analyze_single_interval(interval_config, n_samples=300, num_iter=1, verbose=True):
    """
    Analyse un seul intervalle temporel.
    
    Parameters:
    -----------
    interval_config : dict
        Configuration de l'intervalle avec 'folder', 't1', 't2', 't3'
    n_samples : int
        Nombre d'échantillons pour l'interpolation
    num_iter : int
        Nombre d'itérations d'optimisation
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
    
    data_t1 = data['data_t1'].T 
    data_t2 = data['data_t2'].T 
    data_t3 = data['data_t3'].T 
    B_ref = data['B_ref']
    B_sch = data['B_sch']
    rho_est = data['rho_est'].T
    genes_names = data['genes_names']
    
    n_genes = data_t1.shape[1]
    
    if verbose:
        print(f"Nombre de gènes: {n_genes}")
        print(f"Cellules: t1={data_t1.shape[0]}, t2={data_t2.shape[0]}, t3={data_t3.shape[0]}")
    
    # Optimisation per-gene
    if verbose:
        print(f"\nOptimisation per-gene...")
    
    alpha_opt_per_gene, beta_opt_per_gene = optimize_alpha_beta_complete(
        data_t1=data_t1,
        data_t3=data_t3,
        B_ref=B_ref,
        rho_ref=rho_est[np.random.choice(rho_est.shape[0], size=n_samples * B_ref.shape[0]), :],
        n_samples=n_samples,
        n_iterations=num_iter,
        lr=0.05,
        blur=blur,
        verbose=verbose,
        mode=mode,
        constrain=True,
        n_jobs=-1  # Parallélisations si mode == 'per_gene'
    )
    
    # Interpolation Bursty
    rho_bursty = mccann_interpolation(
        data_t1=data_t1,
        data_t3=data_t3,
        coupling=B_sch,
        alpha=alpha_opt_per_gene,
        beta=beta_opt_per_gene,
        n_samples=n_samples
    )
    
    # Couplage OT classique
    OT_coupling = compute_entropic_ot_coupling(
        data_t1=data_t1,
        data_t3=data_t3,
        epsilon=blur**2, 
        numItermax=10000
    )
    
    # Interpolation OT
    rho_OT = mccann_interpolation(
        data_t1=data_t1,
        data_t3=data_t3,
        coupling=OT_coupling,
        alpha=np.array([0.5]),
        beta=np.array([0.5]),
        n_samples=n_samples
    )
    
    # Calcul des EMD par gène
    EMD_Bursty = np.zeros(n_genes)
    EMD_OT = np.zeros(n_genes)
    
    for gene_idx in range(n_genes):
        EMD_Bursty[gene_idx] = compute_ot_distance(
            rho_bursty[:, gene_idx:gene_idx+1], 
            data_t2[:, gene_idx:gene_idx+1]
        )
        EMD_OT[gene_idx] = compute_ot_distance(
            rho_OT[:, gene_idx:gene_idx+1], 
            data_t2[:, gene_idx:gene_idx+1]
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
        'alpha_per_gene': alpha_opt_per_gene,
        'beta_per_gene': beta_opt_per_gene,
        'EMD_Bursty': EMD_Bursty,
        'EMD_OT': EMD_OT,
        'Delta_EMD': Delta_EMD,
        'n_genes': n_genes
    }


def generate_interval_plots(results, output_folder='outputs'):
    """
    Génère les plots individuels pour chaque intervalle.
    """
    
    Path(output_folder).mkdir(exist_ok=True)
    folder = results['folder']
    gene_labels = results['gene_labels']
    Delta_EMD = results['Delta_EMD']
    EMD_Bursty = results['EMD_Bursty']
    EMD_OT = results['EMD_OT']
    n_genes = results['n_genes']
    
    # Plot Delta_EMD
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ['green' if d > 0 else 'red' for d in Delta_EMD]
    ax.bar(range(n_genes), Delta_EMD, color=colors, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('$\\Delta_{EMD} = EMD_{OT} - EMD_{Bursty}$', fontsize=12)
    ax.set_title(f'Delta EMD pour {folder}\n(Vert = Bursty meilleur, Rouge = Bursty pire)', fontsize=13)
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/delta_emd_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Plot EMD comparison
    fig, ax = plt.subplots(figsize=(14, 5))
    x = np.arange(n_genes)
    width = 0.35
    ax.bar(x - width/2, EMD_Bursty, width, label='Bursty', color='blue', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.bar(x + width/2, EMD_OT, width, label='OT', color='orange', alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Genes', fontsize=12)
    ax.set_ylabel('EMD', fontsize=12)
    ax.set_title(f'Comparaison des EMD pour {folder}', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=8)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_folder}/emd_comparison_{folder}.png', dpi=300, bbox_inches='tight')
    plt.close()


def generate_global_analysis(all_results, output_folder='outputs'):
    """
    Génère l'analyse globale avec boxplots pour tous les intervalles.
    
    Parameters:
    -----------
    all_results : list of dict
        Liste des résultats pour chaque intervalle
    output_folder : str
        Dossier de sortie
    """
    
    Path(output_folder).mkdir(exist_ok=True)
    
    # Récupération des noms de gènes (identiques pour tous les intervalles)
    gene_labels = all_results[0]['gene_labels']
    n_genes = all_results[0]['n_genes']
    n_intervals = len(all_results)
    
    # Construction de la matrice Delta_EMD: (n_intervals, n_genes)
    Delta_EMD_matrix = np.array([r['Delta_EMD'] for r in all_results])
    
    # Figure principale: Boxplots par gène
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Préparation des données pour boxplot
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
    
    # Coloration des boxplots selon la médiane
    for i, box in enumerate(bp['boxes']):
        median_val = np.median(Delta_EMD_matrix[:, i])
        if median_val > 0:
            box.set_facecolor('lightgreen')
        else:
            box.set_facecolor('lightcoral')
    
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.5, zorder=1)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('$\\Delta_{EMD} = EMD_{OT} - EMD_{Bursty}$', fontsize=13)
    ax.set_title(f'Distribution de $\\Delta_{{EMD}}$ par gène sur {n_intervals} intervalles temporels\n'
                 f'(Vert = médiane positive, Rouge = médiane négative)', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.grid(axis='y', alpha=0.3, zorder=0)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_delta_emd_boxplots.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - Boxplots globaux sauvegardés: global_delta_emd_boxplots.png")
    
    # Figure complémentaire: Heatmap des Delta_EMD
    fig, ax = plt.subplots(figsize=(16, 6))
    
    interval_labels = [r['interval'] for r in all_results]
    
    im = ax.imshow(Delta_EMD_matrix, aspect='auto', cmap='RdYlGn', 
                   vmin=-np.abs(Delta_EMD_matrix).max(), 
                   vmax=np.abs(Delta_EMD_matrix).max())
    
    ax.set_xticks(range(n_genes))
    ax.set_xticklabels(gene_labels, rotation=90, fontsize=9)
    ax.set_yticks(range(n_intervals))
    ax.set_yticklabels(interval_labels, fontsize=11)
    ax.set_xlabel('Genes', fontsize=13)
    ax.set_ylabel('Intervalles temporels', fontsize=13)
    ax.set_title('Heatmap des $\\Delta_{EMD}$ (tous intervalles)', fontsize=14, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('$\\Delta_{EMD}$', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_delta_emd_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - Heatmap globale sauvegardée: global_delta_emd_heatmap.png")
    
    # Statistiques globales par gène
    Delta_EMD_mean = Delta_EMD_matrix.mean(axis=0)
    Delta_EMD_std = Delta_EMD_matrix.std(axis=0)
    Delta_EMD_median = np.median(Delta_EMD_matrix, axis=0)
    
    global_stats = pd.DataFrame({
        'Gene': gene_labels,
        'Delta_EMD_mean': Delta_EMD_mean,
        'Delta_EMD_std': Delta_EMD_std,
        'Delta_EMD_median': Delta_EMD_median,
        'Bursty_better_count': (Delta_EMD_matrix > 0).sum(axis=0),
        'OT_better_count': (Delta_EMD_matrix < 0).sum(axis=0)
    })
    
    global_stats.to_csv(f'{output_folder}/global_statistics.csv', index=False)
    print(f"  - Statistiques globales sauvegardées: global_statistics.csv")
    
    # Summary plot: Proportion de gènes où Bursty est meilleur par intervalle
    fig, ax = plt.subplots(figsize=(10, 6))
    
    proportions = [(r['Delta_EMD'] > 0).sum() / n_genes * 100 for r in all_results]
    colors_bar = ['green' if p > 50 else 'red' for p in proportions]
    
    ax.bar(range(n_intervals), proportions, color=colors_bar, edgecolor='black', linewidth=1.5)
    ax.axhline(y=50, color='black', linestyle='--', linewidth=2, label='50% (équivalence)')
    ax.set_xlabel('Intervalles temporels', fontsize=13)
    ax.set_ylabel('% de gènes où Bursty est meilleur', fontsize=13)
    ax.set_title('Performance de Bursty vs OT par intervalle temporel', fontsize=14, fontweight='bold')
    ax.set_xticks(range(n_intervals))
    ax.set_xticklabels(interval_labels, fontsize=11)
    ax.set_ylim([0, 100])
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    for i, p in enumerate(proportions):
        ax.text(i, p + 2, f'{p:.1f}%', ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{output_folder}/global_performance_by_interval.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  - Performance par intervalle sauvegardée: global_performance_by_interval.png")


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
    all_dataframes = []
    
    for i, interval_config in enumerate(TIME_INTERVALS, 1):
        print(f"\n{'='*70}")
        print(f"INTERVALLE {i}/{len(TIME_INTERVALS)}: {interval_config['folder']}")
        print(f"{'='*70}")
        
        try:
            results = analyze_single_interval(
                interval_config,
                n_samples=n_samples,
                num_iter=num_iter,
                verbose=True
            )
            
            all_results.append(results)
            
            # Génération des plots individuels
            print(f"\nGénération des visualisations pour {interval_config['folder']}...")
            generate_interval_plots(results, output_folder='outputs')
            
            # Sauvegarde des résultats individuels
            df_interval = pd.DataFrame({
                'Gene': results['gene_labels'],
                'Alpha_per_gene': results['alpha_per_gene'],
                'Beta_per_gene': results['beta_per_gene'],
                'EMD_Bursty': results['EMD_Bursty'],
                'EMD_OT': results['EMD_OT'],
                'Delta_EMD': results['Delta_EMD']
            })
            df_interval.to_csv(f"outputs/results_{interval_config['folder']}.csv", index=False)
            all_dataframes.append(df_interval)
            
            print(f"✓ Intervalle {interval_config['folder']} terminé avec succès")
            
        except Exception as e:
            print(f"✗ ERREUR lors de l'analyse de {interval_config['folder']}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Génération de l'analyse globale
    if len(all_results) > 0:
        print(f"\n{'='*70}")
        print("GÉNÉRATION DE L'ANALYSE GLOBALE")
        print(f"{'='*70}\n")
        
        generate_global_analysis(all_results, output_folder='outputs')
        
        # Résumé final
        print(f"\n{'='*70}")
        print("ANALYSE TERMINÉE AVEC SUCCÈS!")
        print(f"{'='*70}\n")
        
        print(f"RÉSUMÉ GLOBAL:")
        print(f"  - Intervalles analysés: {len(all_results)}/{len(TIME_INTERVALS)}")
        print(f"  - Nombre de gènes: {all_results[0]['n_genes']}")
        print()
        
        for results in all_results:
            n_better = (results['Delta_EMD'] > 0).sum()
            n_total = results['n_genes']
            print(f"  {results['folder']}: Bursty meilleur pour {n_better}/{n_total} gènes ({100*n_better/n_total:.1f}%)")
        
        print(f"\nTous les résultats sont sauvegardés dans le dossier 'outputs/'")
    
    else:
        print("\n✗ Aucune analyse n'a réussi. Vérifiez vos données et les messages d'erreur.")
    
    return all_results, all_dataframes


if __name__ == "__main__":
    all_results, all_dataframes = main()
