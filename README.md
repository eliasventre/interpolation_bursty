# Validation du Modèle Bursty par Interpolation de McCann

Ce projet implémente une méthode de validation du modèle Bursty en comparant deux approches d'interpolation temporelle entre distributions cellulaires.

## 📋 Description du Projet

Le projet vise à valider le modèle Bursty en démontrant sa capacité à interpoler avec précision la distribution de cellules à des points temporels intermédiaires (t=6h) à partir des distributions aux extrémités (t=0h et t=12h).

### Deux méthodes comparées:

1. **Modèle Bursty avec optimisation**: 
   - Utilise le couplage mécaniste B_sch (Schrödinger)
   - Optimise les paramètres α et β par gène pour minimiser W₂(T^{α,β} # B_ref, ρ_ref)
   - Formule d'interpolation: α·x + β·y

2. **Transport Optimal classique (type WOT)**:
   - Utilise un couplage OT entropique standard
   - Paramètres fixes: α = β = 0.5 pour tous les gènes
   - Formule d'interpolation: 0.5·x + 0.5·y

## 📁 Structure du Projet

```
.
├── optimization.py          # Module d'optimisation (α, β) avec PyTorch + geomloss
├── optimal_transport.py     # Module de transport optimal entropique classique
├── main_analysis.py         # Script principal d'analyse
└── README.md               # Ce fichier
```

## 🔧 Dépendances

```bash
pip install numpy pandas matplotlib torch geomloss POT
```

- **numpy**: Calculs numériques
- **pandas**: Manipulation de données
- **matplotlib**: Visualisations
- **torch**: Optimisation des paramètres
- **geomloss**: Distance de Wasserstein différentiable
- **POT (Python Optimal Transport)**: Transport optimal

## 📊 Données Requises

Le script s'attend à trouver les fichiers suivants dans le répertoire courant:

- `panel_real.txt`: Données scRNA-seq avec timepoints
- `PDMP_ref_0_12.txt`: Couplage de référence B_ref(0h,12h)
- `PDMP_sch_0_12.txt`: Couplage de Schrödinger B_sch(0h,12h)
- `rho_est_tilde_t6.txt`: Distribution de référence ρ_ref à t=6h
- `../Semrau/Data/panel_genes.txt`: Noms des gènes

## 🚀 Utilisation

### Exécution Simple

```bash
python main_analysis.py
```

### Étapes du Pipeline

1. **Chargement des données** - Lecture et organisation des données temporelles
2. **Optimisation α, β** - Minimisation de W₂ via PyTorch et geomloss
3. **Interpolation Bursty** - Application de T^{α*,β*} sur B_sch
4. **Calcul OT entropique** - Couplage classique entre t=0h et t=12h
5. **Interpolation OT** - Application de T^{0.5,0.5} sur OT_coupling
6. **Calcul EMD** - Earth Mover's Distance pour chaque gène
7. **Visualisations** - Génération des graphiques comparatifs
8. **Sauvegarde** - Export des résultats

## 📈 Sorties du Programme

### Fichiers Générés (dans `/mnt/user-data/outputs/`)

1. **delta_emd_comparison.png**: 
   - Barplot de Δ_EMD = EMD_Bursty - EMD_OT par gène
   - Vert: Bursty pire (Δ > 0)
   - Rouge: Bursty meilleur (Δ < 0)

2. **emd_comparison_bars.png**: 
   - Comparaison côte à côte des EMD pour les deux méthodes
   - Bleu: Bursty, Orange: OT

3. **optimization_loss.png**: 
   - Courbe de convergence de l'optimisation
   - Évolution de la distance de Wasserstein

4. **alpha_beta_optimized.png**: 
   - Distribution des paramètres α et β optimaux par gène
   - Ligne rouge: référence à 0.5 (OT classique)

5. **results_summary.csv**: 
   - Tableau complet avec α, β, EMD_Bursty, EMD_OT, Δ_EMD par gène

## 🔍 Interprétation des Résultats

### Δ_EMD (Delta EMD)

- **Δ_EMD < 0** (rouge): Le modèle Bursty interpole MIEUX que OT pour ce gène
- **Δ_EMD > 0** (vert): OT interpole mieux que Bursty pour ce gène
- **Δ_EMD ≈ 0**: Performance équivalente

### Critère de Succès

Le modèle Bursty est considéré comme validé si:
- Δ_EMD < 0 pour une majorité de gènes
- EMD_Bursty significativement plus faible que EMD_OT en moyenne
- Les paramètres α*, β* s'écartent de 0.5, montrant un gain par l'optimisation

## ⚙️ Paramètres Ajustables

Dans `main_analysis.py`, vous pouvez modifier:

- `n_samples=10000`: Nombre de couples échantillonnés pour interpolation
- `n_iterations=1000`: Itérations d'optimisation PyTorch
- `lr=0.01`: Learning rate
- `blur=0.01`: Régularisation entropique (Sinkhorn)
- `epsilon=0.01`: Régularisation pour OT classique

## 📚 Références Méthodologiques

- **Transport Optimal Entropique**: Cuturi (2013), Sinkhorn divergences
- **Interpolation de McCann**: McCann (1997), displacement interpolation
- **WOT**: Schiebinger et al. (2019), Waddington-OT
- **Modèle Bursty**: Ventre et al. (2023), CARDAMOM + PDMP

## 🐛 Troubleshooting

### Erreur: "FileNotFoundError"
→ Vérifiez que tous les fichiers de données sont dans le bon répertoire

### Erreur: "CUDA out of memory"
→ Réduisez `n_samples` ou utilisez CPU: `torch.set_default_device('cpu')`

### Loss ne converge pas
→ Ajustez `lr` (essayez 0.001 ou 0.1) ou augmentez `blur`

### EMD très élevées
→ Vérifiez l'échelle des données (normalisation?) et la correspondance temporelle

## 👥 Auteurs

Clémence Fournié - Janvier 2026

## 📝 Notes

- Le stimulus est exclu des données (ligne 1 retirée)
- Les couplages B_ref et B_sch sont supposés être des matrices de probabilités
- La régularisation entropique est maintenue faible pour rester proche de l'OT exact
- L'optimisation se fait globalement sur tous les gènes simultanément
