"""
Ligand-Receptor Interaction Analysis across 3 scRNA-seq samples
===============================================================
Identifies significantly activated cytokine/chemokine/checkpoint
ligand-receptor pairs within each sample's microenvironment.
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import networkx as nx
import os, warnings, itertools
from scipy.io import mmread
from scipy.stats import mannwhitneyu
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.1,
})

RESULTS = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/results'
PLOTS = os.path.join(RESULTS, 'plots')
DATA = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/data/matrix'
os.makedirs(PLOTS, exist_ok=True)

# ================================================================
# C U R A T E D   L I G A N D - R E C E P T O R   D A T A B A S E
# ================================================================
# Chemokine & Chemokine Receptor pairs
LR_PAIRS = [
    # CCL family
    ('CCL2',  'CCR2'), ('CCL3',  'CCR1'), ('CCL3',  'CCR5'),
    ('CCL4',  'CCR5'), ('CCL4',  'CCR1'), ('CCL5',  'CCR1'),
    ('CCL5',  'CCR3'), ('CCL5',  'CCR4'), ('CCL5',  'CCR5'),
    ('CCL8',  'CCR2'), ('CCL17', 'CCR4'), ('CCL19', 'CCR7'),
    ('CCL20', 'CCR6'), ('CCL21', 'CCR7'), ('CCL22', 'CCR4'),
    # CXCL family
    ('CXCL1', 'CXCR2'), ('CXCL2', 'CXCR2'), ('CXCL8', 'CXCR1'),
    ('CXCL8', 'CXCR2'), ('CXCL9', 'CXCR3'), ('CXCL10','CXCR3'),
    ('CXCL11','CXCR3'), ('CXCL12','CXCR4'), ('CXCL13','CXCR5'),
    ('CXCL16','CXCR6'), ('CX3CL1','CX3CR1'), ('XCL1', 'XCR1'),
    ('XCL2',  'XCR1'),
    # TNF Superfamily
    ('TNF',   'TNFRSF1A'),('TNF',   'TNFRSF1B'),
    ('TNFSF4','TNFRSF4'), ('TNFSF9','TNFRSF9'),
    ('TNFSF10','TNFRSF10A'),('TNFSF11','TNFRSF11A'),
    ('TNFSF13','TNFRSF13C'),('TNFSF13B','TNFRSF13C'),
    ('TNFSF14','TNFRSF14'), ('FASLG', 'FAS'),
    # Interleukins
    ('IL1A',  'IL1R1'), ('IL1B',  'IL1R1'), ('IL2',   'IL2RA'),
    ('IL4',   'IL4R'),  ('IL6',   'IL6R'),  ('IL7',   'IL7R'),
    ('IL10',  'IL10RA'),('IL12A', 'IL12RB1'),('IL13',  'IL13RA1'),
    ('IL15',  'IL15RA'),('IL17A', 'IL17RA'), ('IL18',  'IL18R1'),
    ('IL21',  'IL21R'), ('IL23A', 'IL23R'),  ('IL33',  'ST2'),
    # TGFB family
    ('TGFB1', 'TGFBR1'),('TGFB1', 'TGFBR2'),
    ('TGFB2', 'TGFBR1'),('TGFB2', 'TGFBR2'),
    ('BMP2',  'BMPR1A'),('BMP4',  'BMPR1A'),
    # CSF family
    ('CSF1',  'CSF1R'), ('CSF2',  'CSF2RA'),('CSF3',  'CSF3R'),
    # Type I / II interferons
    ('IFNG',  'IFNGR1'),('IFNG',  'IFNGR2'),
    ('IFNA1', 'IFNAR1'),('IFNA1', 'IFNAR2'),
    # Checkpoint ligand-receptor
    ('CD274', 'PDCD1'), ('PDCD1LG2','PDCD1'),
    ('CD80',  'CTLA4'), ('CD86',  'CTLA4'),
    ('CD80',  'CD28'),  ('CD86',  'CD28'),
    ('ICOSLG','ICOS'),  ('VSIR',  'VSIR'),
    ('TNFSF4','TNFRSF4'),
    ('HHLA2', 'TMIGD2'),('NECTIN2','TIGIT'),
    ('PVR',   'TIGIT'), ('LGALS9','HAVCR2'),
    ('CD47',  'SIRPA'), ('CD276', 'VTCN1'),
    ('VEGFA', 'KDR'),   ('VEGFA', 'FLT1'),
    ('VEGFB', 'FLT1'),  ('PGF',   'KDR'),
    ('PDGFA', 'PDGFRA'),('PDGFB', 'PDGFRB'),
    ('EGF',   'EGFR'),  ('HGF',   'MET'),
    ('ANGPT1','TEK'),   ('ANGPT2','TEK'),
]
# Deduplicate
LR_PAIRS = list(dict.fromkeys(LR_PAIRS))

print(f"Loaded {len(LR_PAIRS)} ligand-receptor pairs in database")

# ================================================================
# 1. Load & process data (same pipeline)
# ================================================================
def load_10x(path):
    X = mmread(os.path.join(path, 'matrix.mtx.gz')).T.tocsr()
    barcodes = pd.read_csv(os.path.join(path, 'barcodes.tsv.gz'), header=None)[0].values
    features = pd.read_csv(os.path.join(path, 'features.tsv.gz'), header=None, sep='\t')[0].values
    adata = sc.AnnData(X=X)
    adata.obs_names = [f'{b}_{os.path.basename(path)}' for b in barcodes]
    adata.var_names = features
    adata.obs['sample'] = os.path.basename(path)
    adata.var_names_make_unique()
    return adata

print("Loading data...")
adatas = [load_10x(os.path.join(DATA, s)) for s in ['ZLF','ZFL','HJX']]
adata = sc.concat(adatas, index_unique='_')
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20].copy()
adata = adata[adata.obs.total_counts < 40000].copy()
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)
hvg = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(hvg, max_value=10)
sc.tl.pca(hvg, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(hvg, min_dist=0.3, spread=1.0)
sc.tl.leiden(hvg, resolution=0.8)
adata.obs['leiden'] = hvg.obs['leiden'].values

marker_genes = {
    'Epithelial':  ['EPCAM','KRT19','KRT18','KRT8','CDH1'],
    'T_cells':     ['CD3D','CD3E','CD3G','CD2'],
    'NK':          ['NKG7','GNLY','KLRD1','KLRB1','GZMB','GZMK'],
    'Myeloid':     ['CD14','CD68','LYZ','FCGR3A','CSF1R','ITGAM'],
    'B_Plasma':    ['CD79A','CD79B','MS4A1','JCHAIN','MZB1','SDC1'],
    'Fibroblasts': ['COL1A1','COL1A2','DCN','LUM','FAP'],
    'Endothelial': ['PECAM1','VWF','CDH5','ENG','FLT1'],
    'Mast':        ['KIT','TPSAB1','TPSB2','CPA3'],
}
cluster_types = {}
for cluster in sorted(adata.obs['leiden'].unique()):
    sub = adata[adata.obs['leiden'] == cluster]
    scores = {ct: (sub[:, [g for g in genes if g in sub.var_names]].X.mean()
                    if any(g in sub.var_names for g in genes) else 0)
              for ct, genes in marker_genes.items()}
    cluster_types[cluster] = max(scores, key=scores.get)
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_types)
print(f"Cell types: {adata.obs['cell_type'].value_counts().to_dict()}")

# ================================================================
# 2. Ligand-Receptor scoring function
# ================================================================
def compute_lr_scores(adata, cell_type_col='cell_type', lr_pairs=None):
    """Compute ligand-receptor interaction scores between all cell type pairs.
    Score = mean(ligand_expr_in_sender) * mean(receptor_expr_in_receiver)
    """
    if lr_pairs is None:
        lr_pairs = LR_PAIRS
    ct_types = adata.obs[cell_type_col].unique()
    results = []
    for ligand, receptor in lr_pairs:
        if ligand not in adata.var_names or receptor not in adata.var_names:
            continue
        lig_expr = np.array(adata[:, ligand].X.toarray()).flatten()
        rec_expr = np.array(adata[:, receptor].X.toarray()).flatten()
        for sender in ct_types:
            for receiver in ct_types:
                mask_s = adata.obs[cell_type_col].values == sender
                mask_r = adata.obs[cell_type_col].values == receiver
                mean_l = lig_expr[mask_s].mean()
                mean_r = rec_expr[mask_r].mean()
                if mean_l > 0 and mean_r > 0:
                    score = mean_l * mean_r
                    results.append({
                        'ligand': ligand, 'receptor': receptor,
                        'sender': sender, 'receiver': receiver,
                        'ligand_mean': mean_l, 'receptor_mean': mean_r,
                        'score': score
                    })
    return pd.DataFrame(results)

# ================================================================
# 3. Compute per-sample and overall LR scores
# ================================================================
print("\nComputing ligand-receptor interaction scores...")

all_samples = ['ZLF','ZFL','HJX']
lr_results = {}
for s in all_samples:
    sub = adata[adata.obs['sample'] == s]
    df = compute_lr_scores(sub)
    df['sample'] = s
    lr_results[s] = df
    print(f"  {s}: {df.shape[0]} ligand-receptor pair-cell type interactions")

lr_all = pd.concat(lr_results.values(), ignore_index=True)

# Identify which pairs are detected in each sample
# A pair is "active" if score > 0 in that sample
pair_active = lr_all.groupby(['ligand','receptor','sender','receiver','sample'])[
    'score'].max().unstack(fill_value=0)

# For visualization: find top interactions overall
top_lr = (lr_all.groupby(['ligand','receptor','sender','receiver'])['score']
          .mean().reset_index())
top_lr = top_lr.sort_values('score', ascending=False)

print(f"\nTop 15 ligand-receptor interactions across all samples:")
print(top_lr.head(15).to_string(index=False))

# ================================================================
# 4. Sample-specific significant LR pairs
# ================================================================
# For each pair, rank by score within each sample
# Use top 30 per sample as "significant"
sig_pairs = {}
for s in all_samples:
    sub = lr_results[s]
    # Normalize scores per sample (0-1 within sample)
    sub = sub.copy()
    if sub.shape[0] > 0:
        sub['score_norm'] = (sub['score'] - sub['score'].min()) / (
            sub['score'].max() - sub['score'].min() + 1e-10)
    sig = sub[sub['score_norm'] > 0.5].sort_values('score', ascending=False)
    sig_pairs[s] = sig
    print(f"\n{s} significant LR pairs (score_norm > 0.5): {sig.shape[0]}")

# Find pairs unique to each sample
pair_key = lr_all.groupby(['ligand','receptor','sender','receiver','sample'
    ]).size().reset_index()
pair_key['pair_id'] = (pair_key['ligand'] + '_' + pair_key['receptor'] 
                       + '_' + pair_key['sender'] + '→' + pair_key['receiver'])
sample_pairs = pair_key.groupby('pair_id')['sample'].apply(set)
for s in all_samples:
    unique_in_s = [pid for pid, samps in sample_pairs.items() if samps == {s}]
    print(f"\n  Unique to {s}: {len(unique_in_s)} interactions")
    for pid in unique_in_s[:10]:
        print(f"    {pid}")

# ================================================================
# 5. V I S U A L I Z A T I O N S
# ================================================================
print("\nGenerating visualizations...")

# ---- 5a. Heatmap: top LR pairs across samples ----
# Select top 20 LR pairs (by max score across samples)
top20_pairs = (top_lr.head(30)
               .assign(pair=lambda x: x['ligand'] + '→' + x['receptor'] 
                       + ' (' + x['sender'] + '-' + x['receiver'] + ')'))
pair_names = top20_pairs['pair'].tolist()

# Build matrix: pair × sample
heat_data = []
for _, row in top20_pairs.head(20).iterrows():
    scores_s = []
    for s in all_samples:
        match = lr_results[s][
            (lr_results[s]['ligand']==row['ligand']) &
            (lr_results[s]['receptor']==row['receptor']) &
            (lr_results[s]['sender']==row['sender']) &
            (lr_results[s]['receiver']==row['receiver'])
        ]
        scores_s.append(match['score'].max() if match.shape[0] > 0 else 0)
    heat_data.append(scores_s)

heat_df = pd.DataFrame(heat_data, 
    index=top20_pairs.head(20)['pair'].tolist(),
    columns=all_samples)

# Re-scale rows to 0-1 for fair comparison
heat_df_norm = heat_df.apply(lambda x: (x - x.min()) / (x.max() - x.min() + 1e-10), axis=1)

fig, ax = plt.subplots(figsize=(6, max(8, heat_df.shape[0]*0.4)))
sns.heatmap(heat_df_norm, annot=heat_df.round(2), fmt='.2f',
            cmap='YlOrRd', linewidths=0.5, ax=ax,
            cbar_kws={'label': 'Score (row-normalized)'})
ax.set_xlabel('Sample')
ax.set_ylabel('Ligand→Receptor (Sender→Receiver)')
ax.set_title('Top Ligand-Receptor Interactions by Sample', fontsize=13, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'lr_top_heatmap.png'))
plt.close()

# ---- 5b. Bubble dot plot: selected pairs × cell types per sample ----
# Combined dotplot approach: for key LR pairs, show expression
key_ligands = ['CCL5','CXCL9','CXCL10','CXCL12','CXCL13',
               'CCL2','CCL3','CCL19','CCL20','CCL22',
               'IFNG','TNF','IL1B','IL2','IL10','IL15','IL18','TGFB1',
               'CD274','CD80','CD86','TNFSF4','TNFSF9']
key_receptors = ['CCR5','CXCR3','CXCR4','CXCR5','CCR2','CCR7','CCR1','CCR4',
                 'IFNGR1','IFNGR2','TNFRSF1A','IL1R1','IL2RA','IL10RA','IL15RA',
                 'IL18R1','TGFBR1','TGFBR2','PDCD1','CTLA4','CD28','TNFRSF4']

# Filter valid
k_lig = [g for g in key_ligands if g in adata.var_names]
k_rec = [g for g in key_receptors if g in adata.var_names]
all_k = list(dict.fromkeys(k_lig + k_rec))

fig, axes = plt.subplots(1, 3, figsize=(24, max(6, len(all_k)*0.35)))
for i, s in enumerate(all_samples):
    sub = adata[adata.obs['sample'] == s]
    if len(all_k) > 0:
        sc.pl.dotplot(sub, all_k, groupby='cell_type',
                      standard_scale='var', ax=axes[i],
                      show=False, dendrogram=False)
        axes[i].set_title(f'{s}', fontsize=14, fontweight='bold')
plt.suptitle('Key Ligands & Receptors Expression per Sample', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'lr_expression_dotplot.png'))
plt.close()

# ---- 5c. Network plot per sample ----
for s in all_samples:
    sub = lr_results[s].copy()
    if sub.shape[0] == 0:
        continue
    # Focus on top interactions
    sub = sub.sort_values('score', ascending=False).head(40)

    G = nx.DiGraph()
    # Edge: sender→receiver, labeled by ligand/receptor
    edge_weights = {}
    for _, row in sub.iterrows():
        edge = (row['sender'], row['receiver'])
        lbl = f"{row['ligand']}→{row['receptor']}"
        if edge not in edge_weights:
            edge_weights[edge] = {'weight': 0, 'labels': []}
        edge_weights[edge]['weight'] += row['score']
        edge_weights[edge]['labels'].append(lbl)

    for (u, v), d in edge_weights.items():
        G.add_edge(u, v, weight=d['weight'], labels=d['labels'])

    if G.number_of_edges() == 0:
        continue

    fig, ax = plt.subplots(figsize=(8, 7))
    pos = nx.circular_layout(G)
    weights = [G[u][v]['weight'] for u, v in G.edges()]
    # Normalize weights for edge width
    w_min, w_max = min(weights), max(weights)
    widths = [1 + 4 * (w - w_min) / (w_max - w_min + 1e-10) for w in weights]

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=2000,
                           node_color='#4C72B0', edgecolors='white', linewidths=2)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold',
                            font_color='white')

    # Draw edges with curvature
    for (u, v), w, width in zip(G.edges(), weights, widths):
        rad = 0.15
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], ax=ax,
                               width=width, alpha=0.6,
                               edge_color='#E74C3C',
                               connectionstyle=f'arc3,rad={rad}',
                               arrowsize=15, arrowstyle='->')

    # Add edge labels (show top 2 LR pairs per edge)
    edge_labels = {}
    for (u, v), d in edge_weights.items():
        lbls = d['labels'][:3]  # show top 3
        if len(d['labels']) > 3:
            lbls.append(f"+{len(d['labels'])-3} more")
        edge_labels[(u, v)] = '\n'.join(lbls)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_size=6, alpha=0.8,
                                 label_pos=0.5)

    ax.set_title(f'{s} – LR Interaction Network (top 40)', fontsize=13,
                  fontweight='bold')
    ax.axis('off')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'lr_network_{s}.png'))
    plt.close()

# ---- 5d. Heatmap: sample-specific strength comparison ----
# For the top shared pairs (detected in >=2 samples), compare their scores
pair_sample_matrix = lr_all.pivot_table(
    index=['ligand','receptor','sender','receiver'],
    columns='sample', values='score', aggfunc='max'
).fillna(0)

# Filter to pairs present in at least 2 samples
nz_count = (pair_sample_matrix > 0).sum(axis=1)
shared = pair_sample_matrix[nz_count >= 2]

if shared.shape[0] > 0:
    # Take top 25 shared pairs by mean score
    shared['mean_score'] = shared.mean(axis=1)
    top_shared = shared.sort_values('mean_score', ascending=False).head(25)
    top_shared.index = [f"{lig}→{rec} ({sen}→{rcv})"
                        for lig, rec, sen, rcv in top_shared.index]
    top_shared = top_shared.drop(columns=['mean_score'])

    fig, ax = plt.subplots(figsize=(5, max(6, top_shared.shape[0]*0.35)))
    sns.heatmap(top_shared, annot=True, fmt='.2f',
                cmap='YlOrRd', linewidths=0.5, ax=ax,
                cbar_kws={'label': 'Interaction Score'},
                mask=(top_shared == 0))
    ax.set_xlabel('Sample')
    ax.set_ylabel('Ligand→Receptor (Sender→Receiver)')
    ax.set_title('Shared LR Pairs: Sample Comparison', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'lr_shared_comparison.png'))
    plt.close()

# ---- 5e. Combined summary figure ----
fig = plt.figure(figsize=(20, 14))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# A: Heatmap of top interactions
ax0 = fig.add_subplot(gs[0, :2])
ax0_tmp = sns.heatmap(heat_df_norm, annot=heat_df.round(2), fmt='.2f',
                       cmap='YlOrRd', linewidths=0.5, ax=ax0,
                       cbar_kws={'label': 'Score'})
ax0.set_xlabel('Sample'); ax0.set_ylabel('LR Pair (Sender→Receiver)')
ax0.set_title('A  Top Ligand-Receptor Interactions', fontsize=12, fontweight='bold')

# B: Counts of unique vs shared interactions
ax1 = fig.add_subplot(gs[0, 2])
# Count unique per sample
unique_counts = []
for s in all_samples:
    uniq = len([pid for pid, samps in sample_pairs.items() if samps == {s}])
    unique_counts.append(uniq)
df_bar = pd.DataFrame({'Sample': all_samples, 'Unique LR pairs': unique_counts})
bars = ax1.bar(df_bar['Sample'], df_bar['Unique LR pairs'],
               color=['#4C72B0','#DD8452','#55A868'], edgecolor='white')
ax1.set_ylabel('Count')
ax1.set_title('B  Sample-Unique LR Pairs', fontsize=12, fontweight='bold')
for bar, v in zip(bars, unique_counts):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             str(v), ha='center', fontsize=10, fontweight='bold')

# C-E: Top interactions per sample
for idx, s in enumerate(all_samples):
    ax = fig.add_subplot(gs[1, idx])
    sub = lr_results[s].sort_values('score', ascending=False).head(10)
    if sub.shape[0] > 0:
        sub['label'] = (sub['ligand'] + '→' + sub['receptor'] + '\n' +
                        sub['sender'] + '→' + sub['receiver'])
        colors = sns.color_palette("rocket", n_colors=min(len(sub), 10))
        xpos = range(len(sub))
        ax.barh(list(sub['label'].iloc[::-1]), 
                list(sub['score'].iloc[::-1]),
                color=list(colors)[::-1] if len(colors) >= len(sub) else 'steelblue',
                edgecolor='white')
        ax.set_xlabel('Score')
        ax.set_title(f'{s}  Top 10', fontsize=12, fontweight='bold')
        ax.invert_yaxis()

plt.suptitle('Ligand-Receptor Interaction Analysis', fontsize=16, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'lr_analysis_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

# ================================================================
# 6. Export results table
# ================================================================
lr_all_sorted = lr_all.sort_values(['sample','score'], ascending=[True, False])
lr_all_sorted.to_csv(os.path.join(RESULTS, 'lr_interaction_scores.csv'), index=False)
print(f"\nResults saved to: {os.path.join(RESULTS, 'lr_interaction_scores.csv')}")

# Print top 10 per sample
for s in all_samples:
    print(f"\n{'='*60}")
    print(f"Top 10 LR interactions in {s}:")
    print(f"{'='*60}")
    top = lr_results[s].sort_values('score', ascending=False).head(10)
    for _, row in top.iterrows():
        print(f"  {row['ligand']:8s} → {row['receptor']:10s}  "
              f"{row['sender']:12s}→{row['receiver']:12s}  "
              f"score={row['score']:.4f}")

print(f"\n{'='*50}")
print("LR Analysis Complete! New figures:")
for f in sorted(os.listdir(PLOTS)):
    if f.startswith('lr_'):
        print(f"  - {f}")
print(f"{'='*50}")
