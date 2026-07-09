"""
Comprehensive scRNA-seq Analysis for NACI2026
==============================================
- Cell clustering & annotation
- Immune cell proportions per sample
- Immune checkpoint, cytokine & chemokine expression
- Beautiful publication-ready figures
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, warnings, itertools
from scipy.io import mmread
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch

warnings.filterwarnings('ignore')

# ============================================================
# Global settings – publication-quality figures
# ============================================================
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'font.size': 10,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

RESULTS = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/results'
PLOTS = os.path.join(RESULTS, 'plots')
DATA = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/data/matrix'
os.makedirs(PLOTS, exist_ok=True)

# Custom color palette
SAMPLE_COLORS = {'ZLF': '#4C72B0', 'ZFL': '#DD8452', 'HJX': '#55A868'}
CELLTYPE_COLORS = {
    'T_cells':      '#E74C3C',
    'B_Plasma':     '#3498DB',
    'Myeloid':      '#2ECC71',
    'NK':           '#9B59B6',
    'Epithelial':   '#F39C12',
    'Fibroblasts':  '#1ABC9C',
    'Endothelial':  '#E67E22',
    'Mast':         '#E91E63',
}

# ============================================================
# 1. Load raw 10X data
# ============================================================
def load_manual_10x(path):
    mtx_path = os.path.join(path, 'matrix.mtx.gz')
    bc_path  = os.path.join(path, 'barcodes.tsv.gz')
    feat_path = os.path.join(path, 'features.tsv.gz')
    X = mmread(mtx_path).T.tocsr()
    barcodes = pd.read_csv(bc_path, header=None)[0].values
    features_df = pd.read_csv(feat_path, header=None, sep='\t')
    gene_names = features_df[0].values
    adata = sc.AnnData(X=X)
    adata.obs_names = [f"{b}_{os.path.basename(path)}" for b in barcodes]
    adata.var_names = gene_names
    return adata

samples = ['ZLF', 'ZFL', 'HJX']
print("Loading data...")
adatas = []
for s in samples:
    a = load_manual_10x(os.path.join(DATA, s))
    a.obs['sample'] = s
    a.var_names_make_unique()
    adatas.append(a)
adata = sc.concat(adatas, index_unique='_')
print(f"Combined shape: {adata.shape}")

# ============================================================
# 2. QC & Filtering
# ============================================================
print("\nRunning QC...")
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)

fig, axes = plt.subplots(1, 3, figsize=(10, 4))
for i, k in enumerate(['n_genes_by_counts', 'total_counts', 'pct_counts_mt']):
    for s in samples:
        subset = adata[adata.obs['sample'] == s]
        axes[i].hist(subset.obs[k], bins=60, alpha=0.5, label=s, color=SAMPLE_COLORS[s])
    axes[i].set_xlabel(k)
    axes[i].set_ylabel('Frequency')
    axes[i].legend(frameon=False)
fig.suptitle('Pre-filtering QC Metrics', fontsize=14, y=1.02)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'qc_histograms.png'))
plt.close()

# Filter
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20].copy()
adata = adata[adata.obs.total_counts < 40000].copy()
print(f"After filtering: {adata.shape}")

# QC stats per sample
qc_stats = adata.obs.groupby('sample').agg(
    CellCount=('n_genes_by_counts', 'count'),
    MeanGenes=('n_genes_by_counts', 'mean')
).round(2)
print(qc_stats)
qc_stats.to_csv(os.path.join(RESULTS, 'qc_statistics.txt'), sep='\t')

# ============================================================
# 3. Normalization, HVG, PCA, UMAP, Clustering
# ============================================================
print("\nNormalizing and scaling...")
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, min_mean=0.0125, max_mean=3, min_disp=0.5)

# Scale & PCA
adata_hvg = adata[:, adata.var.highly_variable].copy()
sc.pp.scale(adata_hvg, max_value=10)
sc.tl.pca(adata_hvg, svd_solver='arpack', n_comps=50)
sc.pp.neighbors(adata_hvg, n_neighbors=15, n_pcs=30)
sc.tl.umap(adata_hvg, min_dist=0.3, spread=1.0)
sc.tl.leiden(adata_hvg, resolution=0.8)

# Copy results back
adata.obsm['X_pca'] = adata_hvg.obsm['X_pca']
adata.obsm['X_umap'] = adata_hvg.obsm['X_umap']
adata.obs['leiden'] = adata_hvg.obs['leiden'].values

# ============================================================
# 4. Cell-type annotation
# ============================================================
print("\nAnnotating cell types...")

# Known marker genes
marker_genes = {
    'Epithelial':   ['EPCAM', 'KRT19', 'KRT18', 'KRT8', 'CDH1'],
    'T_cells':      ['CD3D', 'CD3E', 'CD3G', 'CD2'],
    'NK':           ['NKG7', 'GNLY', 'KLRD1', 'KLRB1', 'GZMB', 'GZMK'],
    'Myeloid':      ['CD14', 'CD68', 'LYZ', 'FCGR3A', 'CSF1R', 'ITGAM'],
    'B_Plasma':     ['CD79A', 'CD79B', 'MS4A1', 'JCHAIN', 'MZB1', 'SDC1'],
    'Fibroblasts':  ['COL1A1', 'COL1A2', 'DCN', 'LUM', 'FAP'],
    'Endothelial':  ['PECAM1', 'VWF', 'CDH5', 'ENG', 'FLT1'],
    'Mast':         ['KIT', 'TPSAB1', 'TPSB2', 'CPA3'],
}

# Score each cluster for each cell type
from collections import Counter
cluster_types = {}
for cluster in sorted(adata.obs['leiden'].unique()):
    mask = adata.obs['leiden'] == cluster
    sub = adata[mask]
    scores = {}
    for ct, genes in marker_genes.items():
        present = [g for g in genes if g in sub.var_names]
        if present:
            scores[ct] = sub[:, present].X.mean()
        else:
            scores[ct] = 0
    best = max(scores, key=scores.get)
    cluster_types[cluster] = best

cluster_map = pd.DataFrame.from_dict(cluster_types, orient='index',
                                      columns=['cell_type'])
cluster_map.index.name = 'leiden'
cluster_map.to_csv(os.path.join(RESULTS, 'cluster_annotation_map.csv'))
print("Cluster annotation map:")
print(cluster_map)

adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_types)
# Merge NK into T_cells if low count
ct_counts = adata.obs['cell_type'].value_counts()
print("Cell type counts:", ct_counts)

# ============================================================
# 5. UMAP plots
# ============================================================
print("\nGenerating UMAP plots...")

def umap_with_style(adata, color, palette, title, fname, 
                    legend_loc='right margin', w=6, h=5):
    fig, ax = plt.subplots(figsize=(w, h))
    sc.pl.umap(adata, color=color, palette=palette, ax=ax,
               show=False, frameon=False, legend_loc=legend_loc,
               title=title, s=4)
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname))
    plt.close()

# For leiden, no palette needed
umap_with_style(adata, 'leiden', None, 'Leiden Clusters', 'umap_leiden.png')
umap_with_style(adata, 'cell_type', CELLTYPE_COLORS, 'Cell Types', 'umap_celltype.png')
umap_with_style(adata, 'sample', SAMPLE_COLORS, 'Samples', 'umap_sample.png')

# Per-sample UMAP
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for i, s in enumerate(samples):
    mask = adata.obs['sample'] == s
    sc.pl.umap(adata[mask], color='cell_type', palette=CELLTYPE_COLORS,
               ax=axes[i], show=False, frameon=False, title=s,
               legend_loc=None, s=5)
    axes[i].set_xlabel('UMAP 1')
    axes[i].set_ylabel('UMAP 2')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'umap_per_sample.png'))
plt.close()

# ============================================================
# 6. Immune cell proportions
# ============================================================
print("\nPlotting cell proportions...")

# Stacked bar
prop_df = pd.crosstab(adata.obs['sample'], adata.obs['cell_type'],
                       normalize='index') * 100
prop_df = prop_df[['T_cells', 'NK', 'B_Plasma', 'Myeloid', 'Epithelial',
                    'Fibroblasts', 'Endothelial', 'Mast']
                   if all(c in prop_df.columns for c in 
                          ['T_cells','NK','B_Plasma','Myeloid','Epithelial',
                           'Fibroblasts','Endothelial','Mast'])
                   else sorted(prop_df.columns)]

colors_ct = [CELLTYPE_COLORS.get(c, '#999999') for c in prop_df.columns]

fig, ax = plt.subplots(figsize=(6, 4))
prop_df.plot(kind='bar', stacked=True, ax=ax, color=colors_ct,
             edgecolor='white', linewidth=0.5)
ax.set_ylabel('Proportion (%)')
ax.set_xlabel('')
ax.legend(frameon=False, bbox_to_anchor=(1, 1))
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for container in ax.containers:
    lbls = [f'{v:.1f}%' if v > 5 else '' for v in container.datavalues]
    ax.bar_label(container, labels=lbls, fontsize=7, label_type='center')
plt.title('Cell Type Proportions by Sample', fontsize=13)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'proportions_stacked_bar.png'))
plt.close()

# Heatmap of proportions
fig, ax = plt.subplots(figsize=(4, 3))
sns.heatmap(prop_df, annot=True, fmt='.1f', cmap='YlOrRd',
            linewidths=0.5, ax=ax, cbar_kws={'label': '%'})
ax.set_ylabel('Sample')
ax.set_xlabel('')
plt.title('Cell Type Proportions (%)', fontsize=12)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'proportions_heatmap.png'))
plt.close()

# ============================================================
# 7. Immune Checkpoints, Cytokines, Chemokines
# ============================================================
print("\nPlotting immune checkpoint / cytokine / chemokine expression...")

# Gene lists
CHECKPOINTS = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'BTLA',
               'CD274', 'PDCD1LG2', 'ICOS', 'VSIR', 'CD276', 'VTCN1',
               'CD80', 'CD86', 'TNFSF4', 'TNFRSF4', 'TNFSF9', 'TNFRSF9']

CYTOKINES = ['IFNG', 'TNF', 'IL2', 'IL4', 'IL5', 'IL6', 'IL7', 'IL10',
             'IL12A', 'IL12B', 'IL13', 'IL15', 'IL17A', 'IL18', 'IL21',
             'IL23A', 'TGFB1', 'CSF2', 'CSF1', 'CSF3']

CHEMOKINES = ['CCL2', 'CCL3', 'CCL4', 'CCL5', 'CCL8', 'CCL17', 'CCL19',
              'CCL20', 'CCL21', 'CCL22', 'CXCL1', 'CXCL2', 'CXCL8',
              'CXCL9', 'CXCL10', 'CXCL11', 'CXCL12', 'CXCL13', 'CXCL16',
              'XCL1', 'XCL2', 'CX3CL1']

def plot_dotplot(gene_list, title, fname, figsize=(10, 5)):
    valid = [g for g in gene_list if g in adata.var_names]
    if not valid:
        print(f"  No genes found for {title}")
        return
    fig, ax = plt.subplots(figsize=figsize)
    sc.pl.dotplot(adata, valid, groupby='cell_type',
                  standard_scale='var', ax=ax,
                  show=False, dendrogram=False)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, fname))
    plt.close()
    print(f"  {title}: {len(valid)}/{len(gene_list)} genes plotted")

plot_dotplot(CHECKPOINTS, 'Immune Checkpoints', 'dotplot_checkpoints.png', (12, 4.5))
plot_dotplot(CYTOKINES, 'Cytokines', 'dotplot_cytokines.png', (12, 4.5))
plot_dotplot(CHEMOKINES, 'Chemokines', 'dotplot_chemokines.png', (12, 4.5))

# ============================================================
# 8. Violin plots for key checkpoints
# ============================================================
key_checkpoints = ['PDCD1', 'CTLA4', 'LAG3', 'HAVCR2', 'TIGIT', 'CD274']
valid_cp = [g for g in key_checkpoints if g in adata.var_names]
if valid_cp:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    axes = axes.flatten()
    for i, gene in enumerate(valid_cp):
        sc.pl.violin(adata, gene, groupby='cell_type', ax=axes[i],
                     show=False, rotation=45)
        axes[i].set_title(gene, fontweight='bold')
        axes[i].set_ylabel('')
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    plt.suptitle('Key Immune Checkpoint Expression', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'violin_checkpoints.png'))
    plt.close()

# ============================================================
# 9. Combined heatmap of immune genes across cell types
# ============================================================
immune_genes = (CHECKPOINTS + CYTOKINES + CHEMOKINES)
valid_ig = [g for g in immune_genes if g in adata.var_names]
if valid_ig:
    import scipy.sparse as sp
    mat = adata[:, valid_ig].X
    if sp.issparse(mat):
        mat = mat.toarray()
    expr_df = pd.DataFrame(mat, index=adata.obs_names, columns=valid_ig)
    expr_df['cell_type'] = adata.obs['cell_type'].values
    mean_expr = expr_df.groupby('cell_type').mean()

    # Z-score normalization
    zscore = lambda x: (x - x.mean()) / x.std()
    mean_expr_z = mean_expr.apply(zscore, axis=0)

    n_genes = len(valid_ig)
    fig, ax = plt.subplots(figsize=(max(8, n_genes * 0.35), 5))
    sns.heatmap(mean_expr_z, cmap='vlag', center=0,
                linewidths=0.5, linecolor='white',
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Z-score', 'shrink': 0.7},
                ax=ax)
    ax.set_xlabel('')
    ax.set_ylabel('')
    plt.title('Immune Gene Expression Across Cell Types', fontsize=13,
              fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'heatmap_immune_genes.png'))
    plt.close()
    print(f"  Heatmap: {len(valid_ig)} genes plotted")

# ============================================================
# 10. Summary figure: multi-panel immune landscape
# ============================================================
fig = plt.figure(figsize=(18, 14))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.25)

# A: UMAP cell types
ax0 = fig.add_subplot(gs[0, 0])
sc.pl.umap(adata, color='cell_type', palette=CELLTYPE_COLORS,
           ax=ax0, show=False, frameon=False, title='A  Cell Types',
           legend_loc=None, s=3)
ax0.set_xlabel('UMAP 1'); ax0.set_ylabel('UMAP 2')

# B: UMAP samples
ax1 = fig.add_subplot(gs[0, 1])
sc.pl.umap(adata, color='sample', palette=SAMPLE_COLORS,
           ax=ax1, show=False, frameon=False, title='B  Samples',
           legend_loc=None, s=3)
ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2')

# C: Proportions
ax2 = fig.add_subplot(gs[0, 2])
prop_df.plot(kind='bar', stacked=True, ax=ax2, color=colors_ct,
             edgecolor='white', linewidth=0.5, legend=False)
ax2.set_ylabel('Proportion (%)')
ax2.set_xlabel('')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.set_title('C  Cell Proportions', fontsize=12, fontweight='bold')

# D: Checkpoint dotplot
ax3 = fig.add_subplot(gs[1, 0])
valid_ck = [g for g in ['PDCD1','CTLA4','LAG3','HAVCR2','TIGIT','BTLA',
                         'CD274','ICOS','CD80','CD86'] if g in adata.var_names]
if valid_ck:
    sc.pl.dotplot(adata, valid_ck, groupby='cell_type',
                  standard_scale='var', ax=ax3,
                  show=False, dendrogram=False)
    ax3.set_title('D  Immune Checkpoints', fontsize=12, fontweight='bold')

# E: Cytokine dotplot
ax4 = fig.add_subplot(gs[1, 1])
valid_cy = [g for g in ['IFNG','TNF','IL2','IL4','IL6','IL10','IL17A',
                         'TGFB1','CSF2','IL18'] if g in adata.var_names]
if valid_cy:
    sc.pl.dotplot(adata, valid_cy, groupby='cell_type',
                  standard_scale='var', ax=ax4,
                  show=False, dendrogram=False)
    ax4.set_title('E  Cytokines', fontsize=12, fontweight='bold')

# F: Chemokine dotplot
ax5 = fig.add_subplot(gs[1, 2])
valid_ch = [g for g in ['CCL2','CCL3','CCL4','CCL5','CCL19','CCL20',
                         'CXCL8','CXCL9','CXCL10','CXCL12','CXCL13',
                         'CX3CL1'] if g in adata.var_names]
if valid_ch:
    sc.pl.dotplot(adata, valid_ch, groupby='cell_type',
                  standard_scale='var', ax=ax5,
                  show=False, dendrogram=False)
    ax5.set_title('F  Chemokines', fontsize=12, fontweight='bold')

# G: Expression heatmap
ax6 = fig.add_subplot(gs[2, :])
all_ig = valid_ck + valid_cy + valid_ch
if all_ig:
    mean_expr_sub = expr_df.groupby('cell_type')[all_ig].mean()
    mean_expr_sub_z = mean_expr_sub.apply(zscore, axis=0)
    sns.heatmap(mean_expr_sub_z, cmap='vlag', center=0,
                linewidths=0.3, linecolor='white',
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Z-score', 'shrink': 0.5},
                ax=ax6)
    ax6.set_xlabel('')
    ax6.set_ylabel('')
    ax6.set_title('G  Summary: Immune Gene Expression (Z-score)', 
                   fontsize=12, fontweight='bold')
    ax6.set_xticklabels(ax6.get_xticklabels(), rotation=45, ha='right')

plt.savefig(os.path.join(PLOTS, 'immune_landscape_summary.png'),
            dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print("Analysis complete! All figures saved to:")
print(f"  {PLOTS}")
print(f"{'='*50}")
print("\nGenerated files:")
for f in sorted(os.listdir(PLOTS)):
    print(f"  - {f}")
