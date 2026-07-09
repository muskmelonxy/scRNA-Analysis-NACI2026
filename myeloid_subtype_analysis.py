"""
Cytokine comparison across samples & Myeloid sub-clustering (M1/M2)
"""
import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import os, warnings
from scipy.io import mmread
warnings.filterwarnings('ignore')

mpl.rcParams.update({
    'font.family': 'sans-serif',
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

SAMPLE_COLORS = {'ZLF':'#4C72B0','ZFL':'#DD8452','HJX':'#55A868'}

# ---------- Load & process ----------
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

# QC & Filter
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20].copy()
adata = adata[adata.obs.total_counts < 40000].copy()

# Normalize & main clustering
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
adata.obsm['X_umap'] = hvg.obsm['X_umap']

# Cell type annotation
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

# ================================================================
# PART 1: Cytokine expression differences across samples
# ================================================================
print("Part 1: Cytokine expression across samples...")

CYTOKINES = ['IFNG','TNF','IL2','IL4','IL5','IL6','IL7','IL10',
             'IL12A','IL12B','IL13','IL15','IL17A','IL18','IL21',
             'IL23A','TGFB1','CSF2','CSF1','CSF3']
valid_cy = [g for g in CYTOKINES if g in adata.var_names]

# Per-sample mean expression
cyto_df = pd.DataFrame(
    (adata[:, valid_cy].X.toarray() if hasattr(adata[:, valid_cy].X, 'toarray')
     else np.array(adata[:, valid_cy].X)),
    columns=valid_cy, index=adata.obs_names
)
cyto_df['sample'] = adata.obs['sample'].values
cyto_mean = cyto_df.groupby('sample')[valid_cy].mean()

# Heatmap
fig, ax = plt.subplots(figsize=(max(6, len(valid_cy)*0.35), 3))
sns.heatmap(cyto_mean, annot=True, fmt='.3f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'label':'Mean Expression'}, ax=ax)
ax.set_title('Cytokine Mean Expression by Sample', fontsize=13, fontweight='bold')
ax.set_ylabel('Sample')
ax.set_xlabel('')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'cytokine_by_sample_heatmap.png'))
plt.close()

# Dot plot per sample
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
for i, s in enumerate(['ZLF','ZFL','HJX']):
    sub = adata[adata.obs['sample'] == s]
    sc.pl.dotplot(sub, valid_cy, groupby='cell_type',
                  standard_scale='var', ax=axes[i],
                  show=False, dendrogram=False)
    axes[i].set_title(f'{s}  (n={sub.n_obs})', fontsize=12, fontweight='bold')
plt.suptitle('Cytokine Expression by Cell Type per Sample', fontsize=14, y=1.05)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'cytokine_per_sample_dotplot.png'))
plt.close()

# ================================================================
# PART 2: Myeloid sub-clustering (M1/M2)
# ================================================================
print("Part 2: Myeloid sub-clustering...")

mye = adata[adata.obs['cell_type'] == 'Myeloid'].copy()
print(f"  Myeloid cells: {mye.n_obs}")

# Re-normalize & cluster within myeloid
sc.pp.normalize_total(mye, target_sum=1e4)
sc.pp.log1p(mye)
sc.pp.highly_variable_genes(mye, min_mean=0.0125, max_mean=3, min_disp=0.5)
mye_hvg = mye[:, mye.var.highly_variable].copy()
sc.pp.scale(mye_hvg, max_value=10)
sc.tl.pca(mye_hvg, svd_solver='arpack', n_comps=30)
sc.pp.neighbors(mye_hvg, n_neighbors=10, n_pcs=15)
sc.tl.umap(mye_hvg, min_dist=0.2, spread=1.0)
sc.tl.leiden(mye_hvg, resolution=0.6)
mye.obs['myeloid_leiden'] = mye_hvg.obs['leiden'].values
mye.obsm['X_umap'] = mye_hvg.obsm['X_umap']

# M1 markers: CD80, CD86, IL1B, TNF, NOS2, CXCL9, CXCL10, CXCL11, STAT1, IRF5
# M2 markers: CD163, CD206(MRC1), IL10, TGFB1, CCL17, CCL22, CSF1R, IRF4, PPARG
# Macrophage core (pan): CD68, CD14, FCGR3A
# Also check CCR5, CCL5

M1_genes = ['CD80','CD86','IL1B','TNF','NOS2','CXCL9','CXCL10','CXCL11','STAT1','IRF5']
M2_genes = ['CD163','MRC1','IL10','TGFB1','CCL17','CCL22','CSF1R','IRF4','PPARG','MSR1']
MACRO_genes = ['CD68','CD14','FCGR3A','ITGAM','CSF1R','LYZ']
TARGET_genes = ['CCR5','CCL5'] + M1_genes + M2_genes + MACRO_genes

# Score M1 / M2
def score_genes_simple(adata, gene_list, score_name):
    valid = [g for g in gene_list if g in adata.var_names]
    if not valid:
        adata.obs[score_name] = 0
        return
    expr = np.array(adata[:, valid].X.mean(axis=1)).flatten() if len(valid) > 1 \
           else np.array(adata[:, valid[0]].X).flatten()
    adata.obs[score_name] = expr

score_genes_simple(mye, M1_genes, 'M1_score')
score_genes_simple(mye, M2_genes, 'M2_score')
score_genes_simple(mye, MACRO_genes, 'macrophage_score')
for g in TARGET_genes:
    if g in mye.var_names:
        mye.obs[g + '_expr'] = np.array(mye[:, g].X.toarray()).flatten()

# Classify clusters as M1 / M2 / unclassified
# Use average M1 vs M2 score per cluster
cluster_m1_m2 = {}
for cl in sorted(mye.obs['myeloid_leiden'].unique()):
    sub = mye[mye.obs['myeloid_leiden'] == cl]
    mean_m1 = sub.obs['M1_score'].mean()
    mean_m2 = sub.obs['M2_score'].mean()
    if mean_m1 > mean_m2 and mean_m1 > 0.1:
        cluster_m1_m2[cl] = 'M1-like'
    elif mean_m2 > mean_m1 and mean_m2 > 0.1:
        cluster_m1_m2[cl] = 'M2-like'
    else:
        cluster_m1_m2[cl] = 'Macrophage (unclassified)'

mye.obs['macrophage_subtype'] = mye.obs['myeloid_leiden'].map(cluster_m1_m2)
subtype_counts = mye.obs['macrophage_subtype'].value_counts()
print("  Macrophage subtype distribution:")
for k, v in subtype_counts.items():
    print(f"    {k}: {v}")

# Also show per-sample
ct_mye = pd.crosstab(mye.obs['sample'], mye.obs['macrophage_subtype'])
print("\n  Per sample:")
print(ct_mye.to_string())

# Subtype UMAP
fig, ax = plt.subplots(figsize=(7, 5.5))
subtype_colors = {'M1-like':'#E74C3C','M2-like':'#3498DB','Macrophage (unclassified)':'#95A5A6'}
sc.pl.umap(mye, color='macrophage_subtype', palette=subtype_colors,
           ax=ax, show=False, frameon=False, title='Myeloid Subtypes (M1/M2)', s=15)
ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'myeloid_subtypes_umap.png'))
plt.close()

# Per-sample subtype proportion
fig, ax = plt.subplots(figsize=(5, 4))
ct_mye_pct = ct_mye.div(ct_mye.sum(axis=1), axis=0) * 100
colors = [subtype_colors.get(c, '#999') for c in ct_mye_pct.columns]
ct_mye_pct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
ax.set_ylabel('Proportion (%)')
ax.set_title('Macrophage Subtypes by Sample', fontweight='bold')
ax.legend(frameon=False)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
plt.tight_layout()
fig.savefig(os.path.join(PLOTS, 'macrophage_subtype_per_sample.png'))
plt.close()

# M1/M2 marker dotplot
mye_genes = [g for g in M1_genes + M2_genes + ['CCR5','CCL5'] if g in mye.var_names]
if mye_genes:
    fig, ax = plt.subplots(figsize=(10, 3.5))
    sc.pl.dotplot(mye, mye_genes, groupby='macrophage_subtype',
                  standard_scale='var', ax=ax, show=False, dendrogram=False)
    ax.set_title('M1/M2 Marker Genes', fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'myeloid_markers_dotplot.png'))
    plt.close()

# ================================================================
# PART 3: CCR5 / CCL5 focused analysis
# ================================================================
print("\nPart 3: CCR5 / CCL5 focused analysis...")

for gene in ['CCR5','CCL5']:
    if gene not in adata.var_names:
        print(f"  {gene} not found in data")
        continue
    # UMAP all cells
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, s in enumerate(['ZLF','ZFL','HJX']):
        sub = adata[adata.obs['sample'] == s].copy()
        sc.pl.umap(sub, color=gene, ax=axes[i], show=False, frameon=False,
                   title=f'{s}: {gene}', color_map='Reds', s=5, vmax='p95')
        axes[i].set_xlabel('UMAP 1'); axes[i].set_ylabel('UMAP 2')
    plt.suptitle(f'{gene} Expression Across Samples', fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'umap_{gene.lower()}_per_sample.png'))
    plt.close()

    # Violin across cell types per sample
    fig, ax = plt.subplots(figsize=(8, 4))
    sc.pl.violin(adata, gene, groupby='cell_type', ax=ax, show=False, rotation=45)
    ax.set_title(f'{gene} by Cell Type', fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, f'violin_{gene.lower()}_celltype.png'))
    plt.close()

# CCR5+CCL5 positive macrophage analysis
ccr5_present = 'CCR5' in adata.var_names
ccl5_present = 'CCL5' in adata.var_names

if ccr5_present or ccl5_present:
    # Find double-positive in myeloid
    mye_obs = mye.obs.copy()
    conditions = []
    labels = []
    if ccr5_present:
        mye_obs['CCR5_pos'] = np.array(mye[:, 'CCR5'].X.toarray()).flatten() > 0
        conditions.append('CCR5_pos')
        labels.append('CCR5')
    if ccl5_present:
        mye_obs['CCL5_pos'] = np.array(mye[:, 'CCL5'].X.toarray()).flatten() > 0
        conditions.append('CCL5_pos')
        labels.append('CCL5')

    if ccr5_present and ccl5_present:
        mye_obs['double_pos'] = mye_obs['CCR5_pos'] & mye_obs['CCL5_pos']
        print(f"\n  CCR5+CCL5+ double-positive macrophages:")
        for s in ['ZLF','ZFL','HJX']:
            n = mye_obs[mye_obs['double_pos'] & (mye_obs['sample'] == s)].shape[0]
            print(f"    {s}: {n} cells")
    # Plot double-positive on UMAP
    mye.obs['CCR5_CCL5_double'] = mye_obs['double_pos'].astype(int).values
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc.pl.umap(mye, color='CCR5_CCL5_double', ax=ax, show=False, frameon=False,
               title='CCR5+CCL5+ Macrophages', s=15, color_map='Reds')
    ax.set_xlabel('UMAP 1'); ax.set_ylabel('UMAP 2')
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'myeloid_ccr5_ccl5_double.png'))
    plt.close()

    # Per-sample CCR5/CCL5 positivity rate in myeloid
    fig, axes = plt.subplots(1, len(labels), figsize=(5*len(labels), 4))
    if len(labels) == 1:
        axes = [axes]
    for ax, col, lb in zip(axes, conditions, labels):
        rates = mye_obs.groupby('sample')[col].mean() * 100
        ax.bar(rates.index, rates.values, color=[SAMPLE_COLORS[s] for s in rates.index], edgecolor='white')
        ax.set_ylabel(f'{lb}+ cells (%)')
        ax.set_title(f'{lb}+ in Myeloid', fontweight='bold')
        for i, v in enumerate(rates.values):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center', fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(PLOTS, 'myeloid_ccr5_ccl5_positivity.png'))
    plt.close()

# ================================================================
# PART 4: Combined summary figure for Myeloid
# ================================================================
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

# A: Myeloid UMAP subtypes
ax0 = fig.add_subplot(gs[0, 0])
sc.pl.umap(mye, color='macrophage_subtype', palette=subtype_colors,
           ax=ax0, show=False, frameon=False, title='A  Myeloid Subtypes',
           legend_loc=None, s=12)
ax0.set_xlabel('UMAP 1'); ax0.set_ylabel('UMAP 2')

# B: Myeloid by sample
ax1 = fig.add_subplot(gs[0, 1])
sc.pl.umap(mye, color='sample', palette=SAMPLE_COLORS,
           ax=ax1, show=False, frameon=False, title='B  Myeloid by Sample',
           legend_loc=None, s=12)
ax1.set_xlabel('UMAP 1'); ax1.set_ylabel('UMAP 2')

# C: Subtype bar
ax2 = fig.add_subplot(gs[0, 2])
ct_mye_pct.plot(kind='bar', stacked=True, ax=ax2, color=[subtype_colors.get(c,'#999') for c in ct_mye_pct.columns],
                edgecolor='white', legend=False)
ax2.set_ylabel('Proportion (%)')
ax2.set_xticklabels(ax2.get_xticklabels(), rotation=0)
ax2.set_title('C  M1/M2 Proportions', fontsize=12, fontweight='bold')

# D: M1/M2 markers dotplot
ax3 = fig.add_subplot(gs[1, 0])
if mye_genes:
    sc.pl.dotplot(mye, mye_genes, groupby='macrophage_subtype',
                  standard_scale='var', ax=ax3, show=False, dendrogram=False)
    ax3.set_title('D  M1/M2 Markers', fontsize=12, fontweight='bold')

# E: CCR5 UMAP in Myeloid
ax4 = fig.add_subplot(gs[1, 1])
if 'CCR5' in mye.var_names:
    sc.pl.umap(mye, color='CCR5', ax=ax4, show=False, frameon=False,
               title='E  CCR5 in Myeloid', color_map='Reds', s=12)
    ax4.set_xlabel('UMAP 1'); ax4.set_ylabel('UMAP 2')

# F: CCL5 UMAP in Myeloid
ax5 = fig.add_subplot(gs[1, 2])
if 'CCL5' in mye.var_names:
    sc.pl.umap(mye, color='CCL5', ax=ax5, show=False, frameon=False,
               title='F  CCL5 in Myeloid', color_map='Reds', s=12)
    ax5.set_xlabel('UMAP 1'); ax5.set_ylabel('UMAP 2')

plt.savefig(os.path.join(PLOTS, 'myeloid_analysis_summary.png'), dpi=300, bbox_inches='tight')
plt.close()

print(f"\n{'='*50}")
print("Analysis complete. New figures:")
for f in sorted(os.listdir(PLOTS)):
    if any(k in f for k in ['cytokine','myeloid','macrophage','ccr5','ccl5']):
        print(f"  - {f}")
print(f"{'='*50}")
