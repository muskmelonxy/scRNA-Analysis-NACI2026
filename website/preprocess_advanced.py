"""
Advanced preprocessing: PAGA/pseudotime + cell-type signature matrix for deconvolution
Run after preprocess_data.py
"""
import scanpy as sc
import pandas as pd
import numpy as np
import os, json, warnings
from scipy.io import mmread
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/data/matrix'
DATA_OUT = os.path.join(BASE, 'data')
os.makedirs(DATA_OUT, exist_ok=True)

# ---- Load the main data from scratch ----
def load_10x(path, name):
    X = mmread(os.path.join(path, 'matrix.mtx.gz')).T.tocsr()
    barcodes = pd.read_csv(os.path.join(path, 'barcodes.tsv.gz'), header=None)[0].values
    features = pd.read_csv(os.path.join(path, 'features.tsv.gz'), header=None, sep='\t')[0].values
    a = sc.AnnData(X=X)
    a.obs_names = [f'{b}_{name}' for b in barcodes]
    a.var_names = features
    a.obs['sample'] = name
    a.var_names_make_unique()
    return a

print("Loading & processing data (same pipeline as preprocess_data.py)...")
adatas = [load_10x(os.path.join(DATA_RAW, s), s) for s in ['ZLF','ZFL','HJX']]
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
adata.obsm['X_umap'] = hvg.obsm['X_umap']

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
for cl in sorted(adata.obs['leiden'].unique()):
    sub = adata[adata.obs['leiden'] == cl]
    scores = {ct: (sub[:,[g for g in genes if g in sub.var_names]].X.mean()
                    if any(g in sub.var_names for g in genes) else 0)
              for ct, genes in marker_genes.items()}
    cluster_types[cl] = max(scores, key=scores.get)
adata.obs['cell_type'] = adata.obs['leiden'].map(cluster_types)

CT_ORDER = ['Epithelial','T_cells','NK','Myeloid','B_Plasma',
            'Fibroblasts','Endothelial','Mast']
CT_ORDER = [c for c in CT_ORDER if c in adata.obs['cell_type'].unique()]

# ================================================================
# 1. Pseudotime: PAGA + DPT
# ================================================================
print("\nComputing PAGA and pseudotime...")

# PAGA on the HVG data
sc.tl.paga(hvg, groups='leiden')
sc.pl.paga(hvg, plot=False)  # just to initialize

# Copy PAGA connectivities to main adata
adata.uns['paga'] = hvg.uns['paga'].copy()

# PAGA connectivities matrix (between clusters)
paga_conn = adata.uns['paga']['connectivities'].toarray()
clusters_ordered = sorted(adata.obs['leiden'].unique().astype(int))
paga_df = pd.DataFrame(paga_conn, index=clusters_ordered, columns=clusters_ordered)
paga_df.to_csv(os.path.join(DATA_OUT, 'paga_connectivity.csv'))
print(f"  PAGA: {paga_df.shape[0]} clusters")

# DPT - root: Epithelial cells (typically the root for carcinoma trajectories)
# Find the cluster with most Epithelial cells
epi_clusters = adata.obs[adata.obs['cell_type'] == 'Epithelial']['leiden'].unique()
if len(epi_clusters) > 0:
    root_cluster = int(max(epi_clusters, key=lambda x: (adata.obs['leiden'] == str(x)).sum()))
    print(f"  Root cluster for DPT: {root_cluster} (Epithelial)")

    # Set root cells and compute DPT
    root_mask = adata.obs['leiden'].astype(str) == str(root_cluster)
    root_idx = np.where(root_mask.values)[0][0]
    hvg.uns['iroot'] = root_idx
    sc.tl.diffmap(hvg, n_comps=15)
    sc.tl.dpt(hvg, n_branchings=0)
    adata.obs['dpt_pseudotime'] = hvg.obs['dpt_pseudotime'].values
    dpt_vals = adata.obs['dpt_pseudotime'].values
    dpt_norm = (dpt_vals - dpt_vals.min()) / (dpt_vals.max() - dpt_vals.min() + 1e-10)
    adata.obs['dpt_pseudotime_norm'] = dpt_norm
else:
    print("  No Epithelial cluster found, skipping DPT")
    adata.obs['dpt_pseudotime'] = 0
    adata.obs['dpt_pseudotime_norm'] = 0

# Save per-cell pseudotime
pt_df = pd.DataFrame({
    'barcode': adata.obs_names,
    'dpt_pseudotime': adata.obs['dpt_pseudotime'].values,
    'dpt_pseudotime_norm': adata.obs['dpt_pseudotime_norm'].values,
})
pt_df.to_csv(os.path.join(DATA_OUT, 'pseudotime.csv.gz'), index=False, compression='gzip')
print(f"  Pseudotime saved for {pt_df.shape[0]} cells")

# ================================================================
# 2. Cell-type gene signature matrix (for bulk deconvolution)
# ================================================================
print("\nComputing cell-type signature matrix for deconvolution...")

# Use all genes for maximum coverage
sig_df_list = []
for ct in CT_ORDER:
    mask = adata.obs['cell_type'].values == ct
    sub = adata[mask]
    mean_expr = np.array(sub.X.mean(axis=0)).flatten()
    pct = np.array((sub.X > 0).mean(axis=0)).flatten()
    # Filter: expressed in > 10% of cells, mean > 0.1
    keep = (pct > 0.1) & (mean_expr > 0.1)
    sig_df_list.append(pd.DataFrame({
        'gene': adata.var_names,
        'mean': mean_expr,
        'pct': pct,
        'cell_type': ct
    }))

sig_all = pd.concat(sig_df_list, ignore_index=True)
# Pivot to get signature matrix (genes × cell types)
sig_pivot = sig_all.pivot_table(index='gene', columns='cell_type', values='mean', fill_value=0)
sig_pivot = sig_pivot[sorted(sig_pivot.columns)]

# Also save marker-only signature (smaller, more robust)
marker_set = set()
for ct, genes in marker_genes.items():
    if ct in CT_ORDER:
        marker_set.update(genes)
marker_set = [g for g in marker_set if g in sig_pivot.index]
sig_markers = sig_pivot.loc[marker_set] if marker_set else sig_pivot

sig_pivot.to_csv(os.path.join(DATA_OUT, 'signature_matrix.csv.gz'), compression='gzip')
sig_markers.to_csv(os.path.join(DATA_OUT, 'signature_matrix_markers.csv.gz'), compression='gzip')
print(f"  Full signature: {sig_pivot.shape[0]} genes × {sig_pivot.shape[1]} cell types")
print(f"  Marker signature: {sig_markers.shape[0]} genes × {sig_markers.shape[1]} cell types")

# ================================================================
# 3. Enhanced LR: interaction counts between cell types
# ================================================================
print("\nComputing cell-cell interaction counts...")

lr_path = os.path.join(DATA_OUT, 'lr_scores.csv')
if os.path.exists(lr_path):
    lr = pd.read_csv(lr_path)
    # Count interactions per sender→receiver pair per sample
    interaction_counts = lr.groupby(['sender', 'receiver', 'sample']).size().reset_index(name='count')
    # Also sum scores
    interaction_strength = lr.groupby(['sender', 'receiver', 'sample'])['score'].sum().reset_index(name='total_strength')
    interaction_summary = interaction_counts.merge(interaction_strength, on=['sender', 'receiver', 'sample'])
    interaction_summary.to_csv(os.path.join(DATA_OUT, 'interaction_summary.csv'), index=False)
    print(f"  {interaction_summary.shape[0]} sender→receiver interactions across samples")
else:
    print("  LR scores not found, skipping")

# ================================================================
# Save metadata about which cell types express which receptors
# ================================================================
# Quick lookup for deconvolution: list available genes
gene_info = pd.DataFrame({
    'gene': adata.var_names,
    'detected_in_pct': [float((adata[:, g].X > 0).mean()) * 100 for g in adata.var_names],
})
gene_info.to_csv(os.path.join(DATA_OUT, 'gene_detection.csv.gz'), index=False, compression='gzip')

print("\nDone! Advanced features data saved to:", DATA_OUT)
print("New files:")
for f in sorted(os.listdir(DATA_OUT)):
    if any(k in f for k in ['paga','pseudotime','signature','interaction','detection']):
        sz = os.path.getsize(os.path.join(DATA_OUT, f))
        print(f"  {f:35s} {sz/1e6:.2f} MB")
