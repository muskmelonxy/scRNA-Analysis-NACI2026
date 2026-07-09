"""
scRNA-seq 数据预处理 — 为可视化网站生成轻量级数据文件
运行方式: python preprocess_data.py
输出目录: data/
"""
import scanpy as sc
import pandas as pd
import numpy as np
import os, json, gzip, warnings
from scipy.io import mmread
warnings.filterwarnings('ignore')

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_RAW = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/data/matrix'
DATA_OUT = os.path.join(BASE, 'data')
os.makedirs(DATA_OUT, exist_ok=True)

# ---- Load ----
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

print("Loading 3 samples...")
samples = ['ZLF', 'ZFL', 'HJX']
adatas = [load_10x(os.path.join(DATA_RAW, s), s) for s in samples]
adata = sc.concat(adatas, index_unique='_')
print(f"  Combined: {adata.shape}")

# ---- QC ----
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(adata, qc_vars=['mt'], percent_top=None, log1p=False, inplace=True)
sc.pp.filter_cells(adata, min_genes=200)
sc.pp.filter_genes(adata, min_cells=3)
adata = adata[adata.obs.pct_counts_mt < 20].copy()
adata = adata[adata.obs.total_counts < 40000].copy()
print(f"  After QC: {adata.shape}")

# ---- Normalize & Cluster ----
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

# ---- Cell-type annotation ----
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

# ================================================================
# File 1: cell_metadata.csv.gz — per-cell UMAP + annotations
# ================================================================
print("Saving cell metadata...")
meta = pd.DataFrame({
    'barcode': adata.obs_names,
    'UMAP_1': adata.obsm['X_umap'][:, 0],
    'UMAP_2': adata.obsm['X_umap'][:, 1],
    'sample': adata.obs['sample'].values,
    'leiden': adata.obs['leiden'].values,
    'cell_type': adata.obs['cell_type'].values,
})
meta.to_csv(os.path.join(DATA_OUT, 'cell_metadata.csv.gz'), index=False, compression='gzip')
print(f"  {meta.shape[0]} cells saved")

# ================================================================
# File 2: gene_list.json — all genes
# ================================================================
gene_list = sorted(adata.var_names.tolist())
with open(os.path.join(DATA_OUT, 'gene_list.json'), 'w') as f:
    json.dump(gene_list, f)
print(f"  {len(gene_list)} genes in list")

# ================================================================
# File 3: immune_genes.json — curated immune-related genes
# ================================================================
IMMUNE_GENES = [
    'PDCD1','CTLA4','LAG3','HAVCR2','TIGIT','BTLA',
    'CD274','PDCD1LG2','ICOS','VSIR','CD276','VTCN1',
    'CD80','CD86','TNFSF4','TNFRSF4','TNFSF9','TNFRSF9',
    'IFNG','TNF','IL2','IL4','IL5','IL6','IL7','IL10',
    'IL12A','IL12B','IL13','IL15','IL17A','IL18','IL21',
    'IL23A','TGFB1','CSF1','CSF2','CSF3',
    'CCL2','CCL3','CCL4','CCL5','CCL8','CCL17','CCL19',
    'CCL20','CCL21','CCL22','CXCL1','CXCL2','CXCL8',
    'CXCL9','CXCL10','CXCL11','CXCL12','CXCL13','CXCL16',
    'XCL1','XCL2','CX3CL1','CCR1','CCR2','CCR3','CCR4',
    'CCR5','CCR6','CCR7','CXCR1','CXCR2','CXCR3','CXCR4',
    'CXCR5','CXCR6','CX3CR1',
    'CD3D','CD3E','CD4','CD8A','CD8B','CD19','CD14','CD68',
    'NCAM1','NKG7','GNLY','CD163','MRC1','NOS2','IL1B',
    'CD40','CD40LG','FAS','FASLG','GZMB','GZMK','PRF1',
    'EPCAM','KRT19','PECAM1','VWF','COL1A1','KIT','JCHAIN',
]
valid_immune = [g for g in IMMUNE_GENES if g in adata.var_names]
with open(os.path.join(DATA_OUT, 'immune_genes.json'), 'w') as f:
    json.dump(valid_immune, f)
print(f"  {len(valid_immune)} immune genes curated")

# ================================================================
# File 4: immune_expr.csv.gz — expression matrix for immune genes
# ================================================================
print("Saving immune gene expression...")
expr = adata[:, valid_immune].X.toarray()
expr_df = pd.DataFrame(expr, index=adata.obs_names, columns=valid_immune)
expr_df.insert(0, 'barcode', adata.obs_names)
expr_df.to_csv(os.path.join(DATA_OUT, 'immune_expr.csv.gz'), index=False, compression='gzip')
print(f"  {expr_df.shape[0]} cells × {expr_df.shape[1]-1} genes")

# ================================================================
# File 5: ct_mean_expr.csv.gz — cell_type × all genes (mean expr)
# ================================================================
print("Saving cell-type mean expression...")
ct_order = ['Epithelial','T_cells','NK','Myeloid','B_Plasma',
            'Fibroblasts','Endothelial','Mast']
ct_order = [c for c in ct_order if c in adata.obs['cell_type'].unique()]
ct_means = []
ct_pcts = []
for ct in ct_order:
    mask = adata.obs['cell_type'].values == ct
    mat = adata[mask].X
    ct_means.append(np.array(mat.mean(axis=0)).flatten())
    ct_pcts.append(np.array((mat > 0).mean(axis=0)).flatten())

mean_df = pd.DataFrame(ct_means, index=ct_order, columns=adata.var_names)
pct_df = pd.DataFrame(ct_pcts, index=ct_order, columns=adata.var_names)
mean_df.to_csv(os.path.join(DATA_OUT, 'ct_mean_expr.csv.gz'), compression='gzip')
pct_df.to_csv(os.path.join(DATA_OUT, 'ct_pct_expr.csv.gz'), compression='gzip')
print(f"  {len(ct_order)} cell types × {mean_df.shape[1]} genes")

# ================================================================
# File 6: proportions.json — cell type × sample counts
# ================================================================
prop = pd.crosstab(adata.obs['sample'], adata.obs['cell_type'])
prop_dict = {s: {ct: int(prop.loc[s, ct]) for ct in prop.columns}
             for s in prop.index}
# Also total per sample
total_per_sample = adata.obs['sample'].value_counts().to_dict()
with open(os.path.join(DATA_OUT, 'proportions.json'), 'w') as f:
    json.dump({'by_sample': prop_dict, 'totals': total_per_sample}, f)
print(f"  Proportions saved")

# ================================================================
# File 7: summary_stats.json
# ================================================================
stats = {
    'total_cells': int(adata.n_obs),
    'total_genes': int(adata.n_vars),
    'samples': samples,
    'cell_types': {ct: int((adata.obs['cell_type']==ct).sum()) for ct in ct_order},
    'per_sample': {
        s: {
            'cells': int((adata.obs['sample']==s).sum()),
            'mean_genes': float(adata[adata.obs['sample']==s].obs['n_genes_by_counts'].mean())
        } for s in samples
    }
}
with open(os.path.join(DATA_OUT, 'summary_stats.json'), 'w') as f:
    json.dump(stats, f, indent=2)
print(f"  Stats saved")

# ================================================================
# File 8: lr_scores.csv.gz (reuse from earlier analysis if exists)
# ================================================================
lr_path = '/Users/yixu/Downloads/NACI2026/scRNA_Analysis/results/lr_interaction_scores.csv'
if os.path.exists(lr_path):
    import shutil
    shutil.copy(lr_path, os.path.join(DATA_OUT, 'lr_scores.csv'))
    print("  LR scores copied")

# ================================================================
# File 9: cluster_annotation_map.json
# ================================================================
cl_map = {str(k): v for k, v in cluster_types.items()}
with open(os.path.join(DATA_OUT, 'cluster_map.json'), 'w') as f:
    json.dump(cl_map, f)

print("\nDone! All data files saved to:", DATA_OUT)
print("\nFiles created:")
for f in sorted(os.listdir(DATA_OUT)):
    sz = os.path.getsize(os.path.join(DATA_OUT, f))
    print(f"  {f:30s} {sz/1e6:.2f} MB")
