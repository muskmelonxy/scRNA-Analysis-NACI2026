# -*- coding: utf-8 -*-
"""
scRNA-seq 可视化探索网站 — Flask 后端
"""
import os, json, io, base64, uuid
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
import scanpy as sc
from flask import (Flask, jsonify, request, send_file, send_from_directory,
                   render_template)

app = Flask(__name__)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE, 'data')
PLOTS_DIR = os.path.join(BASE, 'user_plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 10,
    'figure.dpi': 150,
})

# ── Load precomputed data ─────────────────────────────────
print("Loading precomputed data...")

cell_meta = pd.read_csv(os.path.join(DATA_DIR, 'cell_metadata.csv.gz'))
with open(os.path.join(DATA_DIR, 'gene_list.json')) as f:
    GENE_LIST = json.load(f)
with open(os.path.join(DATA_DIR, 'immune_genes.json')) as f:
    IMMUNE_GENES = json.load(f)
with open(os.path.join(DATA_DIR, 'proportions.json')) as f:
    PROPORTIONS = json.load(f)
with open(os.path.join(DATA_DIR, 'summary_stats.json')) as f:
    SUMMARY = json.load(f)
with open(os.path.join(DATA_DIR, 'cluster_map.json')) as f:
    CLUSTER_MAP = json.load(f)

ct_mean = pd.read_csv(os.path.join(DATA_DIR, 'ct_mean_expr.csv.gz'), index_col=0)
ct_pct  = pd.read_csv(os.path.join(DATA_DIR, 'ct_pct_expr.csv.gz'), index_col=0)
immune_expr = pd.read_csv(os.path.join(DATA_DIR, 'immune_expr.csv.gz'))

lr_df = pd.read_csv(os.path.join(DATA_DIR, 'lr_scores.csv'))

CT_ORDER = ['Epithelial','T_cells','NK','Myeloid','B_Plasma',
            'Fibroblasts','Endothelial','Mast']
CT_ORDER = [c for c in CT_ORDER if c in cell_meta['cell_type'].unique()]
SAMPLE_COLORS = {'ZLF':'#4C72B0','ZFL':'#DD8452','HJX':'#55A868'}
CELLTYPE_COLORS = {
    'Epithelial':'#F39C12','T_cells':'#E74C3C','NK':'#9B59B6',
    'Myeloid':'#2ECC71','B_Plasma':'#3498DB','Fibroblasts':'#1ABC9C',
    'Endothelial':'#E67E22','Mast':'#E91E63',
}
print(f"  {cell_meta.shape[0]} cells, {len(GENE_LIST)} genes, {len(CT_ORDER)} cell types")

# ── Helper ────────────────────────────────────────────────
def gini_coeff(x):
    x = np.sort(x)
    n = len(x)
    return (2 * sum((i+1)*v for i, v in enumerate(x)) / (n * sum(x)) - (n+1)/n)

def expand_sample_colors(samples):
    return [SAMPLE_COLORS.get(s, '#999') for s in samples]

# ── Routes ────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/summary')
def api_summary():
    return jsonify(SUMMARY)

@app.route('/api/cell_metadata')
def api_cell_metadata():
    sample = request.args.get('sample')
    ct = request.args.get('cell_type')
    df = cell_meta.copy()
    if sample:
        df = df[df['sample'] == sample]
    if ct:
        df = df[df['cell_type'] == ct]
    return jsonify(df.to_dict(orient='list'))

@app.route('/api/genes')
def api_genes():
    q = request.args.get('q', '').lower()
    if q:
        matches = [g for g in GENE_LIST if q in g.lower()]
        return jsonify(matches[:200])
    return jsonify(GENE_LIST[:200])

@app.route('/api/immune_genes')
def api_immune_genes():
    return jsonify(IMMUNE_GENES)

@app.route('/api/celltype_order')
def api_celltype_order():
    return jsonify(CT_ORDER)

@app.route('/api/proportions')
def api_proportions():
    return jsonify(PROPORTIONS)

@app.route('/api/lr_scores')
def api_lr_scores():
    sample = request.args.get('sample')
    top_n = int(request.args.get('top', 50))
    df = lr_df.copy()
    if sample:
        df = df[df['sample'] == sample]
    df = df.sort_values('score', ascending=False).head(top_n)
    return jsonify(df.to_dict(orient='records'))

@app.route('/api/gene_expr')
def api_gene_expr():
    """返回指定基因的每个细胞的表达值（仅限于immune_genes, 防止原始数据泄露）"""
    gene = request.args.get('gene', '').upper()
    if gene not in IMMUNE_GENES:
        return jsonify({
            'error': f'Gene "{gene}" not accessible. '
                     f'Only curated immune genes are available for per-cell query.',
            'available_in_ct': list(ct_mean.index)
        }), 403

    vals = immune_expr[gene].values.tolist() if gene in immune_expr.columns else []
    return jsonify({
        'gene': gene,
        'barcodes': cell_meta['barcode'].tolist(),
        'expression': vals,
        'metadata': {
            'cell_types': cell_meta['cell_type'].tolist(),
            'samples': cell_meta['sample'].tolist(),
            'mean': float(np.mean(vals)) if vals else 0,
            'pct_expressing': float(np.mean(np.array(vals) > 0)) * 100 if vals else 0,
        }
    })

@app.route('/api/ct_expression')
def api_ct_expression():
    genes = request.args.getlist('genes')
    if not genes:
        genes = IMMUNE_GENES
    valid = [g for g in genes if g in ct_mean.columns]
    if not valid:
        return jsonify({'error': 'No valid genes found'}), 400

    means = ct_mean[valid].to_dict(orient='index')
    pcts = ct_pct[valid].to_dict(orient='index')
    return jsonify({'mean': means, 'pct': pcts, 'cell_types': CT_ORDER, 'genes': valid})

@app.route('/api/cluster_map')
def api_cluster_map():
    return jsonify(CLUSTER_MAP)

@app.route('/api/marker_genes')
def api_marker_genes():
    """Return canonical marker genes for each cell type"""
    markers = {
        'T_cells': ['CD3D','CD3E','CD2','CD4','CD8A'],
        'NK': ['NKG7','GNLY','KLRD1','GZMB','GZMK'],
        'Myeloid': ['CD14','CD68','LYZ','FCGR3A','CSF1R'],
        'B_Plasma': ['CD79A','CD79B','MS4A1','JCHAIN','MZB1'],
        'Epithelial': ['EPCAM','KRT19','KRT18','KRT8','CDH1'],
        'Fibroblasts': ['COL1A1','COL1A2','DCN','LUM','FAP'],
        'Endothelial': ['PECAM1','VWF','CDH5','ENG','FLT1'],
        'Mast': ['KIT','TPSAB1','TPSB2','CPA3'],
    }
    return jsonify({ct: [g for g in genes if g in GENE_LIST]
                    for ct, genes in markers.items()})

# ── Custom analysis endpoints ─────────────────────────────
@app.route('/api/custom/dotplot', methods=['POST'])
def custom_dotplot():
    """Generate a custom dotplot and return as image"""
    data = request.json
    genes = data.get('genes', [])
    groupby = data.get('groupby', 'cell_type')
    sample = data.get('sample')
    title = data.get('title', 'Custom Dotplot')
    cmap = data.get('cmap', 'Reds')

    valid_genes = [g for g in genes if g in ct_mean.columns]
    if not valid_genes:
        return jsonify({'error': 'No valid genes'}), 400

    # Build the dotplot using scanpy
    # We need the original adata - but we can reconstruct from precomputed data
    # For the dotplot, we can use cell-level expression from immune_expr + subset
    fig, ax = plt.subplots(figsize=(max(6, len(valid_genes)*0.6), 4))

    gene_data = ct_mean[valid_genes].loc[CT_ORDER]
    pct_data = ct_pct[valid_genes].loc[CT_ORDER]

    # Simple dotplot using imshow
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=gene_data.min().min(), vmax=gene_data.max().max())
    cmap_obj = plt.colormaps.get(cmap) if hasattr(plt.colormaps, 'get') else plt.cm.get_cmap(cmap)
    for i, ct in enumerate(CT_ORDER):
        for j, g in enumerate(valid_genes):
            size = pct_data.loc[ct, g] * 500
            color = cmap_obj(norm(gene_data.loc[ct, g]))
            ax.scatter(j, i, s=size, color=color, edgecolors='none', zorder=3)

    ax.set_xticks(range(len(valid_genes)))
    ax.set_xticklabels(valid_genes, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(len(CT_ORDER)))
    ax.set_yticklabels(CT_ORDER, fontsize=9)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title(title, fontsize=13, fontweight='bold')

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label='Mean Expression')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    uid = uuid.uuid4().hex[:8]
    fname = f'dotplot_{uid}.png'
    with open(os.path.join(PLOTS_DIR, fname), 'wb') as f:
        f.write(buf.getvalue())
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode()

    return jsonify({
        'image': f'data:image/png;base64,{img_b64}',
        'filename': fname,
        'download_url': f'/download/{fname}'
    })

@app.route('/api/custom/violin', methods=['POST'])
def custom_violin():
    data = request.json
    gene = data.get('gene', '').upper()
    groupby = data.get('groupby', 'cell_type')
    sample = data.get('sample')
    title = data.get('title', f'{gene} Expression')

    if gene not in IMMUNE_GENES:
        return jsonify({'error': 'Gene not in curated immune set'}), 403
    if gene not in immune_expr.columns:
        return jsonify({'error': 'Gene data not found'}), 400

    fig, ax = plt.subplots(figsize=(7, 4))

    expr_vals = immune_expr[gene].values
    meta = cell_meta.copy()
    if sample:
        meta = meta[meta['sample'] == sample]
        expr_vals = immune_expr.loc[meta.index, gene].values if gene in immune_expr.columns else []

    groups = meta.groupby(groupby)
    positions = []
    labels = []
    all_data = []
    for i, (name, grp) in enumerate(groups):
        vals = immune_expr.loc[grp.index, gene].values
        all_data.append(vals)
        labels.append(name)
        positions.append(i)

    parts = ax.violinplot(all_data, positions=positions, showmeans=True, showmedians=True)
    for pc in parts['bodies']:
        pc.set_facecolor('#4C72B0')
        pc.set_alpha(0.6)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, fontsize=9)
    ax.set_ylabel('Expression (log-normalized)')
    ax.set_title(title, fontsize=13, fontweight='bold')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    uid = uuid.uuid4().hex[:8]
    fname = f'violin_{uid}.png'
    with open(os.path.join(PLOTS_DIR, fname), 'wb') as f:
        f.write(buf.getvalue())

    img_b64 = base64.b64encode(buf.read()).decode()
    return jsonify({
        'image': f'data:image/png;base64,{img_b64}',
        'filename': fname,
        'download_url': f'/download/{fname}'
    })

@app.route('/api/custom/barplot', methods=['POST'])
def custom_barplot():
    data = request.json
    groupby = data.get('groupby', 'sample')
    normalize = data.get('normalize', True)
    title = data.get('title', 'Cell Type Proportions')
    ct_filter = data.get('cell_types', CT_ORDER)

    df = cell_meta.copy()
    ct_valid = [c for c in ct_filter if c in df['cell_type'].unique()]
    if ct_valid:
        df = df[df['cell_type'].isin(ct_valid)]

    if groupby not in df.columns:
        groupby = 'sample'

    ct = pd.crosstab(df[groupby], df['cell_type'])
    if normalize:
        ct = ct.div(ct.sum(axis=1), axis=0) * 100

    ct = ct[[c for c in CT_ORDER if c in ct.columns]]

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = [CELLTYPE_COLORS.get(c, '#999') for c in ct.columns]
    ct.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
    ax.set_ylabel('Proportion (%)' if normalize else 'Cell Count')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    uid = uuid.uuid4().hex[:8]
    fname = f'barplot_{uid}.png'
    with open(os.path.join(PLOTS_DIR, fname), 'wb') as f:
        f.write(buf.getvalue())

    img_b64 = base64.b64encode(buf.read()).decode()
    return jsonify({
        'image': f'data:image/png;base64,{img_b64}',
        'filename': fname,
        'download_url': f'/download/{fname}',
        'table': ct.to_dict()
    })

@app.route('/api/custom/feature_umap', methods=['POST'])
def custom_feature_umap():
    data = request.json
    gene = data.get('gene', '').upper()
    sample = data.get('sample')
    title = data.get('title', f'{gene} Expression on UMAP')

    if gene not in IMMUNE_GENES:
        return jsonify({'error': 'Gene not in curated immune set'}), 403
    if gene not in immune_expr.columns:
        return jsonify({'error': f'Gene {gene} data not found'}), 400

    df = cell_meta.copy()
    expr = immune_expr[gene].values
    if sample:
        mask = df['sample'] == sample
        df = df[mask]
        expr_val = expr[mask.values]
    else:
        expr_val = expr

    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(df['UMAP_1'], df['UMAP_2'], c=expr_val, s=2,
                    cmap='Reds', alpha=0.7, rasterized=True)
    plt.colorbar(sc, ax=ax, label='Expression')
    ax.set_xlabel('UMAP 1')
    ax.set_ylabel('UMAP 2')
    ax.set_title(title, fontsize=13, fontweight='bold')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    uid = uuid.uuid4().hex[:8]
    fname = f'feature_umap_{uid}.png'
    with open(os.path.join(PLOTS_DIR, fname), 'wb') as f:
        f.write(buf.getvalue())

    img_b64 = base64.b64encode(buf.read()).decode()
    return jsonify({
        'image': f'data:image/png;base64,{img_b64}',
        'filename': fname,
        'download_url': f'/download/{fname}'
    })

@app.route('/download/<filename>')
def download_file(filename):
    return send_from_directory(PLOTS_DIR, filename, as_attachment=True)

# ── Raw data protection ──────────────────────────────────
@app.route('/api/raw_matrix')
def api_raw_matrix():
    return jsonify({'error': 'Raw expression matrix download is not allowed.'}), 403

@app.route('/api/export/table', methods=['POST'])
def export_table():
    """Export a table as CSV"""
    data = request.json
    table_type = data.get('type', 'proportions')
    if table_type == 'proportions':
        df = pd.crosstab(cell_meta['sample'], cell_meta['cell_type'])
    elif table_type == 'gene_expr':
        gene = data.get('gene')
        if gene not in IMMUNE_GENES:
            return jsonify({'error': 'Gene not in curated set'}), 403
        df = pd.DataFrame({'barcode': cell_meta['barcode'],
                           'sample': cell_meta['sample'],
                           'cell_type': cell_meta['cell_type'],
                           gene: immune_expr[gene].values})
    elif table_type == 'lr_scores':
        sample = data.get('sample')
        df = lr_df.copy()
        if sample:
            df = df[df['sample'] == sample]
    elif table_type == 'ct_expression':
        genes = data.get('genes', IMMUNE_GENES)
        valid = [g for g in genes if g in ct_mean.columns]
        df = ct_mean[valid].copy()
        df.insert(0, 'cell_type', df.index)
    else:
        return jsonify({'error': 'Unknown table type'}), 400

    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    csv_buf.seek(0)
    b64 = base64.b64encode(csv_buf.read().encode()).decode()
    return jsonify({'csv': b64, 'filename': f'{table_type}.csv'})

# ── Start ─────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
