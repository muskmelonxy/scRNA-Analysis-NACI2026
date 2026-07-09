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

# ── Load advanced data ────────────────────────────────────
print("Loading advanced analysis data...")
pseudotime = None
paga_conn = None
interaction_summary = None
sig_matrix = None
sig_matrix_markers = None
try:
    pt_path = os.path.join(DATA_DIR, 'pseudotime.csv.gz')
    if os.path.exists(pt_path):
        pseudotime = pd.read_csv(pt_path)
        print(f"  Pseudotime: {pseudotime.shape[0]} cells")

    paga_path = os.path.join(DATA_DIR, 'paga_connectivity.csv')
    if os.path.exists(paga_path):
        paga_conn = pd.read_csv(paga_path, index_col=0)
        print(f"  PAGA: {paga_conn.shape[0]} clusters")

    inter_path = os.path.join(DATA_DIR, 'interaction_summary.csv')
    if os.path.exists(inter_path):
        interaction_summary = pd.read_csv(inter_path)
        print(f"  Interaction summary: {interaction_summary.shape[0]} pairs")

    sig_path = os.path.join(DATA_DIR, 'signature_matrix.csv.gz')
    if os.path.exists(sig_path):
        sig_matrix = pd.read_csv(sig_path, index_col=0)
        print(f"  Signature matrix: {sig_matrix.shape[0]} genes × {sig_matrix.shape[1]} cell types")

    sig_m_path = os.path.join(DATA_DIR, 'signature_matrix_markers.csv.gz')
    if os.path.exists(sig_m_path):
        sig_matrix_markers = pd.read_csv(sig_m_path, index_col=0)
        print(f"  Marker sig matrix: {sig_matrix_markers.shape[0]} genes × {sig_matrix_markers.shape[1]} cell types")
except Exception as e:
    print(f"  Warning: advanced data load error: {e}")
print("Done loading.")

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

# ── Pseudotime ─────────────────────────────────────────────
@app.route('/api/pseudotime')
def api_pseudotime():
    if pseudotime is None:
        return jsonify({'error': 'Pseudotime not available'}), 404
    # Merge with cell metadata
    df = cell_meta[['barcode','cell_type','sample']].merge(pseudotime, on='barcode')
    return jsonify(df.to_dict(orient='list'))

@app.route('/api/paga')
def api_paga():
    if paga_conn is None:
        return jsonify({'error': 'PAGA not available'}), 404
    data = {
        'clusters': [str(c) for c in paga_conn.index],
        'connectivity': paga_conn.values.tolist(),
        'labels': [CLUSTER_MAP.get(str(c), str(c)) for c in paga_conn.index],
    }
    return jsonify(data)

# ── Interaction summary ───────────────────────────────────
@app.route('/api/interaction_summary')
def api_interaction_summary():
    if interaction_summary is None:
        return jsonify({'error': 'Not available'}), 404
    sample = request.args.get('sample')
    df = interaction_summary.copy()
    if sample:
        df = df[df['sample'] == sample]
    return jsonify(df.to_dict(orient='records'))

# ── Bulk RNA-seq deconvolution ────────────────────────────
@app.route('/api/deconvolve/reference_genes')
def api_deconvolve_reference():
    if sig_matrix is None:
        return jsonify({'error': 'Reference not available'}), 404
    all_genes = sig_matrix.index.tolist()
    markers = sig_matrix_markers.index.tolist() if sig_matrix_markers is not None else []
    return jsonify({
        'all_genes': all_genes[:500],  # return first 500 for preview
        'marker_genes': markers,
        'total_genes': len(all_genes),
        'cell_types': sig_matrix.columns.tolist()
    })

@app.route('/api/deconvolve', methods=['POST'])
def api_deconvolve():
    """Upload bulk RNA-seq data and deconvolve using NNLS"""
    if sig_matrix is None:
        return jsonify({'error': 'Reference not available'}), 404

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400

    try:
        # Read uploaded file (CSV/TSV)
        fname = file.filename.lower()
        sep = '\t' if fname.endswith('.tsv') else ','
        bulk = pd.read_csv(file, sep=sep, index_col=0)
    except Exception as e:
        return jsonify({'error': f'Cannot parse file: {str(e)}'}), 400

    # Determine format: genes in rows or columns?
    # Expect: genes × samples (genes as index)
    if bulk.shape[0] < 10:
        return jsonify({'error': 'Too few rows. Expect genes as rows, samples as columns.'}), 400

    # Match genes between bulk and reference
    ref = sig_matrix.copy()
    common_genes = list(set(bulk.index) & set(ref.index))
    if len(common_genes) < 50:
        return jsonify({
            'error': f'Only {len(common_genes)} genes matched between bulk and reference (need ≥50)'
        }), 400

    bulk_matched = bulk.loc[common_genes]
    ref_matched = ref.loc[common_genes]

    # Normalize bulk sample counts to library size (CPM-like)
    from scipy.optimize import nnls
    results = {}
    for col in bulk_matched.columns:
        bulk_vec = bulk_matched[col].values.astype(float)
        # Normalize to sum-to-1 (relative abundance) to match reference scale
        total = bulk_vec.sum()
        if total > 0:
            bulk_vec = bulk_vec / total * 1e4  # scale to CPM-like
        else:
            bulk_vec = bulk_vec

        ref_mat = ref_matched.values.astype(float)
        # Normalize reference columns to sum-to-1
        ref_norm = ref_mat / (ref_mat.sum(axis=0) + 1e-10)

        # NNLS: solve min ||Ax - b|| with x >= 0
        sol, residual = nnls(ref_norm, bulk_vec)
        # Normalize proportions to sum to 1
        proportions = sol / (sol.sum() + 1e-10)

        results[col] = {
            prop: float(proportions[i])
            for i, prop in enumerate(ref.columns)
            if proportions[i] > 0.001
        }

    # Also compute a combined result (mean across samples)
    all_props = pd.DataFrame(results).T.fillna(0)
    all_props['_sample'] = all_props.index

    # Generate a bar plot of results
    fig, ax = plt.subplots(figsize=(max(6, len(all_props) * 0.8), 4))
    plot_data = all_props.drop(columns=['_sample'])
    colors = [CELLTYPE_COLORS.get(c, '#999') for c in plot_data.columns]
    plot_data.plot(kind='bar', stacked=True, ax=ax, color=colors, edgecolor='white')
    ax.set_ylabel('Estimated Proportion')
    ax.set_title('Bulk Deconvolution: Cell Type Proportions', fontweight='bold')
    ax.legend(frameon=False, bbox_to_anchor=(1, 1))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    uid = uuid.uuid4().hex[:8]
    fname = f'deconv_{uid}.png'
    with open(os.path.join(PLOTS_DIR, fname), 'wb') as f:
        f.write(buf.getvalue())
    img_b64 = base64.b64encode(buf.read()).decode()

    # Also generate gene overlap info
    overlap_info = {
        'total_bulk_genes': len(bulk.index),
        'matched_genes': len(common_genes),
        'percent_matched': round(len(common_genes) / len(bulk.index) * 100, 1),
    }

    return jsonify({
        'image': f'data:image/png;base64,{img_b64}',
        'filename': fname,
        'download_url': f'/download/{fname}',
        'results': results,
        'cell_types': list(ref.columns),
        'overlap': overlap_info,
    })

@app.route('/api/deconvolve/csv', methods=['POST'])
def api_deconvolve_csv():
    """Return deconvolution result as CSV"""
    data = request.json
    results = data.get('results', {})
    if not results:
        return jsonify({'error': 'No results'}), 400
    df = pd.DataFrame(results).T.fillna(0)
    csv_buf = io.StringIO()
    df.to_csv(csv_buf)
    csv_buf.seek(0)
    b64 = base64.b64encode(csv_buf.read().encode()).decode()
    return jsonify({'csv': b64, 'filename': 'deconvolution_results.csv'})

# ── Start ─────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5003, debug=True)
