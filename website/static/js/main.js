/* ── scRNA-seq Dashboard: Main JavaScript ── */
// ── Global state ──
const STATE = {
    summary: null,
    cellMeta: null,
    ctOrder: [],
    immuneGenes: [],
    loaded: false,
    lastCustom: null
};

const CT_COLORS = {
    'Epithelial':'#F39C12','T_cells':'#E74C3C','NK':'#9B59B6',
    'Myeloid':'#2ECC71','B_Plasma':'#3498DB','Fibroblasts':'#1ABC9C',
    'Endothelial':'#E67E22','Mast':'#E91E63'
};
const SAMPLE_COLORS = {'ZLF':'#4C72B0','ZFL':'#DD8452','HJX':'#55A868'};

// ── Utility ──
function toast(msg, type='info') {
    const c = document.getElementById('toastContainer');
    const t = document.createElement('div');
    t.className = `toast ${type}`;
    t.textContent = msg;
    c.appendChild(t);
    setTimeout(() => t.remove(), 3000);
}

function $(id) { return document.getElementById(id); }

function showLoading(el) {
    if (typeof el === 'string') el = $(el);
    el.innerHTML = '<div class="loading">加载中</div>';
}

async function fetchJSON(url) {
    const r = await fetch(url);
    if (!r.ok) { const e = await r.json().catch(()=>({})); throw new Error(e.error || r.statusText); }
    return r.json();
}

// ── Tab switching ──
document.querySelectorAll('.nav-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
        document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        $(`tab-${tab.dataset.tab}`).classList.add('active');
        // Trigger resize for plots
        window.dispatchEvent(new Event('resize'));
    });
});

// ── Initialize ──
(async function init() {
    try {
        STATE.summary = await fetchJSON('/api/summary');
        STATE.immuneGenes = await fetchJSON('/api/immune_genes');
        STATE.ctOrder = await fetchJSON('/api/celltype_order');

        // Fill header stats
        $('stat-cells').textContent = STATE.summary.total_cells.toLocaleString();
        $('stat-genes').textContent = STATE.summary.total_genes.toLocaleString();
        $('stat-samples').textContent = STATE.summary.samples.length;
        $('ov-cells').textContent = STATE.summary.total_cells.toLocaleString();
        $('ov-genes').textContent = STATE.summary.total_genes.toLocaleString();
        $('ov-ctypes').textContent = Object.keys(STATE.summary.cell_types).length;

        // Sample stats table
        const tb = document.querySelector('#ov-sample-table tbody');
        for (const s of STATE.summary.samples) {
            const d = STATE.summary.per_sample[s];
            const pct = (d.cells / STATE.summary.total_cells * 100).toFixed(1);
            tb.innerHTML += `<tr><td><strong>${s}</strong></td><td>${d.cells.toLocaleString()}</td><td>${d.mean_genes.toFixed(0)}</td><td>${pct}%</td></tr>`;
        }

        // Cell type selector
        const selCt = $('umap-ct');
        STATE.ctOrder.forEach(ct => {
            selCt.innerHTML += `<option value="${ct}">${ct}</option>`;
        });

        // Quick gene tags
        const qt = $('gene-quick-tags');
        ['CD3D','CD4','CD8A','NKG7','CD14','CD68','CD79A','EPCAM','PDCD1','IFNG','TNF','CCL5','CCR5'].forEach(g => {
            const tag = document.createElement('span');
            tag.className = 'gene-tag';
            tag.textContent = g;
            tag.onclick = () => { $('gene-search').value = g; updateGeneExpr(); };
            qt.appendChild(tag);
        });

        // Gene search datalist
        const allGenes = await fetchJSON('/api/genes?q=');
        const dl = $('gene-list');
        allGenes.forEach(g => { const o = document.createElement('option'); o.value = g; dl.appendChild(o); });

        // Load cell metadata
        STATE.cellMeta = await fetchJSON('/api/cell_metadata');

        STATE.loaded = true;

        // Render initial plots
        renderOverviewUMAP();
        renderOverviewBar();
        renderProportions();
        renderImmuneDotplots();
        updateLR();

        toast('数据加载完成！', 'success');
    } catch (e) {
        toast('加载失败: ' + e.message, 'error');
        console.error(e);
    }
})();

// ================================================================
// TAB 1: OVERVIEW
// ================================================================
function renderOverviewUMAP() {
    const meta = STATE.cellMeta;
    const trace = {
        x: meta.UMAP_1, y: meta.UMAP_2,
        mode: 'markers',
        type: 'scattergl',
        marker: {
            size: 2.5,
            color: meta.cell_type.map(ct => CT_COLORS[ct] || '#999'),
            opacity: 0.6
        },
        text: meta.barcode.map((b,i) =>
            `Barcode: ${b}<br>Cell Type: ${meta.cell_type[i]}<br>Sample: ${meta.sample[i]}`),
        hoverinfo: 'text'
    };
    const layout = {
        title: '', height: 420,
        margin: {l:40,r:20,t:10,b:40},
        xaxis: {title:'UMAP 1', showgrid:false, zeroline:false},
        yaxis: {title:'UMAP 2', showgrid:false, zeroline:false},
        hovermode: 'closest',
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
    };
    Plotly.newPlot('plot-overview-umap', [trace], layout, {responsive:true, displayModeBar:false});

    // Legend
    const leg = $('ov-legend');
    leg.innerHTML = STATE.ctOrder.map(ct =>
        `<span class="color-legend-item"><span class="color-dot" style="background:${CT_COLORS[ct]||'#999'}"></span>${ct}</span>`
    ).join('');
}

function renderOverviewBar() {
    const meta = STATE.cellMeta;
    const groups = {};
    meta.cell_type.forEach((ct, i) => {
        if (!groups[ct]) groups[ct] = 0;
        groups[ct]++;
    });
    const sorted = STATE.ctOrder.filter(c => groups[c]);
    const data = [{
        x: sorted, y: sorted.map(c => groups[c]),
        type: 'bar',
        marker: {color: sorted.map(c => CT_COLORS[c] || '#999')},
        text: sorted.map(c => groups[c].toLocaleString()),
        textposition: 'outside'
    }];
    const layout = {
        title: '', height: 380,
        margin: {l:60,r:30,t:10,b:100},
        yaxis: {title:'Cell Count'},
        xaxis: {tickangle: -30, tickfont: {size: 10}},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        bargap: 0.3
    };
    Plotly.newPlot('plot-overview-bar', data, layout, {responsive:true, displayModeBar:false});
}

// ================================================================
// TAB 2: UMAP
// ================================================================
async function updateUMAP() {
    const colorBy = $('umap-color').value;
    const sample = $('umap-sample').value;
    const ct = $('umap-ct').value;

    let url = '/api/cell_metadata';
    const params = [];
    if (sample) params.push(`sample=${sample}`);
    if (ct) params.push(`cell_type=${ct}`);
    if (params.length) url += '?' + params.join('&');

    try {
        const data = await fetchJSON(url);
        const colorMap = colorBy === 'cell_type' ? CT_COLORS :
                         colorBy === 'sample' ? SAMPLE_COLORS : {};
        const colors = data[colorBy].map(v => colorMap[v] || '#999');

        const trace = {
            x: data.UMAP_1, y: data.UMAP_2,
            mode: 'markers', type: 'scattergl',
            marker: {size: 3, color: colors, opacity: 0.6},
            text: data.barcode.map((b,i) =>
                `Cell: ${data.cell_type[i]}<br>Sample: ${data.sample[i]}<br>Cluster: ${data.leiden[i]}`),
            hoverinfo: 'text'
        };
        const layout = {
            title: `${colorBy} ${sample ? '— '+sample : ''} ${ct ? '— '+ct : ''}`,
            height: 480,
            margin: {l:50,r:20,t:40,b:50},
            xaxis: {title:'UMAP 1', showgrid:false, zeroline:false},
            yaxis: {title:'UMAP 2', showgrid:false, zeroline:false},
            hovermode: 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
        };
        Plotly.newPlot('plot-umap', [trace], layout, {responsive:true});
    } catch(e) {
        toast('UMAP更新失败: '+e.message, 'error');
    }
}

// ================================================================
// TAB 3: PROPORTIONS
// ================================================================
async function renderProportions() {
    const prop = await fetchJSON('/api/proportions');
    const bySample = prop.by_sample;
    const samples = Object.keys(bySample);
    const cts = STATE.ctOrder.filter(ct => samples.some(s => bySample[s][ct]));

    // Stacked bar
    const traces = cts.map(ct => ({
        x: samples,
        y: samples.map(s => bySample[s][ct] || 0),
        name: ct,
        type: 'bar',
        marker: {color: CT_COLORS[ct] || '#999'},
        text: samples.map(s => bySample[s][ct] ? (bySample[s][ct]/prop.totals[s]*100).toFixed(1)+'%' : ''),
        textposition: 'inside',
        textfont: {size: 10, color: 'white'}
    }));
    const layout1 = {
        title: '', height: 400,
        barmode: 'stack',
        margin: {l:50,r:20,t:10,b:50},
        yaxis: {title:'Cell Count'},
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
        legend: {orientation:'h', y:-0.15}
    };
    Plotly.newPlot('plot-prop-bar', traces, layout1, {responsive:true});

    // Per-sample split UMAP (subplots)
    const meta = STATE.cellMeta;
    const subTraces = samples.map((s, idx) => {
        const sub = {UMAP_1:[], UMAP_2:[], cell_type:[], barcode:[]};
        meta.barcode.forEach((b, i) => {
            if (meta.sample[i] === s) {
                sub.UMAP_1.push(meta.UMAP_1[i]);
                sub.UMAP_2.push(meta.UMAP_2[i]);
                sub.cell_type.push(meta.cell_type[i]);
                sub.barcode.push(b);
            }
        });
        return {
            x: sub.UMAP_1, y: sub.UMAP_2,
            mode: 'markers', type: 'scattergl',
            marker: {size: 2.5, color: sub.cell_type.map(ct => CT_COLORS[ct]||'#999'), opacity: 0.6},
            name: `${s} (n=${sub.barcode.length})`,
            text: sub.barcode.map((b,i) => `${b}<br>${sub.cell_type[i]}`),
            hoverinfo: 'text+name',
            xaxis: `x${idx+1}`, yaxis: `y${idx+1}`
        };
    });

    const gridLayout = {
        title: '', height: 360,
        grid: {rows: 1, columns: 3, pattern: 'independent'},
        margin: {l:40,r:20,t:30,b:40},
        showlegend: false,
        paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
    };
    samples.forEach((s, i) => {
        gridLayout[`xaxis${i+1}`] = {title:'UMAP 1', showgrid:false, zeroline:false};
        gridLayout[`yaxis${i+1}`] = {title:'UMAP 2', showgrid:false, zeroline:false};
        gridLayout[`annotations`] = (gridLayout.annotations || []).concat([{
            text: s, x: 0.17 + i*0.33, y: 1.05, xref:'paper', yref:'paper',
            showarrow: false, font: {size: 12, weight: 'bold'}
        }]);
    });
    Plotly.newPlot('plot-prop-umap', subTraces, gridLayout, {responsive:true});

    // Table
    const tbl = $('prop-table');
    const thead = tbl.querySelector('thead tr');
    const tbody = tbl.querySelector('tbody');
    thead.innerHTML = '<th>样本</th>' + cts.map(c => `<th>${c}</th>`).join('') + '<th>总计</th>';
    tbody.innerHTML = samples.map(s => {
        const row = cts.map(c => bySample[s][c] || 0);
        const total = row.reduce((a,b) => a+b, 0);
        return `<tr><td><strong>${s}</strong></td>${row.map(v => `<td>${v}</td>`).join('')}<td><strong>${total}</strong></td></tr>`;
    }).join('');
}

// ================================================================
// TAB 4: GENE EXPRESSION
// ================================================================
async function updateGeneExpr() {
    const gene = $('gene-search').value.trim().toUpperCase();
    if (!gene) return toast('请输入基因名', 'info');
    if (!STATE.immuneGenes.includes(gene)) {
        return toast(`"${gene}" 不在可查询的免疫基因列表中`, 'error');
    }

    const sample = $('gene-sample').value;

    try {
        // Get expression data
        const data = await fetchJSON(`/api/gene_expr?gene=${gene}`);

        // Filter by sample
        let idx = data.barcodes.map((_,i) => i);
        let expr = data.expression;
        if (sample) {
            const meta = STATE.cellMeta;
            idx = data.barcodes.map((b,i) => meta.sample[i] === sample ? i : -1).filter(i => i >= 0);
            expr = idx.map(i => data.expression[i]);
        }

        // Feature UMAP
        const meta = STATE.cellMeta;
        const umapX = idx.map(i => meta.UMAP_1[i]);
        const umapY = idx.map(i => meta.UMAP_2[i]);

        const trace = {
            x: umapX, y: umapY,
            mode: 'markers', type: 'scattergl',
            marker: {
                size: 3,
                color: expr,
                colorscale: 'Reds',
                showscale: true,
                colorbar: {title:'Expression', thickness: 15},
                opacity: 0.7
            },
            text: idx.map(i =>
                `${data.barcodes[i]}<br>${meta.cell_type[i]}<br>${gene}: ${data.expression[i].toFixed(3)}`),
            hoverinfo: 'text'
        };
        const layout = {
            title: `${gene} Expression on UMAP ${sample ? '— '+sample : ''}`,
            height: 400,
            margin: {l:50,r:30,t:40,b:50},
            xaxis: {title:'UMAP 1', showgrid:false, zeroline:false},
            yaxis: {title:'UMAP 2', showgrid:false, zeroline:false},
            hovermode: 'closest',
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
        };
        $('gene-umap-title').textContent = `🗺️ ${gene} UMAP 表达`;
        Plotly.newPlot('plot-gene-umap', [trace], layout, {responsive:true});

        // Violin by cell type
        const groups = {};
        idx.forEach((i, j) => {
            const ct = meta.cell_type[i];
            if (!groups[ct]) groups[ct] = [];
            groups[ct].push(expr[j]);
        });

        const vioTraces = STATE.ctOrder.filter(ct => groups[ct]).map(ct => ({
            y: groups[ct],
            name: ct,
            type: 'violin',
            meanline: {visible: true},
            box: {visible: true},
            line: {color: CT_COLORS[ct] || '#999'},
            fillcolor: CT_COLORS[ct] || '#999',
            opacity: 0.6
        }));
        const vioLayout = {
            title: `${gene} Expression by Cell Type`,
            height: 330,
            margin: {l:50,r:20,t:40,b:80},
            yaxis: {title:'Expression (log-norm)'},
            xaxis: {tickangle: -30},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
        };
        Plotly.newPlot('plot-gene-violin', vioTraces, vioLayout, {responsive:true});

        // Bar: mean expression per cell type
        const ctMeans = {};
        Object.entries(groups).forEach(([ct, vals]) => {
            ctMeans[ct] = vals.reduce((a,b) => a+b, 0) / vals.length;
        });
        const ordered = STATE.ctOrder.filter(ct => ctMeans[ct] !== undefined);
        const barTrace = {
            x: ordered, y: ordered.map(ct => ctMeans[ct]),
            type: 'bar',
            marker: {color: ordered.map(ct => CT_COLORS[ct] || '#999')},
            text: ordered.map(ct => ctMeans[ct].toFixed(3)),
            textposition: 'outside'
        };
        const barLayout = {
            title: `${gene} Mean Expression by Cell Type`,
            height: 280,
            margin: {l:50,r:20,t:40,b:80},
            yaxis: {title:'Mean Expression'},
            xaxis: {tickangle: -30},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
            bargap: 0.3
        };
        Plotly.newPlot('plot-gene-bar', [barTrace], barLayout, {responsive:true});

        toast(`显示 ${gene} 表达`, 'success');
    } catch(e) {
        toast('查询失败: ' + e.message, 'error');
    }
}

async function downloadGeneExpr() {
    const gene = $('gene-search').value.trim().toUpperCase();
    if (!gene) return toast('请先查询基因', 'info');
    try {
        const res = await fetchJSON(`/api/gene_expr?gene=${gene}`);
        // Build CSV from the data
        const meta = STATE.cellMeta;
        let csv = 'barcode,sample,cell_type,' + gene + '\n';
        res.barcodes.forEach((b, i) => {
            csv += `${b},${meta.sample[i]},${meta.cell_type[i]},${res.expression[i]}\n`;
        });
        downloadCSV(csv, `${gene}_expression.csv`);
        toast('下载完成', 'success');
    } catch(e) { toast('下载失败', 'error'); }
}

function downloadCSV(content, filename) {
    const blob = new Blob([content], {type: 'text/csv;charset=utf-8;'});
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = filename;
    link.click();
    URL.revokeObjectURL(link.href);
}

// ================================================================
// TAB 5: IMMUNE
// ================================================================
async function renderImmuneDotplots() {
    try {
        const resp = await fetchJSON('/api/ct_expression');
        const ctExpr = resp.mean;
        const ctPct = resp.pct;
        const cts = resp.cell_types;

        // Define gene groups
        const groups = {
            'checkpoint': ['PDCD1','CTLA4','LAG3','HAVCR2','TIGIT','BTLA','CD274','ICOS','CD80','CD86'],
            'cytokine': ['IFNG','TNF','IL2','IL4','IL6','IL10','IL15','IL18','TGFB1','CSF2'],
            'chemokine': ['CCL2','CCL3','CCL4','CCL5','CCL19','CCL20','CXCL8','CXCL9','CXCL10','CXCL12','CXCL13','CX3CL1']
        };

        Object.entries(groups).forEach(([key, genes]) => {
            const valid = genes.filter(g => resp.genes.includes(g));
            if (!valid.length) return;

            const traces = valid.map(g => {
                const x = cts.map(ct => ctExpr[ct][g] || 0);
                const sizes = cts.map(ct => (ctPct[ct][g] || 0) * 40);
                return {
                    x: cts, y: Array(cts.length).fill(g),
                    mode: 'markers',
                    marker: {
                        size: sizes,
                        color: x,
                        colorscale: 'Reds',
                        showscale: true,
                        colorbar: {title:'Mean', thickness: 10, len: 0.6},
                        sizemin: 2,
                        sizeref: 2
                    },
                    text: cts.map((ct, i) =>
                        `${g} in ${ct}<br>Mean: ${x[i].toFixed(3)}<br>%Exp: ${(ctPct[ct][g]*100||0).toFixed(1)}%`),
                    hoverinfo: 'text',
                    showlegend: false
                };
            });

            const layout = {
                title: '',
                height: Math.max(220, valid.length * 42 + 100),
                margin: {l:90,r:50,t:10,b:100},
                xaxis: {title:'', tickangle: -30},
                yaxis: {title:'', autorange:'reversed', tickfont: {size: 11}},
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
                colorbar: {thickness: 10, len: 0.5}
            };
            const divId = key === 'checkpoint' ? 'plot-immune-checkpoint' :
                          key === 'cytokine' ? 'plot-immune-cytokine' : 'plot-immune-chemokine';
            Plotly.newPlot(divId, traces, layout, {responsive:true});
        });
    } catch(e) {
        toast('免疫点图加载失败: ' + e.message, 'error');
    }
}

// ================================================================
// TAB 6: LR INTERACTION
// ================================================================
async function updateLR() {
    const sample = $('lr-sample').value;
    try {
        const data = await fetchJSON(`/api/lr_scores?sample=${sample}&top=30`);

        // Bar chart
        const labels = data.map(d =>
            `${d.ligand}→${d.receptor}<br>${d.sender}→${d.receiver}`);
        const scores = data.map(d => d.score);
        const colors = data.map(d => {
            if (d.score > 0.5) return '#E74C3C';
            if (d.score > 0.2) return '#F39C12';
            return '#3498DB';
        });

        const trace = {
            y: labels.reverse(),
            x: scores.reverse(),
            type: 'bar',
            orientation: 'h',
            marker: {color: colors.reverse()},
            text: scores.reverse().map(s => s.toFixed(3)),
            textposition: 'outside',
            textfont: {size: 9}
        };
        const layout = {
            title: `${sample || 'All Samples'} — Top LR Pairs`,
            height: 480,
            margin: {l:180,r:70,t:40,b:40},
            xaxis: {title:'Interaction Score'},
            yaxis: {tickfont: {size: 10}},
            paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
        };
        Plotly.newPlot('plot-lr-bar', [trace], layout, {responsive:true});

        // LR expression dotplot (ligand + receptor expression)
        const allGenes = [...new Set(data.flatMap(d => [d.ligand, d.receptor]))].slice(0, 30);
        if (allGenes.length) {
            const resp = await fetchJSON(`/api/ct_expression?genes=${allGenes.join('&genes=')}`);
            const cts = resp.cell_types;
            const validGenes = resp.genes;

            const lrTraces = validGenes.map(g => {
                const x = cts.map(ct => ctExpr => ctExpr[ct][g] || 0);
                // Actually use resp.mean
                const x2 = cts.map(ct => resp.mean[ct][g] || 0);
                const sizes = cts.map(ct => Math.max(4, (resp.pct[ct][g] || 0) * 50));
                return {
                    x: cts, y: Array(cts.length).fill(g),
                    mode: 'markers',
                    marker: {
                        size: sizes,
                        color: x2,
                        colorscale: 'Reds',
                        showscale: true,
                        colorbar: {title:'Mean', thickness: 10, len: 0.5},
                        sizemin: 3
                    },
                    text: cts.map((ct, i) => `${g} in ${ct}<br>Mean: ${x2[i].toFixed(3)}`),
                    hoverinfo: 'text',
                    showlegend: false
                };
            });

            Plotly.newPlot('plot-lr-dotplot', lrTraces, {
                title: 'Ligand & Receptor Expression by Cell Type',
                height: Math.max(220, validGenes.length * 40 + 100),
                margin: {l:90,r:50,t:40,b:100},
                xaxis: {tickangle: -30, tickfont: {size: 11}},
                yaxis: {autorange:'reversed', tickfont: {size: 11}},
                paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)'
            }, {responsive:true});
        }
    } catch(e) {
        toast('LR加载失败: ' + e.message, 'error');
    }
}

// ================================================================
// TAB 7: CUSTOM ANALYSIS
// ================================================================
function generateCustomPlot() {
    const type = $('cust-type').value;
    const genes = $('cust-genes').value.split(',').map(g => g.trim().toUpperCase()).filter(Boolean);
    const title = $('cust-title').value || `Custom ${type}`;
    const groupby = $('cust-groupby').value;

    let url, body;

    switch(type) {
        case 'dotplot':
            if (!genes.length) return toast('请输入至少一个基因', 'info');
            url = '/api/custom/dotplot';
            body = {genes, groupby, title};
            break;
        case 'violin':
            if (!genes.length) return toast('请输入基因名', 'info');
            url = '/api/custom/violin';
            body = {gene: genes[0], groupby, title};
            break;
        case 'barplot':
            url = '/api/custom/barplot';
            body = {groupby, title};
            break;
        case 'feature_umap':
            if (!genes.length) return toast('请输入基因名', 'info');
            url = '/api/custom/feature_umap';
            body = {gene: genes[0], title};
            break;
    }

    showLoading('plot-custom-result');

    fetch(url, {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
    })
    .then(r => r.json())
    .then(resp => {
        if (resp.error) { toast(resp.error, 'error'); return; }
        $('plot-custom-result').innerHTML =
            `<img src="${resp.image}" alt="Custom Plot" style="width:100%">`;
        $('cust-download-bar').style.display = 'flex';
        STATE.lastCustom = resp;
        $('cust-result-title').textContent = `📊 ${title}`;
        // Show CSV download only for barplot
        $('cust-csv-btn').style.display = type === 'barplot' ? 'inline-flex' : 'none';
        toast('图表生成成功！', 'success');
    })
    .catch(e => toast('生成失败: ' + e.message, 'error'));
}

function downloadCustomPlot() {
    if (!STATE.lastCustom) return;
    const a = document.createElement('a');
    a.href = STATE.lastCustom.download_url;
    a.download = STATE.lastCustom.filename;
    a.click();
    toast('下载中...', 'info');
}

function downloadCustomCSV() {
    if (!STATE.lastCustom || !STATE.lastCustom.table) return;
    const cols = Object.keys(STATE.lastCustom.table);
    const rows = Object.values(STATE.lastCustom.table);
    if (!rows.length) return;
    const n = rows[0].length;
    let csv = cols.join(',') + '\n';
    for (let i = 0; i < n; i++) {
        csv += cols.map(c => STATE.lastCustom.table[c][i]).join(',') + '\n';
    }
    downloadCSV(csv, 'custom_barplot.csv');
    toast('CSV下载完成', 'success');
}

// ── Download functions ──
async function downloadTable(type) {
    try {
        switch(type) {
            case 'proportions':
                const prop = await fetchJSON('/api/proportions');
                const bySample = prop.by_sample;
                const samples = Object.keys(bySample);
                const cts = STATE.ctOrder.filter(ct => samples.some(s => bySample[s][ct]));
                let csv = 'sample,' + cts.join(',') + ',total\n';
                samples.forEach(s => {
                    const row = cts.map(ct => bySample[s][ct] || 0);
                    const total = row.reduce((a,b) => a+b, 0);
                    csv += `${s},${row.join(',')},${total}\n`;
                });
                downloadCSV(csv, 'proportions.csv');
                break;
            case 'ct_expression':
                const resp = await fetchJSON('/api/ct_expression');
                const genes = resp.genes;
                const ctypes = resp.cell_types;
                csv = 'cell_type,' + genes.join(',') + '\n';
                ctypes.forEach(ct => {
                    csv += `${ct},${genes.map(g => resp.mean[ct][g] || 0).join(',')}\n`;
                });
                downloadCSV(csv, 'celltype_expression.csv');
                break;
            case 'lr_scores':
                const sample = $('lr-sample').value;
                const lr = await fetchJSON(`/api/lr_scores?sample=${sample}&top=1000`);
                csv = 'ligand,receptor,sender,receiver,score\n';
                lr.forEach(d => { csv += `${d.ligand},${d.receptor},${d.sender},${d.receiver},${d.score}\n`; });
                downloadCSV(csv, 'lr_scores.csv');
                break;
        }
        toast('表格下载完成', 'success');
    } catch(e) { toast('下载失败: ' + e.message, 'error'); }
}
