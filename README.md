# scRNA-seq Analysis (NACI2026)

This folder contains the results of the single-cell RNA-seq processing for 3 samples (ZLF, ZFL, HJX).

## Folder Structure
- `data/`: Contains unzipped matrix files.
- `results/`: Contains generated plots and analysis output.
- `scRNA_Processing.ipynb`: Jupyter Notebook with the complete code.

## Processing Summary
1. **Manual Loading**: Handled non-standard 1-column feature files.
2. **QC**: Calculated mitochondrial percentage and gene counts.
3. **Filtering**: Removed low-quality cells (< 200 genes, > 20% MT).
4. **Analysis**: Normalized, identified HVGs, performed PCA and UMAP.
5. **Clustering**: Applied Leiden clustering.

Note: The final AnnData export (.h5ad) was skipped due to local disk space limitations (dense matrix size ~5.3GB). The notebook can be re-run to regenerate the object if needed.
