###########################
# scRNA : https://support.10xgenomics.com/single-cell-gene-expression/datasets/2.1.0/pbmc8k
#       Gene/cell matrix (filtered)
#
# ATAC : https://support.10xgenomics.com/single-cell-atac/datasets/1.2.0/atac_v1_pbmc_10k
#       Transcription Factor by cell matrix (filtered)
#       Fragments (TSV)
#       Fragments index (TBI)
#
# Azimuth :
#       scRNA reference file : https://zenodo.org/record/4546839 
#       ATAC reference file : https://zenodo.org/record/7770389#.ZFhDx-zMLHo 
#       homologs.rds : https://seurat.nygenome.org/azimuth/references/homologs.rds 
#
#

source('RunAzimuthLocal.R')

root_dir = 'I:/single_cell/PBMC'
reference_path = file.path(root_dir, "reference")
pbmc_scrna_data_dir = file.path(root_dir, 'pbmc8k_filtered_gene_bc_matrices', 'filtered_gene_bc_matrices', 'GRCh38')
atac_fragment_path = file.path(root_dir, "atac_v1_pbmc_10k_fragments.tsv.gz")

##########处理scRNA
pbmc_scrna_data <- Read10X(pbmc_scrna_data_dir)
pbmc_scrna_obj <- CreateSeuratObject(counts = pbmc_scrna_data, project = "pbmc8k")

reference_path = file.path(root_dir, "reference")
pbmc_scrna <- RunAzimuth_local(query=pbmc_scrna_obj, reference=reference_path)
write.table(pbmc_scrna@meta.data, file='pbmc_scrna_metadata.tsv', sep=',', quote=FALSE)

# saveRDS(pbmc_scrna_obj, file.path(tmp_dir, "pbmc8k_scrna.rds"))
# AzimuthApp(reference=reference_path, demodataset="pbmc8k_scrna.rds",
#            Azimuth.app.homologs="homologs.rds", max_cells=100000,
#            default_metadata='celltype.l2')

##########处理ATAC
pbmc_atac_data_dir = file.path(root_dir, 'atac_v1_pbmc_10k_filtered_peak_bc_matrix','filtered_peak_bc_matrix')
pbmc_atac_data <- ReadMtx(mtx=file.path(pbmc_atac_data_dir,'matrix.mtx'), 
                          cells=file.path(pbmc_atac_data_dir, 'barcodes.tsv'),
                          features=file.path(pbmc_atac_data_dir, 'peaks.tsv'),
                          feature.column=1)
pbmc_atac_obj <- CreateSeuratObject(counts = pbmc_atac_data, project = "pbmc10k", assay="ATAC")


pbmc_atac <- RunAzimuth(query=pbmc_atac_obj, query.modality="ATAC",
                        reference=reference_path, 
                        fragment.path=atac_fragment_path)
write.table(pbmc_atac@meta.data, file='pbmc_atac_metadata.tsv', sep=',', quote=FALSE)

# saveRDS(atac_obj, "pbmc10k_atac.rds")
# AzimuthApp(reference=reference_path, demodataset="pbmc10k_atac.rds",
#            Azimuth.app.homologs="homologs.rds", max_cells=100000,
#            default_metadata='celltype.l2', do_bridge=TRUE, fragment.path=fragment_path)