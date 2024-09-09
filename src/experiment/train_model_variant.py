import os
from src.trainer import Trainer, OldEvaluator
from src.dataset import DeepTFniDataModule
from src.benchmark.models.evaluator import Evaluator

from src.experiment.train import run_with


if __name__=="__main__":
    root_dirs = {
                 "pbmc": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC/graph/hg19_HOCOMOCOv11",
                 "lung": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/Lung_GSM4508936/graph/hg19_HOCOMOCOv11",
                 "ad": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_GSE174367/graph/hg38_HOCOMOCOv11",
                 "pbmc_signac": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC_signac/graph/hg19_HOCOMOCOv11"
                 }
    cell_types = {"pbmc": ["B_naive"],# "CD14_Mono", "CD4_TCM", "CD8_Naive"],
                  "lung": ["Vascular_endothelial_cells", "Ciliated_epithelial_cells", "Bronchiolar_and_alveolar_epithelial_cells", "Lymphatic_endothelial_cells"],
                  "ad": ["ASC", "MG", "ODC", "OPC", "PER.END"],
                  "pbmc_signac": ["CD14+_Monocytes", "B_cell_progenitor", "CD16+_Monocytes", "CD4_Memory"]}
    key = "pbmc"

    comment = "just_atac"
    run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAEVariant', accelerator='auto',
             use_atac_feature=True, use_scrna_feature=False, use_edge_attr=False)

    comment = "just_scrna"
    run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAEVariant', accelerator='auto',
             use_atac_feature=False, use_scrna_feature=True, use_edge_attr=False)

    comment = "no_edge_attr"
    run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAEVariant', accelerator='auto',
             use_atac_feature=True, use_scrna_feature=True, use_edge_attr=False)












