
from src.experiment.train import run_with, remove_checkpoints

if __name__=="__main__":
    root_dirs = {
                 "pbmc": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC/graph/hg19_HOCOMOCOv11",
                 "lung": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/Lung_GSM4508936/graph/hg19_HOCOMOCOv11",
                 "ad": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_GSE174367/graph/hg38_HOCOMOCOv11",
                 "pbmc_signac": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC_signac/graph/hg19_HOCOMOCOv11"
                 }
    cell_types = {"pbmc": ["B_naive"],# "CD14_Mono", "CD4_TCM", "CD8_Naive"],
                  "lung": ["Vascular_endothelial_cells", "Ciliated_epithelial_cells", "Bronchiolar_and_alveolar_epithelial_cells", "Lymphatic_endothelial_cells","Lymphoid_cells",
                           "Megakaryocytes","Myeloid_cells","Neuroendocrine_cells","Stromal_cells"],
                  "ad": ["ASC", "MG", "ODC", "OPC", "PER.END"],
                  "pbmc_signac": ["CD14+_Monocytes", "B_cell_progenitor", "CD16+_Monocytes", "CD4_Memory"]}
    key = "lung"

    comment = "mask_pos"
    # mask_type = 'drop'

    mask_type = 'flip'
    comment = 'flip_pos'
    # comment = 'debug'
    # remove_checkpoints("/home/jm/PycharmProjects/xjl/debug/src/experiment/lightning_logs",
    #                    comment="flip_pos")
    # remove_checkpoints("/home/jm/PycharmProjects/xjl/debug/src/experiment/lightning_logs",
    #                    comment="mask_pos")
    # remove_checkpoints("/home/jm/PycharmProjects/xjl/debug/src/experiment/lightning_logs",
    #                    comment="first")

    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.6, mask_type=mask_type)
    #
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.7, mask_type=mask_type)

    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.05, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.08, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.10, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.12, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.15, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.30, mask_type=mask_type)
    #
    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.50, mask_type=mask_type)
    #
    comment = "mask_pos"
    mask_type = 'drop'

   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.9, mask_type=mask_type)
    #
   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.02, mask_type=mask_type)
    #
    #run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.05, mask_type=mask_type)
    #
   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.08, mask_type=mask_type)
    #
   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.10, mask_type=mask_type)
    #
   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.12, mask_type=mask_type)
    #
  #  run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.15, mask_type=mask_type)
    #
   # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.30, mask_type=mask_type)
    #
    run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto', mask_rate=0.50, mask_type=mask_type)
