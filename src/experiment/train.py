import json
import os
import shutil
import pandas as pd
from tqdm import tqdm
from src.trainer import Trainer, OldEvaluator
from src.dataset import DeepTFniDataModule
from src.benchmark.models.evaluator import Evaluator

def run_with_grnboost2(data_dir, cell_types, scrna_just_node=False, comment='test', mask_rate=0.0, **param):
    scrna_method = "grnboost2"
    model_cls = "GRNBoost2"
    for cell_type in cell_types:
        root_dir = os.path.join(data_dir, cell_type)
        adj_file = os.path.join(root_dir, f"x_graph.txt")
        node_file = os.path.join(root_dir, f"x_node.txt")
        feature_name = f"{scrna_method}_just_node" if scrna_just_node else scrna_method
        scrna_feature_file = os.path.join(root_dir, f"x_scRNA_{feature_name}_feature.txt")
        atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
        scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv" if scrna_just_node else "scrna.csv")
        parser = Evaluator.add_argparse_args(name=cell_type, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                           comment=comment,
                                           atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                           model_cls=model_cls, scrna_just_node=scrna_just_node, mask_rate=mask_rate, **param)
        config = parser.parse_args()
        config.accelerator = 'cpu'
        Evaluator.run_all_fold(**vars(config))
        # Trainer.run_single_fold(**vars(config))


def run_with_genie3(data_dir, cell_types, scrna_just_node=False, mask_rate=0.0, comment='test', **param):
    scrna_method = "genie3"
    model_cls = "Genie3"
    for cell_type in cell_types:
        root_dir = os.path.join(data_dir, cell_type)
        adj_file = os.path.join(root_dir, f"x_graph.txt")
        node_file = os.path.join(root_dir, f"x_node.txt")
        feature_name = f"{scrna_method}_just_node" if scrna_just_node else scrna_method
        scrna_feature_file = os.path.join(root_dir, f"x_scRNA_{feature_name}_feature.txt")
        atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
        scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv" if scrna_just_node else "scrna.csv")
        parser = Evaluator.add_argparse_args(name=cell_type, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                           comment=comment,
                                           atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                           model_cls=model_cls, scrna_just_node=scrna_just_node, mask_rate=mask_rate, **param)
        config = parser.parse_args()
        config.accelerator = 'cpu'
        Evaluator.run_all_fold(**vars(config))


def run_with_deepwalk(data_dir, cell_types, comment='test', mask_rate=0.0, **param):
    from src.benchmark.models.deepwalk import DeepWalk
    scrna_method = "genie3"
    scrna_just_node = False
    model_cls = "DeepWalk"
    for cell_type in cell_types:
        root_dir = os.path.join(data_dir, cell_type)
        adj_file = os.path.join(root_dir, f"x_graph.txt")
        node_file = os.path.join(root_dir, f"x_node.txt")
        feature_name = f"{scrna_method}_just_node" if scrna_just_node else scrna_method
        scrna_feature_file = os.path.join(root_dir, f"x_scRNA_{feature_name}_feature.txt")
        atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
        scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv" if scrna_just_node else "scrna.csv")
        parser = Evaluator.add_argparse_args(name=cell_type, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                           comment=comment,
                                           atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                           model_cls=model_cls, scrna_just_node=scrna_just_node, mask_rate=mask_rate, **param)
        parser = DeepWalk.add_argparse_args(parser)
        config = parser.parse_args()
        config.accelerator = 'cpu'
        Evaluator.run_all_fold(**vars(config))


def run_with_genelink(data_dir, cell_types, comment='test', mask_rate=0.0, **param):
    from src.benchmark.models.genelink import GENELink
    scrna_method = "grnboost2"
    scrna_just_node = False
    model_cls = "GENELink"
    for cell_type in cell_types:
        root_dir = os.path.join(data_dir, cell_type)
        adj_file = os.path.join(root_dir, f"x_graph.txt")
        node_file = os.path.join(root_dir, f"x_node.txt")
        feature_name = f"{scrna_method}_just_node" if scrna_just_node else scrna_method
        scrna_feature_file = os.path.join(root_dir, f"x_scRNA_{feature_name}_feature.txt")
        atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
        scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv" if scrna_just_node else "scrna.csv")
        parser = Evaluator.add_argparse_args(name=cell_type, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                           comment=comment,
                                           atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                           model_cls=model_cls, scrna_just_node=scrna_just_node, mask_rate=mask_rate, **param)
        parser = GENELink.add_argparse_args(parser)
        config = parser.parse_args()
        config.accelerator = 'gpu'
        Evaluator.run_all_fold(**vars(config))


def run_with(data_dir, cell_types, scrna_just_node=False, scrna_method="grnboost2", comment='test', accelerator='cpu', model_cls="DeepTFni", mask_rate=0.0, **param):

    for cell_type in cell_types:
        root_dir = os.path.join(data_dir, cell_type)
        adj_file = os.path.join(root_dir, f"x_graph.txt")
        node_file = os.path.join(root_dir, f"x_node.txt")
        feature_name = f"{scrna_method}_just_node" if scrna_just_node else scrna_method
        scrna_feature_file = os.path.join(root_dir, f"x_scRNA_{feature_name}_feature.txt")
        atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
        scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv" if scrna_just_node else "scrna.csv")
        parser = Trainer.add_argparse_args(name=cell_type, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                           comment=comment,
                                           atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                           model_cls=model_cls, mask_rate=mask_rate, data_dir='data_control2',**param)
        config = parser.parse_args()
        config.accelerator = accelerator
        Trainer.run_all_fold(**vars(config))
        # Trainer.run_single_fold(**vars(config))

def patch_dataset_file(dataset_dir, log_dir):
    data_files = {}
    for cell_type in os.listdir(dataset_dir):
        data_files[cell_type] = {}
        for tag in os.listdir(os.path.join(dataset_dir, cell_type)):
            data_dir = os.path.join(dataset_dir, cell_type, tag)
            train_files = sorted([os.path.abspath(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if "train" in file])
            val_files = sorted([os.path.abspath(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if "val" in file])
            test_files = sorted([os.path.abspath(os.path.join(data_dir, file)) for file in os.listdir(data_dir) if "test" in file])
            assert len(train_files)==len(val_files) and len(val_files)==len(test_files)
            data_files[cell_type][tag] = {"train": train_files,
                                          "val": val_files,
                                          "test": test_files}
    for cell_type in os.listdir(log_dir):
        for model in tqdm(os.listdir(os.path.join(log_dir, cell_type))):
            for comment in os.listdir(os.path.join(log_dir, cell_type, model)):
                for split in os.listdir(os.path.join(log_dir, cell_type, model, comment)):
                    split_dir = os.path.join(log_dir, cell_type, model, comment, split)
                    if model not in ['DeepWalk', 'Genie3', 'GRNBoost2']:
                        split_dir = os.path.join(log_dir, cell_type, model, comment, split, "lightning_logs")
                    if not os.path.exists(split_dir):
                        continue
                    for version in os.listdir(split_dir):
                        config_file = os.path.join(split_dir, version, "config.json")
                        config_file = config_file if os.path.exists(config_file) else os.path.join(split_dir, version, "metrics.json")
                        if not os.path.exists(config_file):
                            continue
                        with open(config_file) as f:
                            config = json.load(f)
                        tag = f"seed-{config['dataset_seed']}_fold-{config['n_splits']}_rate-{config['train_split_rate']}_all-train-{config.get('rest_all_train',True)}"
                        if "test_egge_file" in config:
                            config['test_edge_file'] = config['test_egge_file']
                            del config['test_egge_file']
                            with open(config_file, "w") as f:
                                json.dump(config, f)

                        if not "train_edge_file" in config:
                            config['train_edge_file'] = data_files[cell_type][tag]['train'][config['split_id']-1]
                            config['test_edge_file'] = data_files[cell_type][tag]['test'][config['split_id'] - 1]
                            config['val_edge_file'] = data_files[cell_type][tag]['val'][config['split_id'] - 1]
                            with open(config_file, "w") as f:
                                json.dump(config, f)
                config_dir = os.path.join(log_dir, cell_type, model, comment, "configs")
                if os.path.exists(config_dir):
                    files = [file for file in os.listdir(config_dir) if file.endswith("config.csv")]
                    for file in files:
                        file = os.path.join(config_dir, file)
                        config = pd.read_csv(file)
                        if "train_edge_file" not in config.columns:
                            config['train_edge_file'] = None
                            config['test_edge_file'] = None
                            config['val_edge_file'] = None
                            for idx, row in config.iterrows():
                                tag = f"seed-{row['dataset_seed']}_fold-{row['n_splits']}_rate-{row['train_split_rate']}_all-train-{row.get('rest_all_train',True)}"
                                config.loc[idx, "train_edge_file"] = data_files[cell_type][tag]['train'][row['split_id']-1]
                                config.loc[idx, "val_edge_file"] = data_files[cell_type][tag]['val'][row['split_id'] - 1]
                                config.loc[idx, "test_edge_file"] = data_files[cell_type][tag]['test'][row['split_id'] - 1]
                            config.to_csv(file, index=False)


def remove_checkpoints(log_dir, comment="debug"):
    for cell_type in os.listdir(log_dir):
        for model in os.listdir(os.path.join(log_dir, cell_type)):
            tgt_dir = os.path.join(log_dir, cell_type, model, comment)
            if not os.path.exists(tgt_dir):
                continue
            for split in os.listdir(tgt_dir):
                split_dir = os.path.join(log_dir, cell_type, model, comment, split)
                if model not in ['DeepWalk', 'Genie3', 'GRNBoost2']:
                    split_dir = os.path.join(log_dir, cell_type, model, comment, split, "lightning_logs")
                if not os.path.exists(split_dir):
                    continue
                for version in os.listdir(split_dir):
                    checkpoint_dir = os.path.join(split_dir, version, "checkpoints")
                    if os.path.exists(checkpoint_dir):
                        shutil.rmtree(checkpoint_dir)




if __name__=="__main__":
    root_dirs = {
                 "pbmc": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC/graph/hg19_HOCOMOCOv11",
                 "lung": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/Lung_GSM4508936/graph/hg19_HOCOMOCOv11",
                 "ad": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_GSE174367/graph_Control/hg38_HOCOMOCOv11",
                 "pbmc_signac": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/PBMC_signac/graph/hg19_HOCOMOCOv11",
                 "earlyAD": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/earlyAD/graph/hg38_HOCOMOCOv11",
                 "lateAD": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/lateAD/graph/hg38_HOCOMOCOv11",
                 "nonAD": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/nonAD/graph/hg38_HOCOMOCOv11",
                 "early_lateAD": "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_epigenome/early_lateAD/graph/hg38_HOCOMOCOv11",
                 }
    cell_types = {"pbmc": ["B_naive", "CD14_Mono", "CD4_TCM", "CD8_Naive"],
                  "lung": ["Vascular_endothelial_cells", "Ciliated_epithelial_cells", "Bronchiolar_and_alveolar_epithelial_cells", "Lymphatic_endothelial_cells","Lymphoid_cells","Megakaryocytes","Megakaryocytes","Neuroendocrine_cells","Myeloid_cells"],
                  #"lung": ["Myeloid_cells"],
                  "pbmc_signac": ["CD14+_Monocytes", "B_cell_progenitor", "CD16+_Monocytes", "CD4_Memory"],
                  "earlyAD": ["Ast", "Ex", "In", "Microglia", "Oligo", "OPC"],
                  #"lateAD": ["Ast", "Ex", "In", "Microglia", "Oligo", "OPC"],
                  "lateAD": ["Ast","Microglia", "Oligo", "OPC"],
                  "early_lateAD": ["Ast", "Ex", "In", "Microglia", "OPC"],
                  "nonAD": ["Ast", "Ex", "In", "Microglia", "OPC"],
                  }
    key = "lung"
  #  key = "early_lateAD"
    comment = "debuggg_con"
    comment = 'first'
    # comment = 'three_control'
   # run_with_genie3(root_dirs[key], cell_types[key], comment=comment)
    # run_with_grnboost2(root_dirs[key], cell_types[key], comment=comment)
   # run_with_deepwalk(root_dirs[key], cell_types[key], comment=comment)
    # run_with_deepwalk(root_dirs[key], cell_types[key], comment=comment, dot_decode=False)
    #run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='DeepTFni')

    # run_with(root_dirs[key], cell_types[key], comment=comment, model_cls='GraFRankAE', accelerator='auto')

    run_with_genelink(root_dirs[key], cell_types[key], comment=comment)
    # patch_dataset_file("/mnt/7287870B13F6EA82/xjl/DeepTFni/code/src/experiment/data",
    #                "/mnt/7287870B13F6EA82/xjl/DeepTFni/code/src/experiment/lightning_logs")