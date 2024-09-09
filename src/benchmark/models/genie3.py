import os
import time
import numpy as np
import pandas as pd
from arboreto.algo import genie3
from src.benchmark.models.base_model import BaseModel
import pyscenic
from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
class Genie3(BaseModel):
    def __init__(self, root_dir, scrna_file, scrna_feature_file, node_file, **config):
        self.scrna_file = scrna_file
        self.scrna_feature_file = scrna_feature_file
        self.node_file = node_file
        self.config = config
        self.root_dir = root_dir

    def run(self, scrna_file, tf_names=None):
        data = pd.read_csv(scrna_file, index_col=0)
        genes = tf_names if tf_names is not None else list(data.index)
        other_gene = list(set(data.index) - set(genes))
        all_gene = list(genes) + other_gene
        genes2id = {key: i for i, key in enumerate(all_gene)}
        association = genie3(expression_data=data.T, tf_names=list(genes), seed=0, verbose=True)
        network = pd.DataFrame(np.zeros((len(genes), len(genes2id))), index=genes, columns=all_gene)
        for idx, row in association.iterrows():
            network.loc[row['TF'], row['target']] = row['importance']
        return network

    def train(self, **data):
        self.log_dir = os.path.join(self.root_dir, f"{time.time()}")
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        nodes = list(np.loadtxt(self.node_file, dtype=str))
        if "genie3" in self.scrna_feature_file:
            tgt_file = self.scrna_feature_file
        else:
            network = self.run(self.scrna_file)
            tgt_file = os.path.join(self.log_dir, "genie3_associations.csv")
            network.to_csv(tgt_file, sep='\t')
        network = pd.read_csv(tgt_file, index_col=0, sep='\t')
        rec = network.loc[nodes, nodes]
        self.rec = rec.values

    def evaluate(self, adj, edge_index, label, threshold=None, **data):
        metircs = self.evaluate_fn(self.rec, edge_index, label, threshold=threshold)
        return metircs, self.rec


if __name__=="__main__":
    from src.benchmark.models.evaluator import Evaluator

    target = 'CD14_Mono'
    root_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/raw_data/AD_GSE174367/graph/hg38_HOCOMOCOv11/ASC"
    adj_file = os.path.join(root_dir, f"x_graph.txt")
    node_file = os.path.join(root_dir, f"x_node.txt")
    scrna_feature_file = os.path.join(root_dir, f"x_scRNA_genie3_feature.txt")
    # scrna_feature_file = os.path.join(root_dir, f"x_scRNA_grnboost2_feature.txt")
    atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
    scrna_file = os.path.join(root_dir, "tmp", "just_scrna.csv")
    model_cls = "DeepWalk"
    model_cls = "Genie3"
    parser = Evaluator.add_argparse_args(name=target, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                         comment="test",
                                         atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                         model_cls=model_cls)
    # parser = DeepWalk.add_argparse_args(parser)
    config = parser.parse_args()
    # config.data_dir = data_dir
    Evaluator.run_single_fold(**vars(config))