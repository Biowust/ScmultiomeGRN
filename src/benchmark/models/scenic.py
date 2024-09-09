import os
import glob
import pickle
import pandas as pd
import numpy as np

from dask.diagnostics import ProgressBar

from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2

from ctxcore.rnkdb import FeatherRankingDatabase as RankingDatabase
from pyscenic.utils import modules_from_adjacencies, load_motifs
from pyscenic.prune import prune2df, df2regulons
from pyscenic.aucell import aucell

import seaborn as sns
import scipy.io as sio
from arboreto.utils import load_tf_names
from arboreto.algo import grnboost2, genie3
from tqdm import tqdm
from src.utils import load_mtx


def run_grnboost2(scrna_dir, node_file, use_variable_feature=True):
    genes = pd.read_csv(node_file, names=['gene'])['gene'].values
    data = load_mtx(scrna_dir, use_variable_feature=use_variable_feature, additional_genes=genes)
    other_gene = list(set(data.index)-set(genes))
    all_gene = list(genes)+other_gene
    genes2id = {key:i for i, key in enumerate(all_gene)}
    association = grnboost2(expression_data=data.T, tf_names=list(genes), seed=0, verbose=True)
    print(len(np.unique(association['TF'])), len(np.unique(association['target'])))

    network = pd.DataFrame(np.zeros((len(genes), len(genes2id))), index=genes, columns=all_gene)
    for idx, row in tqdm(association.iterrows(), total=len(association)):
        network.loc[row['TF'], row['target']] = row['importance']
    return network, network[genes]

def run_genie3(scrna_dir, node_file, use_variable_feature=True):
    genes = pd.read_csv(node_file, names=['gene'])['gene'].values
    data = load_mtx(scrna_dir, use_variable_feature=use_variable_feature, additional_genes=genes)
    other_gene = list(set(data.index)-set(genes))
    all_gene = list(genes)+other_gene
    genes2id = {key:i for i, key in enumerate(all_gene)}
    association = genie3(expression_data=data.T, tf_names=list(genes), seed=0, verbose=True)
    print(len(np.unique(association['TF'])), len(np.unique(association['target'])))

    network = pd.DataFrame(np.zeros((len(genes), len(genes2id))), index=genes, columns=all_gene)
    for idx, row in tqdm(association.iterrows(), total=len(association)):
        network.loc[row['TF'], row['target']] = row['importance']
    return network, network[genes]


if __name__=="__main__":
    scrna_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/raw_data/PBMC/pbmc_sep_data/CD14 Mono/scrna"
    node_file = "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/raw_data/PBMC/graph/hg19_HOCOMOCOv11/CD14_Mono/x_node.txt"
    import time
    start = time.time()
    run_grnboost2(scrna_dir, node_file)
    end = time.time()
    run_genie3(scrna_dir, node_file)
    end2 = time.time()
    print(end-start, end2-end)
