#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import time

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

import numpy as np
import pandas as pd
import torch

from src.benchmark.deepwalk import graph
from src.benchmark.deepwalk import walks as serialized_walks
from gensim.models import Word2Vec
from src.benchmark.deepwalk.skipgram import Skipgram

from six.moves import range

import psutil
from multiprocessing import cpu_count

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass


from sklearn.svm import LinearSVR, SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from src.benchmark.models.base_model import BaseModel

from sklearn.linear_model import LogisticRegression


class DeepWalk(BaseModel):
    def __init__(self, root_dir, dot_decode=False, **config):
        self.config = config
        self.root_dir = root_dir
        self.log_dir = None
        self.dot_decode = dot_decode
        # self.regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        self.regr = make_pipeline(StandardScaler(), LogisticRegression())

    def load_graph(self, adj, undirected=False):
        row, col = np.nonzero(adj)
        G = graph.Graph()
        for x, y in zip(row, col):
            G[x].append(y)
            if undirected:
                G[y].append(x)
        G.make_consistent()
        return G

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument("--dot_decode", default='False', action='store_true')

        parser.add_argument('--matfile-variable-name', default='network',
                            help='variable name of adjacency matrix inside a .mat file.')

        parser.add_argument('--max-memory-data-size', default=1000000000, type=int,
                            help='Size to start dumping walks to disk, instead of keeping them in memory.')

        parser.add_argument('--number-walks', default=10, type=int,
                            help='Number of random walks to start at each node')

        parser.add_argument('--representation-size', default=16, type=int,
                            help='Number of latent dimensions to learn for each node.')

        parser.add_argument('--seed', default=0, type=int,
                            help='Seed for random walk generator.')

        parser.add_argument('--undirected', default=True, type=bool,
                            help='Treat graph as undirected.')

        parser.add_argument('--vertex-freq-degree', default=False, action='store_true',
                            help='Use vertex degree to estimate the frequency of nodes '
                                 'in the random walks. This option is faster than '
                                 'calculating the vocabulary.')

        parser.add_argument('--walk-length', default=40, type=int,
                            help='Length of the random walk started at each node')

        parser.add_argument('--window-size', default=5, type=int,
                            help='Window size of skipgram model.')

        parser.add_argument('--workers', default=1, type=int,
                            help='Number of parallel processes.')
        return parser

    def run(self, adj, undirected=True, number_walks=10, walk_length=40, max_memory_data_size=1000000000, seed=0,
            representation_size=16, window_size=5, workers=1, vertex_freq_degree=False, **config):

        output_dir = os.path.join(self.root_dir, f"{time.time()}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.log_dir = output_dir

        output = os.path.join(output_dir, "feature.csv")

        G = self.load_graph(adj, undirected)
        print("Number of nodes: {}".format(len(G.nodes())))

        num_walks = len(G.nodes()) * number_walks

        print("Number of walks: {}".format(num_walks))

        data_size = num_walks * walk_length

        print("Data size (walks*length): {}".format(data_size))

        if data_size < max_memory_data_size:
            print("Walking...")
            walks = graph.build_deepwalk_corpus(G, num_paths=number_walks,
                                                path_length=walk_length, alpha=0, rand=random.Random(seed))
            print("Training...")
            model = Word2Vec(walks, vector_size=representation_size, window=window_size, min_count=0, sg=1, hs=1,
                             workers=workers)
        else:
            print("Data size {} is larger than limit (max-memory-data-size: {}).  Dumping walks to disk.".format(
                data_size, max_memory_data_size))
            print("Walking...")

            walks_filebase = output + ".walks"
            walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, num_paths=number_walks,
                                                              path_length=walk_length, alpha=0,
                                                              rand=random.Random(seed),
                                                              num_workers=workers)

            print("Counting vertex frequency...")
            if not vertex_freq_degree:
                vertex_counts = serialized_walks.count_textfiles(walk_files, workers)
            else:
                # use degree distribution for frequency in tree
                vertex_counts = G.degree(nodes=G.nodes())

            print("Training...")
            walks_corpus = serialized_walks.WalksCorpus(walk_files)
            model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                             size=representation_size,
                             window=window_size, min_count=0, trim_rule=None, workers=workers)
        model.wv.save_word2vec_format(output)
        feature = pd.read_csv(output, skiprows=1, index_col=0, sep=' ', header=None)
        index = np.argsort(feature.index)
        feature = feature.iloc[index]
        return feature.values

    def train_clf(self, feature, edge_index, label):
        x = np.concatenate([feature[edge_index[0]], feature[edge_index[1]]], axis=1)
        self.regr.fit(x, label)

    def train(self, adj, edge_index, label, **data):
        feature = self.run(adj, **data, **self.config)
        self.feature = feature
        if self.dot_decode:
            rec = self.sigmoid(feature @ feature.T)
        else:
            x = np.concatenate([feature[edge_index[0]], feature[edge_index[1]]], axis=1)
            self.regr.fit(x, label)
            row, col = np.nonzero(np.ones_like(adj))
            all_x = np.concatenate([feature[row], feature[col]], axis=1)
            rec = self.regr.predict_proba(all_x)[:, 1].reshape(adj.shape)

        self.rec = rec

    def evaluate(self, adj, edge_index, label, threshold=None, **data):
        rec = self.rec
        metircs = self.evaluate_fn(self.rec, edge_index, label, threshold=threshold)
        return metircs, rec


if __name__=="__main__":
    from src.benchmark.models.evaluator import Evaluator

    target = 'CD14_Mono'
    root_dir = "/mnt/7287870B13F6EA82/xjl/DeepTFni/raw_data/AD_GSE174367/graph/hg38_HOCOMOCOv11/ASC"
    adj_file = os.path.join(root_dir, f"x_graph.txt")
    node_file = os.path.join(root_dir, f"x_node.txt")
    scrna_feature_file = os.path.join(root_dir, f"x_scRNA_genie3_feature.txt")
    atac_feature_file = os.path.join(root_dir, f"x_atac_rp_score_feature.txt")
    scrna_file = os.path.join(root_dir, "tmp", "scrna.csv")
    model_cls = "DeepWalk"
    # model_cls = "Genie3"
    parser = Evaluator.add_argparse_args(name=target, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                         comment="test",
                                         atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                         model_cls=model_cls)
    parser = DeepWalk.add_argparse_args(parser)
    config = parser.parse_args()
    # config.data_dir = data_dir
    Evaluator.run_single_fold(**vars(config))



