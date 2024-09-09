import os
import copy
import random
import pandas as pd
import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import KFold


def load_adj(adj_file, sep='\t'):
    # loading ddjacency matrix
    data = pd.read_csv(adj_file, sep=sep, index_col=0)
    adj = data.values.astype(int)
    # Remove diagonal elements
    adj = adj - np.diag(np.diag(adj))
    assert np.diag(adj).sum() == 0
    return adj


def preprocess_graph(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    # return sparse_to_tuple(adj_normalized)
    return adj_normalized


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)

class GraphData():
    def __init__(self, **kwargs):
        self.data = kwargs

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.data


class GraphDataModule():
    def __init__(self, name, adj_file, node_file, atac_feature_file, scrna_feature_file, scrna_file=None,
                 train_edge_file=None, val_edge_file=None, test_edge_file=None, shape=None, mask_rate=0.0, mask_seed=666, mask_type='drop',
                 dataset_seed=666, train_split_rate=0.8, n_splits=10, data_dir="data", split_id=1, rest_all_train=True, **kwargs):
        super(GraphDataModule, self).__init__()
        self.adj_file = adj_file
        self.node_file = node_file
        self.atac_feature_file = atac_feature_file
        self.scrna_feature_file = scrna_feature_file
        self.scrna_file = scrna_file
        self.train_edge_file = train_edge_file
        self.val_edge_file = val_edge_file
        self.test_edge_file = test_edge_file
        self.seed = dataset_seed
        self.train_split_rate = train_split_rate
        self.n_splits = n_splits
        self.save_dir = os.path.join(data_dir, name)
        self.split_id = split_id
        self.shape = shape
        self.rest_all_train = rest_all_train
        self.mask_rate = mask_rate
        self.mask_seed = mask_seed
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.all_data = None
        self.masked = None
        self.mask_type = mask_type
        assert mask_type in ['flip', 'drop']

    @classmethod
    def add_argparse_args(cls, parent_parser, **kwargs):
        parent_parser.add_argument("--name", type=str)
        parent_parser.add_argument("--adj_file", type=str)
        parent_parser.add_argument("--node_file", type=str)
        parent_parser.add_argument("--atac_feature_file", type=str)
        parent_parser.add_argument("--scrna_feature_file", type=str)
        parent_parser.add_argument("--scrna_file", type=str)
        parent_parser.add_argument("--dataset_seed", type=int, default=666)
        parent_parser.add_argument("--train_split_rate", type=float, default=0.8)
        parent_parser.add_argument("--n_splits", type=int, default=10)
        parent_parser.add_argument("--data_dir", type=str, default="data")
        parent_parser.add_argument('--split_id', type=int, default=1)
        return parent_parser

    def compute_edge_attr(self, scrna_file, adj_norm, bins=16):
        rna = pd.read_csv(scrna_file, sep=',', index_col=0)
        edge_index_i, edge_index_j = adj_norm.row, adj_norm.col
        TF_index = self.node_names
        edge_attr = []
        for i, j in zip(edge_index_i, edge_index_j):
            tf1_rna = rna.loc[TF_index[i], :].values
            tf2_rna = rna.loc[TF_index[j], :].values
            tf1_rna = np.log10(tf1_rna + 1e-2)
            tf2_rna = np.log10(tf2_rna + 1e-2)
            H_T = np.histogram2d(tf1_rna, tf2_rna, bins=bins)
            H = H_T[0].T
            HT = (np.log10(H / len(tf1_rna) + 1e-4) + 4) / 4
            edge_attr.append(HT)
        edge_attr = np.array(edge_attr)
        return edge_attr

    def prepare_data(self):
        """
        adj对角线置0,
        train data：去除val和test的所有样本，预测的对角线0变1
        val data: 正负样本1：1，预测的对角线0变1
        test data: 正负样本1：1，无对角线元素，与train data共享输入graph
        all data: 预测adj,对角线为0，与train data共享输入graph
        """
        if self.all_data is not None:
            return
        self.node_names = pd.read_csv(self.node_file, names=['name'])['name'].values
        if not isinstance(self.train_edge_file, str) or not os.path.exists(self.train_edge_file):
            data_dir = self.split_dataset(self.adj_file, seed=self.seed, train_split_rate=self.train_split_rate,
                                          n_splits=self.n_splits, save_dir=self.save_dir, rest_all_train=self.rest_all_train)
            pattern = f"split-{self.split_id:03d}"
            files = [file for file in os.listdir(data_dir) if file.startswith(pattern)]
            self.train_edge_file = os.path.join(data_dir, [file for file in files if "train" in file][0])
            self.val_edge_file = os.path.join(data_dir, [file for file in files if "val" in file][0])
            self.test_edge_file = os.path.join(data_dir, [file for file in files if "test" in file][0])

        self.train_data = self.load_data(self.train_edge_file, mask_rate=self.mask_rate, mask_seed=self.mask_seed)
        self.val_data = self.load_data(self.val_edge_file)

        test_data = self.load_data(self.test_edge_file)
        self.test_data = copy.deepcopy(self.train_data)
        self.test_data.data['edge_index'] = test_data.data['edge_index']
        self.test_data.data['label'] = test_data.data['label']

        self.all_data = copy.deepcopy(self.train_data)
        adj = load_adj(self.adj_file)
        self.all_data.data['edge_index'] = np.vstack(np.nonzero(np.ones_like(adj)))
        self.all_data.data['label'] = np.reshape(adj, -1)

        self.scrna_feat_dim = self.all_data.data['scrna_feature'].shape[-1]
        self.atac_feat_dim = self.all_data.data['atac_feature'].shape[-1]

    def mask_interaction(self, row, col, label, mask_rate, mask_seed, mask_type):
        seed_everything(mask_seed)
        pos_ids = np.nonzero(label)[0]
        mask_ids = np.random.choice(pos_ids, size=int(len(pos_ids) * mask_rate), replace=False)
        mask = np.ones_like(label, dtype=bool)
        mask[mask_ids] = False
        masked = pd.DataFrame({"row": row[~mask],
                               "col": col[~mask],
                               "label": label[~mask]})
        if mask_type=="flip":
            label[~mask] = 0
        elif mask_type=="drop":
            label = label[mask]
            row = row[mask]
            col = col[mask]
        return (row, col, label), masked

    def load_data(self, file, mask_rate=0.0, mask_seed=666):
        edge = pd.read_csv(file)
        row = edge['row'].values
        col = edge['col'].values
        label = edge['label'].values

        if mask_rate>0.0:
            (row, col, label), self.masked = self.mask_interaction(row, col, label, self.mask_rate,
                                                                   self.mask_seed, self.mask_type)


        shape = file.split("_")[-1].split(".")[0].split('x')
        if self.shape is None:
            self.shape = (int(shape[0]), int(shape[1]))

        adj = np.zeros(self.shape).astype(int)
        adj[row, col] = label
        adj = adj + adj.T

        label = np.concatenate([label, label])
        row, col = np.concatenate([row, col]), np.concatenate([col, row])

        adj_norm = preprocess_graph(adj)
        pos_weight = (adj.shape[0] * adj.shape[1] - adj.sum()) / adj.sum()
        norm_weight = (adj.shape[0] * adj.shape[1]) / ((adj.shape[0] * adj.shape[1] - adj.sum()) * 2)
        scrna_feature = pd.read_csv(self.scrna_feature_file, sep='\t', index_col=0).values
        atac_feature = pd.read_csv(self.atac_feature_file, sep='\t', index_col=0).values
        edge_index = np.unique(np.vstack([row, col]), axis=1)
        np.fill_diagonal(adj, 1)
        label = adj[edge_index[0], edge_index[1]]

        edge_attr = self.compute_edge_attr(self.scrna_file, adj_norm)

        data = GraphData(adj=adj, adj_norm=adj_norm, pos_weight=pos_weight, norm_weight=norm_weight,
                         scrna_feature=scrna_feature, atac_feature=atac_feature, n_node=adj.shape[0],
                         edge_index=edge_index, label=label, edge_attr=edge_attr)
        print("load from", file)
        print(f"adj:{self.shape[0]}x{self.shape[1]}, edge:{label.sum()}, atac:{atac_feature.shape[0]}x{atac_feature.shape[1]}, \
scrna:{scrna_feature.shape[0]}x{scrna_feature.shape[1]}, edge_attr:{edge_attr.shape}, pos_weight:{pos_weight:.5f}, norm_weight:{norm_weight:.5f}")
        return data

    @classmethod
    def split_dataset(cls, adj_file, seed=666, train_split_rate=0.8, n_splits=10, save_dir=".", rest_all_train=True):
        save_dir = os.path.join(save_dir, f"seed-{seed}_fold-{n_splits}_rate-{train_split_rate}_all-train-{rest_all_train}")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            print(f"skip split_dataset with seed:{seed}, train_split_rate:{train_split_rate}, n_splits:{n_splits}")
            return save_dir

        adj = load_adj(adj_file, sep='\t')
        edge_pos = np.vstack(np.nonzero(np.triu(adj, k=1))).T
        neg_mask = np.triu(np.ones_like(adj) ^ adj, k=1)
        edge_neg = np.vstack(np.nonzero(neg_mask)).T
        seed_everything(seed)
        print(f"split_dataset with seed:{seed}, train_split_rate:{train_split_rate}, n_splits:{n_splits}")
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        for i, (train_val_pos_edge_idx, test_pos_edge_idx) in enumerate(kf.split(edge_pos)):
            train_num = int(len(train_val_pos_edge_idx) * train_split_rate)
            np.random.shuffle(train_val_pos_edge_idx)
            train_pos_edge_idx = train_val_pos_edge_idx[:train_num]
            val_pos_edge_idx = train_val_pos_edge_idx[train_num:]

            neg_edge_idx = np.random.permutation(len(edge_neg))
            test_neg_edge_idx = neg_edge_idx[:len(test_pos_edge_idx)]

            train_val_neg_mask = copy.deepcopy(neg_mask)
            train_val_neg_mask[edge_neg[test_neg_edge_idx][:, 0], edge_neg[test_neg_edge_idx][:, 1]] = 0
            np.fill_diagonal(train_val_neg_mask, 1)
            train_val_edge_neg = np.vstack(np.nonzero(train_val_neg_mask)).T
            train_val_neg_edge_idx = np.random.permutation(len(train_val_edge_neg))
            val_neg_edge_idx = train_val_neg_edge_idx[:len(train_val_pos_edge_idx)-train_num]
            if rest_all_train:
                train_neg_edge_idx = train_val_neg_edge_idx[len(train_val_pos_edge_idx)-train_num:]
            else:
                train_neg_edge_idx = train_val_neg_edge_idx[len(train_val_pos_edge_idx)-train_num: len(train_val_pos_edge_idx)]

            # val_neg_edge_idx = neg_edge_idx[len(test_pos_edge_idx):len(edge_pos)-train_num]
            # if rest_all_train:
            #     train_neg_edge_idx = neg_edge_idx[len(edge_pos)-train_num:]
            # else:
            #     train_neg_edge_idx = neg_edge_idx[len(edge_pos)-train_num: len(edge_pos)]

            train_pos_edge = edge_pos[train_pos_edge_idx]
            # train_neg_edge = edge_neg[train_neg_edge_idx]
            train_neg_edge = train_val_edge_neg[train_neg_edge_idx]

            val_pos_edge = edge_pos[val_pos_edge_idx]
            # val_neg_edge = edge_neg[val_neg_edge_idx]
            val_neg_edge = train_val_edge_neg[val_neg_edge_idx]

            test_pos_edge = edge_pos[test_pos_edge_idx]
            test_neg_edge = edge_neg[test_neg_edge_idx]

            train_edge_index = np.concatenate([train_pos_edge, train_neg_edge])
            train_edge_label = np.concatenate([np.ones(len(train_pos_edge)), np.zeros(len(train_neg_edge))]).astype(int)

            val_edge_index = np.concatenate([val_pos_edge, val_neg_edge])
            val_edge_label = np.concatenate([np.ones(len(val_pos_edge)), np.zeros(len(val_neg_edge))]).astype(int)

            test_edge_index = np.concatenate([test_pos_edge, test_neg_edge])
            test_edge_label = np.concatenate([np.ones(len(test_pos_edge)), np.zeros(len(test_neg_edge))]).astype(int)

            train_edge = pd.DataFrame({"row":train_edge_index[:, 0], "col":train_edge_index[:, 1], "label":train_edge_label})
            val_edge = pd.DataFrame({"row":val_edge_index[:, 0], "col":val_edge_index[:, 1], "label":val_edge_label})
            test_edge = pd.DataFrame({"row":test_edge_index[:, 0], "col":test_edge_index[:, 1], "label":test_edge_label})
            train_edge.to_csv(os.path.join(save_dir, f"split-{i+1:03d}_train-edge-{train_edge_label.sum()}_{adj.shape[0]}x{adj.shape[1]}.csv"), index=False)
            val_edge.to_csv(os.path.join(save_dir, f"split-{i+1:03d}_val-edge-{val_edge_label.sum()}_{adj.shape[0]}x{adj.shape[1]}.csv"), index=False)
            test_edge.to_csv(os.path.join(save_dir, f"split-{i+1:03d}_test-edge-{test_edge_label.sum()}_{adj.shape[0]}x{adj.shape[1]}.csv"), index=False)
        return save_dir