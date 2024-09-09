#! /usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import random
import time

import numpy as np
import pandas as pd

import scanpy as sc
from tqdm import tqdm

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.nn import functional as F

print(os.getcwd())
sys.path.append('/mnt/7287870B13F6EA82/xjl/DeepTFni/code')


from src.benchmark.models.base_model import BaseModel
from src.benchmark.genelink import scGNN
from src.benchmark.genelink.utils import scRNADataset, load_data, adj2saprse_tensor


class GENELink(BaseModel):
    def __init__(self, root_dir, **config):
        self.config = config
        self.root_dir = root_dir
        self.log_dir = None
        # self.log_dir = None
        # self.dot_decode = dot_decode
        # # self.regr = make_pipeline(StandardScaler(), SVR(C=1.0, epsilon=0.2))
        # self.regr = make_pipeline(StandardScaler(), LogisticRegression())

    @classmethod
    def add_argparse_args(cls, parser):
        parser.add_argument('--lr', type=float, default=3e-3, help='Initial learning rate.')
        parser.add_argument('--epochs', type=int, default= 90, help='Number of epoch.')
        parser.add_argument('--num_head', type=list, default=[3,3], help='Number of head attentions.')
        parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
        parser.add_argument('--hidden_dim', type=int, default=[128,64,32], help='The dimension of hidden layer')
        parser.add_argument('--output_dim', type=int, default=16, help='The dimension of latent layer')
        parser.add_argument('--batch_size', type=int, default=256, help='The size of each batch')
        parser.add_argument('--loop', type=bool, default=False, help='whether to add self-loop in adjacent matrix')
        parser.add_argument('--seed', type=int, default=8, help='Random seed')
        parser.add_argument('--Type',type=str,default='dot', help='score metric')
        parser.add_argument('--flag', type=bool, default=False, help='the identifier whether to conduct causal inference')
        parser.add_argument('--reduction',type=str,default='concate', help='how to integrate multihead attention')
        return parser

    def train(self, adj, edge_index, label, **data):

        output_dir = os.path.join(self.root_dir, f"{time.time()}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.log_dir = output_dir

        hidden_dim = self.config['hidden_dim']
        output_dim = self.config['output_dim']
        num_head = self.config['num_head']
        alpha = self.config['alpha']
        Type = self.config['Type']
        reduction = self.config['reduction']
        lr = self.config['lr']
        max_epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        flag = self.config['flag']
        device = "cuda" if torch.cuda.is_available() else 'cpu'

        scrna_data = pd.read_csv(self.config['scrna_file'], index_col=0).T
        node = pd.read_csv(self.config['node_file'], names=['node'])['node'].values
        adata = sc.AnnData(scrna_data)
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        adata = adata[:, node]

        loader = load_data(pd.DataFrame(adata.X.T))
        feature = loader.exp_data()
        tf = None
        train_data = np.stack([edge_index[0], edge_index[1], label]).T
        train_load = scRNADataset(train_data, feature.shape[0], flag=flag)
        adj = train_load.Adj_Generate(tf,loop=self.config['loop'])
        adj = adj2saprse_tensor(adj)

        train_data = torch.from_numpy(train_data)
        feature = torch.from_numpy(feature)
        # test_data = torch.from_numpy(test_data)
        # val_data = torch.from_numpy(validation_data)

        model = scGNN.GENELink(input_dim=feature.shape[1],
                        hidden1_dim=hidden_dim[0],
                        hidden2_dim=hidden_dim[1],
                        hidden3_dim=hidden_dim[2],
                        output_dim=output_dim,
                        num_head1=num_head[0],
                        num_head2=num_head[1],
                        alpha=alpha,
                        device=device,
                        type=Type,
                        reduction=reduction
                        )

        adj = adj.to(device)
        model = model.to(device)
        train_data = train_data.to(device)
        data_feature = feature.to(device)
        # test_data = test_data.to(device)
        # validation_data = val_data.to(device)

        optimizer = Adam(model.parameters(), lr=lr)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.99)

        for epoch in tqdm(range(max_epochs)):
            running_loss = 0.0

            for train_x, train_y in DataLoader(train_load, batch_size=batch_size, shuffle=True):
                model.train()
                optimizer.zero_grad()

                if flag:
                    train_y = train_y.to(device)
                else:
                    train_y = train_y.to(device).view(-1, 1)

                train_x = train_x.to(device)
                # train_y = train_y.to(device).view(-1, 1)
                pred = model(data_feature, adj, train_x)

                #pred = torch.sigmoid(pred)
                if flag:
                    pred = torch.softmax(pred, dim=1)
                else:
                    pred = torch.sigmoid(pred)
                loss_BCE = F.binary_cross_entropy(pred, train_y)

                loss_BCE.backward()
                optimizer.step()
                scheduler.step()

                running_loss += loss_BCE.item()
        self.model = model
        self.data_feature = data_feature
        self.adj = adj



    def evaluate(self, adj, edge_index, label, threshold=None, **data):
        model = self.model
        flag = self.config['flag']
        data_feature = self.data_feature
        adj = self.adj

        all_data = np.stack(np.meshgrid(np.arange(adj.shape[0]), np.arange(adj.shape[1]))).T
        all_data = torch.from_numpy(all_data.reshape(-1, all_data.shape[-1])).to(data_feature.device)
        model.eval()
        score = model(data_feature, adj, all_data)
        if flag:
            score = torch.softmax(score, dim=1)
        else:
            score = torch.sigmoid(score)
        score = score.reshape(adj.shape)
        score = (score+score.T)/2
        rec = score.detach().cpu().numpy()
        metrics = self.evaluate_fn(rec, edge_index, label.astype(int), threshold=threshold)
        return metrics, rec


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
    model_cls = "Genie3"
    model_cls = "GENELink"
    parser = Evaluator.add_argparse_args(name=target, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                         comment="test",
                                         atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                         model_cls=model_cls)
    parser = GENELink.add_argparse_args(parser)
    config = parser.parse_args()
    # config.data_dir = data_dir
    config.epochs = 2
    Evaluator.run_single_fold(**vars(config))



