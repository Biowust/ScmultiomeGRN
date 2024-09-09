import torch
from torch import nn, optim
import torch.nn.functional as F
from src.models.model_helper import ModelBase


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        return adj


class DeepTFni(ModelBase):
    def __init__(self, atac_feat_dim, hidden1=32, hidden2=16, dropout=0., lr=1e-5, **kwargs):
        super(DeepTFni, self).__init__()
        self.model = GCNModelVAE(atac_feat_dim, hidden1, hidden2, dropout)
        self.lr = lr

    def forward(self, adj_norm, atac_feature, **kwargs):
        recovered, mu, logvar = self.model(atac_feature, adj_norm)
        return recovered, mu, logvar

    def share_step(self, batch, batch_idx=None):
        label = batch['label']
        n_nodes = batch['n_node']
        norm = batch['norm_weight']
        pos_weight = batch['pos_weight']
        edge_index = batch['edge_index']
        rec, mu, logvar = self.forward(**batch)
        pred = rec[edge_index[0], edge_index[1]]
        # preds = torch.cat([pred, torch.diag(rec)])
        # labels = torch.cat([label, torch.ones(n_nodes, device=label.device)])
        loss_rec = self.loss_rec(pred, label, norm=norm, pos_weight=pos_weight)
        loss_kl = self.loss_kl(mu, logvar, n_nodes)
        loss = loss_kl+loss_rec

        adj_rec = torch.sigmoid(mu@mu.T)
        adj_pred = adj_rec[edge_index[0], edge_index[1]]
        return {"loss": loss,
                "loss_kl": loss_kl,
                "loss_rec": loss_rec,
                "label": label,
                "pred": adj_pred,
                "rec": adj_rec,
                "feature": mu}

