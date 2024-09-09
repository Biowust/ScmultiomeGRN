from typing import Optional
from typing import Union, Tuple
import torch

from torch import Tensor
import torch.nn as nn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import OptPairTensor, Adj, Size, OptTensor
from torch_geometric.utils import softmax
# from torch_sparse import SparseTensor
import torch.nn.functional as F

class GraFrankConv(MessagePassing):
    """
    Modality-specific neighbor aggregation in GraFrank implemented by stacking message-passing layers that are
    parameterized by friendship attentions over individual node features and pairwise link features.
    """

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, normalize: bool = False,
                 bias: bool = True, **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'add')
        super(GraFrankConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.negative_slope = 0.1
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)

        self.self_linear = nn.Linear(in_channels[1], out_channels, bias=True)
        self.message_linear = nn.Linear(in_channels[0], out_channels, bias=True)

        self.attn = nn.Linear(out_channels, 1, bias=True)
        self.attn_i = nn.Linear(out_channels, 1, bias=True)

        self.lin_l = nn.Linear(out_channels, out_channels, bias=True)
        self.lin_r = nn.Linear(out_channels, out_channels, bias=True)

        self.reset_parameters()
        self.dropout = 0.1

        # self.lin_edge = nn.Linear(32*32, 32*32, bias=False)
        # self.edge_conv = nn.Sequential(
        #     torch.nn.Conv2d(1, 1, (3, 3)),
        #     torch.nn.Conv2d(1, 1, (3, 3)),
        #     torch.nn.MaxPool2d((2, 2)),
        #     # nn.Dropout(0.1)
        # )

        self.edge_conv = nn.Sequential(
            torch.nn.Conv2d(1, 1, (3, 3)),
            torch.nn.Conv2d(1, 1, (3, 3)),
            torch.nn.MaxPool2d((2, 2)),
            nn.Dropout(0.1)
        )
        # self.lin = nn.Linear(14*14, 14*14, bias=True)

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, edge_attr: OptTensor = None,
                size: Size = None) -> Tensor:
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)

        x_l, x_r = x[0], x[1]
        self_emb = self.self_linear(x_r)
        alpha_i = self.attn_i(self_emb)
        out = self.propagate(edge_index, x=(x_l, x_r), alpha=alpha_i, edge_attr=edge_attr, size=size)
        out = self.lin_l(out) + self.lin_r(self_emb)  # dense layer.

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)
        return out

    def message(self, x_j: Tensor, alpha_i: Tensor, edge_attr: Tensor, index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        # 有边特征的时候
        edge_attr = torch.unsqueeze(edge_attr, 1)
        a = self.edge_conv(edge_attr)
        a = a.reshape(a.shape[0], -1)
        # 没边特征的变体，就是上面3行换成下面这一行
        # a = torch.zeros((x_j.shape[0], 0)).to(DEVICE)

        message = torch.cat([x_j, a], dim=-1)

        # message = x_j
        out = self.message_linear(message)
        alpha = self.attn(out) + alpha_i
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = out * alpha
        return out

    def message_and_aggregate(self, adj_t) -> Tensor:
        pass

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
                                   self.out_channels)


class CrossModalityAttention(nn.Module):
    """
    Cross-Modality Fusion in GraFrank implemented by an attention mechanism across the K modalities.
    """

    def __init__(self, hidden_channels):
        super(CrossModalityAttention, self).__init__()
        self.hidden_channels = hidden_channels
        self.multi_linear = nn.Linear(hidden_channels, hidden_channels, bias=True)
        self.multi_attn = nn.Sequential(self.multi_linear, nn.Tanh(), nn.Linear(hidden_channels, 1, bias=True))
        self.dropout = nn.Dropout(0.1)

    def forward(self, modality_x_list):
        """
        :param modality_x_list: list of modality-specific node embeddings.
        :return: final node embedding after fusion.
        """
        # result : 768 * 4 * 64
        # result = torch.cat([x.unsqueeze(-2) for x in modality_x_list], -2)  # [...., K, hidden_channels]
        result = torch.cat([x.relu().unsqueeze(-2) for x in modality_x_list], -2)  # [...., K, hidden_channels]
        # wts : 768 * 4
        wts = torch.softmax(self.multi_attn(result).squeeze(-1), dim=-1)
        # wts.unsqueeze(-1) : 768 * 4 * 1
        # self.multi_linear(result) : 768 * 4 * 64
        return torch.sum(wts.unsqueeze(-1) * self.multi_linear(result), dim=-2)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__, self.hidden_channels,
                                   self.hidden_channels)


class EdgeDropout(nn.Module):
    def __init__(self, p=0.05):
        super(EdgeDropout, self).__init__()
        assert p>=0
        self.register_buffer("p", torch.tensor(p))

    def forward(self, edge_index, edge_weight):
        if self.training:
            mask = torch.rand(edge_index.shape[1], device=edge_weight.device)
            mask = torch.floor(mask+self.p).type(torch.bool)
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]/self.p
        return edge_index, edge_weight


class GraphConvolution(nn.Module):
    def __init__(self, input_dim, output_dim, dropout, bias=True):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            nn.init.zeros_(self.bias)
        else:
            self.register_parameter('bias', None)
        # self.dropout = nn.Dropout(dropout)
        self.edge_dropout = EdgeDropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        # inputs: (N, n_channels), adj: sparse_matrix (N, N)
        # input = self.dropout(input)
        support = torch.mm(input, self.weight)
        # idx, val = self.edge_dropout(adj._indices(), adj._values())
        # adj = torch.sparse_coo_tensor(indices=idx, values=val, size=adj.size())
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output = output + self.bias
        return output


class InnerProductDecoder(nn.Module):
    '''
    内积用来做decoder，用来生成邻接矩阵
    '''
    def __init__(self, in_dim=512, hidden_dim=256, dropout=0.1):
        super(InnerProductDecoder, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.mlp = nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, z):
        # z = self.dropout(z);
        shape = z.shape
        _A = z.expand(shape[0], shape[0], shape[1])
        _B = _A.permute(1, 0, 2)
        res = torch.cat((_B, _A), dim=-1)
        res.squeeze(1)
        out = self.mlp(res)
        out = out.squeeze(-1)
        adj = (out+out.T)/2
        
        return adj
