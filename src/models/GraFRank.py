import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.layers import GraFrankConv, CrossModalityAttention, GraphConvolution, InnerProductDecoder, EdgeDropout

class GraFrank(nn.Module):
    """
    GraFrank Model for Multi-Faceted Friend Ranking with multi-modal node features and pairwise link features.
    (a) Modality-specific neighbor aggregation: modality_convs
    (b) Cross-modality fusion layer: cross_modality_attention
    """

    def __init__(self, in_channels, hidden_channels, edge_channels, num_layers, input_dim_list):
        """
        :param in_channels: total cardinality of node features.
        :param hidden_channels: latent embedding dimensionality.
        :param edge_channels: number of link features.
        :param num_layers: number of message passing layers.
        :param input_dim_list: list containing the cardinality of node features per modality.
        """
        super(GraFrank, self).__init__()
        self.num_layers = num_layers
        self.modality_convs = nn.ModuleList()
        self.edge_channels = edge_channels
        # we assume that the input features are first partitioned and then concatenated across the K modalities.
        self.input_dim_list = input_dim_list

        for inp_dim in self.input_dim_list:
            modality_conv_list = nn.ModuleList()
            for i in range(num_layers):
                # in_channels = in_channels if i == 0 else hidden_channels
                # modality_conv_list.append(GraFrankConv((inp_dim + edge_channels, inp_dim), hidden_channels))
                if i == 0:
                    modality_conv_list.append(GraFrankConv((inp_dim + edge_channels, inp_dim), hidden_channels))
                else:
                    modality_conv_list.append(GraFrankConv((hidden_channels + edge_channels, hidden_channels), hidden_channels))
            self.modality_convs.append(modality_conv_list)
        # print(self.modality_convs)
        # exit()
        self.cross_modality_attention = CrossModalityAttention(hidden_channels)

    # def forward(self, x, adjs, edge_attrs):
    #     """ Compute node embeddings by recursive message passing, followed by cross-modality fusion.
    #
    #     :param x: node features [B', in_channels] where B' is the number of nodes (and neighbors) in the mini-batch.
    #     :param adjs: list of sampled edge indices per layer (EdgeIndex format in PyTorch Geometric) in the mini-batch.
    #     :param edge_attrs: [E', edge_channels] where E' is the number of sampled edge indices per layer in the mini-batch.
    #     :return: node embeddings. [B, hidden_channels] where B is the number of target nodes in the mini-batch.
    #     """
    #     adjs = [adjs]
    #     result = []
    #     for k, convs_k in enumerate(self.modality_convs):
    #         emb_k = None
    #         # forward 函数的参数 adjs edge_attrs 是多层的adj和edge 下面就是得到每一层的输入
    #         # for i, ((edge_index, _, size), edge_attr) in enumerate(zip(adjs, edge_attrs)):
    #         print(adjs)
    #         exit()
    #         # print()
    #         # print(adjs[0]._values())
    #         # for i, ((edge_index, _, size, _, _), edge_attr) in enumerate(zip(adjs, edge_attrs)):
    #         if True:
    #             i = 0
    #             edge_index = adjs[0]._indices()
    #             size = adjs[0].size()
    #             edge_attr = edge_attrs[0]
    #             x_target = x[:size[1]]  # Target nodes are always placed first.
    #             x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
    #             x_target_list = torch.split(x_target, split_size_or_sections=self.input_dim_list, dim=-1)
    #
    #             x_k, x_target_k = x_list[k], x_target_list[k]
    #             # 第一层的emb_k 好像没有用到
    #             emb_k = convs_k[i]((x_k, x_target_k), edge_index, edge_attr=edge_attr)
    #             if i != self.num_layers - 1:
    #                 emb_k = emb_k.relu()
    #                 emb_k = F.dropout(emb_k, p=0.5, training=self.training)
    #
    #         result.append(emb_k)
    #
    #     # result 里面 4个 768 * 64大小的Tensor
    #     # 经过self.cross_modality_attention之后，变成1个768 * 64
    #     return self.cross_modality_attention(result)

    def full_forward(self, x, edge_index, edge_attr):
        """ Auxiliary function to compute node embeddings for all nodes at once for small graphs.

        :param x: node features [N, in_channels] where N is the total number of nodes in the graph.
        :param edge_index: edge indices [2, E] where E is the total number of edges in the graph.
        :param edge_attr: link features [E, edge_channels] across all edges in the graph.
        :return: node embeddings. [N, hidden_channels] for all nodes in the graph.
        """
        # x_list : 2708 350, 2708 350, 2708 350, 2708 383
        x_list = torch.split(x, split_size_or_sections=self.input_dim_list, dim=-1)  # modality partition
        result = []
        for k, convs_k in enumerate(self.modality_convs):
            x_k = x_list[k]
            emb_k = x_k
            for i, conv in enumerate(convs_k):
                # emb_k = conv(x_k, edge_index, edge_attr=edge_attr)
                emb_k = conv(emb_k, edge_index, edge_attr=edge_attr)
                # emb_k = emb_k.relu()
                if i != self.num_layers - 1:
                    emb_k = emb_k.relu()
                    emb_k = F.dropout(emb_k, p=0.1, training=self.training)
            result.append(emb_k) # 每一个append的embedding都是2708, 64
        # print(self.cross_modality_attention(result))
        # return result[0] + result[1]
        return self.cross_modality_attention(result)


class GraFRankModelAE(nn.Module):
    def __init__(self, atac_feat_dim, scrna_feat_dim, fusion_dim=512, edge_channels=36, num_layers=2, hidden_dim=256, dropout=0):
        super(GraFRankModelAE, self).__init__()
        self.fusion = GraFrank(atac_feat_dim+scrna_feat_dim, fusion_dim, edge_channels, num_layers, [scrna_feat_dim, atac_feat_dim])
        self.gc1 = GraphConvolution(fusion_dim, hidden_dim, dropout=0.05)
        self.ip = InnerProductDecoder(in_dim=hidden_dim*2, dropout=dropout)
        self.edge_dropout = EdgeDropout(0.95)


    def encode(self, input, adj_norm, edge_attr):
        fusioned = self.fusion.full_forward(input, adj_norm._indices(), edge_attr)
        output = self.gc1(fusioned, adj_norm)
        return output

    def forward(self, atac_feature, scrna_feature, adj_norm, edge_attr):
        output = self.encode(torch.cat((scrna_feature, atac_feature), dim=1), adj_norm, edge_attr)
        return self.ip(output)


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, *args):
        return args[0]

    def full_forward(self, *args):
        return args[0]


class GraFRankModelAEVariant(nn.Module):
    def __init__(self, atac_feat_dim, scrna_feat_dim, use_atac_feature=False, use_scrna_feature=False, use_edge_attr=False, fusion_dim=512, edge_channels=36, num_layers=2, hidden_dim=256, dropout=0):
        super(GraFRankModelAEVariant, self).__init__()

        if use_atac_feature and use_scrna_feature:
            self.fusion = GraFrank(atac_feat_dim + scrna_feat_dim, fusion_dim, edge_channels, num_layers,
                                   [scrna_feat_dim, atac_feat_dim])
        else:
            assert not use_edge_attr
            self.fusion = Identity()
            if not use_scrna_feature and not use_atac_feature:
                pass
            elif use_atac_feature:
                fusion_dim = atac_feat_dim
            elif use_scrna_feature:
                fusion_dim = scrna_feat_dim
            else:
                fusion_dim = scrna_feat_dim

        self.gc1 = GraphConvolution(fusion_dim, hidden_dim, dropout=0.05)
        self.ip = InnerProductDecoder(in_dim=hidden_dim*2, dropout=dropout)
        self.edge_dropout = EdgeDropout(0.95)
        self.use_edge_attr = use_edge_attr
        self.use_atac_feature = use_atac_feature
        self.use_scrna_feature = use_scrna_feature

    def encode(self, input, adj_norm, edge_attr):
        fusioned = self.fusion.full_forward(input, adj_norm._indices(), edge_attr)
        output = self.gc1(fusioned, adj_norm)
        return output

    def forward(self, atac_feature, scrna_feature, adj_norm, edge_attr):
        if self.use_atac_feature and self.use_scrna_feature:
            input = torch.cat((scrna_feature, atac_feature), dim=1)
        elif self.use_atac_feature:
            input = atac_feature
        elif self.use_scrna_feature:
            input = scrna_feature
        else:
            input = torch.ones_like(scrna_feature)
        if not self.use_edge_attr:
            edge_attr = torch.zeros_like(edge_attr)
        output = self.encode(input, adj_norm, edge_attr)
        return self.ip(output)