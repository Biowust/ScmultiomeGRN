from src.models.model_helper import ModelBase
from src.models.GraFRank import GraFRankModelAE, GraFRankModelAEVariant


class scMultiomeGRN(ModelBase):
    def __init__(self, atac_feat_dim, scrna_feat_dim, lr=1e-5, **kwargs):
        super(scMultiomeGRN, self).__init__()
        self.model = GraFRankModelAE(atac_feat_dim, scrna_feat_dim)
        self.lr = lr

    def forward(self, atac_feature, scrna_feature, adj_norm, edge_attr, **kwargs):
        return self.model(atac_feature, scrna_feature, adj_norm, edge_attr)

    def share_step(self, batch, batch_idx=None):
        label = batch['label']
        n_nodes = batch['n_node']
        norm = batch['norm_weight']
        pos_weight = batch['pos_weight']
        edge_index = batch['edge_index']
        rec = self.forward(**batch)
        pred = rec[edge_index[0], edge_index[1]]

        loss_rec = self.loss_rec(pred, label, norm=norm, pos_weight=pos_weight)
        loss = loss_rec

        return {"loss": loss,
                "loss_rec": loss_rec,
                "label": label,
                "pred": pred,
                "rec": rec,
                }


class scMultiomeGRNVariant(scMultiomeGRN):
    def __init__(self, atac_feat_dim, scrna_feat_dim, lr=1e-5, use_atac_feature=False, use_scrna_feature=False, use_edge_attr=False,  **kwargs):
        super(scMultiomeGRNVariant, self).__init__(atac_feat_dim, scrna_feat_dim, lr=lr, **kwargs)
        self.model = GraFRankModelAEVariant(atac_feat_dim, scrna_feat_dim, use_atac_feature=use_atac_feature,
                                            use_scrna_feature=use_scrna_feature, use_edge_attr=use_edge_attr)
