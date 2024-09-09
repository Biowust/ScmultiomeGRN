import os
import json
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score
try:
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger
except:
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import CSVLogger

from src.utils import metric_fn

class ModelBase(pl.LightningModule):
    def __init__(self):
        super(ModelBase, self).__init__()
        self.monitor = "val/loss/dataloader_idx_0"
        self.train_output_list = []
        self.test_output_list = []
        self.test_tag = None
        self._threshold = None

    def on_test_start(self):
        version = self.logger.version
        save_dir = self.logger.save_dir
        existed_logger_cls = [logger.__class__ for logger in self.trainer.loggers]
        if CSVLogger not in existed_logger_cls:
            self.trainer.loggers.append(CSVLogger(save_dir=save_dir, version=version))

    def metrics(self, predict, label, threshold=None):
        label = label.detach().cpu()
        predict = predict.detach().cpu()
        return metric_fn(predict, label, threshold=threshold)

    def loss_rec(self, preds, labels, norm=1.0, pos_weight=None):
        cost = norm * F.binary_cross_entropy_with_logits(preds, labels.float(), pos_weight=pos_weight)
        return cost

    def loss_kl(self, mu, logvar, n_nodes):
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 / n_nodes * torch.mean(torch.sum(
            1 + 2 * logvar - mu.pow(2) - logvar.exp().pow(2), 1))
        return KLD

    def on_fit_start(self):
        version = self.logger.version
        save_dir = self.logger.save_dir
        existed_logger_cls = [logger.__class__ for logger in self.trainer.loggers]
        if CSVLogger not in existed_logger_cls:
            self.trainer.loggers.append(CSVLogger(save_dir=save_dir, version=version))

    def training_step(self, batch, batch_idx=None):
        ans = self.share_step(batch, batch_idx=batch_idx)
        self.train_output_list.append(ans)
        return ans

    def on_train_epoch_end(self):
        for key, value in self.train_output_list[0].items():
            if "loss" in key:
                self.log(f"train/{key}", value, prog_bar=True)
        self.train_output_list.clear()

    def validation_step(self, batch, batch_idx=None, dataloader_idx=0):
        ans = self.share_step(batch, batch_idx=batch_idx)
        for key, value in ans.items():
            if "loss" in key:
                self.log(f"val/{key}", value)
        metrics, curve = self.metrics(ans['pred'], ans['label'])
        for key, value in metrics.items():
            self.log(f"val/{key}", value)
        return ans

    def on_test_epoch_start(self):
        self.metrics_output_list = []

    def set_tag(self, tag):
        self.test_tag = tag

    def set_threshold(self, threshold):
        self._threshold = threshold

    def test_step(self, batch, batch_idx=None, dataloader_idx=None):
        ans = self.share_step(batch, batch_idx=batch_idx)
        for key, value in ans.items():
            if "loss" in key:
                name = f"test/{key}" if self.test_tag is None else f"test/{self.test_tag}/{key}"
                self.log(name, value, prog_bar=True)
        metrics, curve = self.metrics(ans['pred'], ans['label'], threshold=self._threshold)
        for key, value in metrics.items():
            name = f"test/{key}" if self.test_tag is None else f"test/{self.test_tag}/{key}"
            self.log(name, value)
        score_file = "score_matrix.txt" if self.test_tag is None else f"{self.test_tag}_score_matrix.txt"
        curve_file = "curve.json" if self.test_tag is None else f"{self.test_tag}_curve.json"

        rec = ans['rec'].detach().cpu().numpy()
        np.savetxt(os.path.join(self.trainer.log_dir, score_file), rec)
        with open(os.path.join(self.trainer.log_dir, curve_file), "w") as f:
            json.dump(curve, f)

        return ans

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1000,
                                                            verbose=True, threshold=1e-4, threshold_mode='rel',
                                                            cooldown=0, min_lr=0, eps=1e-8)
        return [optimizer], [{"scheduler": lr_scheduler,
                              "monitor": self.monitor,
                              "interval": "epoch"}]