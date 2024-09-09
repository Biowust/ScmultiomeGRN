import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, recall_score, roc_curve, precision_recall_curve, auc
from src.utils import metric_fn

class BaseModel():

    def metrics(self, predict, label, threshold=None):
        return metric_fn(predict, label, threshold=threshold)

    def evaluate_fn(self, rec, edge_index, label, threshold=None):
        pred = rec[edge_index[0], edge_index[1]]
        metrics = self.metrics(pred, label, threshold=threshold)
        return metrics

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def train(self, **data):
        raise NotImplementedError

    def evaluate(self, **data):
        raise  NotImplementedError