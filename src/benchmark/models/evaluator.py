import os
import json
import pprint
import time
import argparse
import random
import numpy as np
import pandas as pd

from src.benchmark.dataset import GraphDataModule
from src.benchmark.models.deepwalk import DeepWalk
from src.benchmark.models.genie3 import Genie3
from src.benchmark.models.grnboost2 import GRNBoost2
from src.benchmark.models.genelink import GENELink

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)


class Evaluator():

    @classmethod
    def add_argparse_args(cls, **kwargs):
        parser = argparse.ArgumentParser()
        parser = GraphDataModule.add_argparse_args(parser)
        parser.add_argument("--model_cls", type=str, default="DeepTFni")
        parser.add_argument("--model_seed", type=int, default=666)
        parser.set_defaults(**kwargs)
        return parser

    @classmethod
    def get_root_dir(cls, config):
        return os.path.join("lightning_logs", config['name'], config['model_cls'], config['comment'],
                            f"split_{config['split_id']}")

    @classmethod
    def run_single_fold(cls, data_module=GraphDataModule, **config):
        """
        val dataloader_idx_0: val data
        val dataloader_idx_1: test data
        test dataloader_idx_0: test data
        test_dataloader_idx_1: all data
        test_dataloader_idx: best model all data
        """
        model_seed = config['model_seed']
        model_cls = globals()[config['model_cls']]
        data = data_module(**config)
        data.prepare_data()
        config['atac_feat_dim'] = data.atac_feat_dim
        config['scrna_feat_dim'] = data.scrna_feat_dim
        config['root_dir'] = cls.get_root_dir(config)

        seed_everything(model_seed)
        model = model_cls(**config)
        print(config['model_cls'])
        model.train(**data.train_data.data)
        (test_metrics, test_curve), score_matrix = model.evaluate(**data.test_data.data)
        (all_metrics, all_curve), last_score = model.evaluate(threshold=test_metrics['threshold'], **data.all_data.data)

        log_dir = model.log_dir
        config['log_dir'] = os.path.abspath(log_dir)
        config['train_edge_file'] = os.path.abspath(data.train_edge_file)
        config['test_edge_file'] = os.path.abspath(data.test_edge_file)
        config['val_edge_file'] = os.path.abspath(data.val_edge_file)

        edge_index = data.test_data.data['edge_index']
        last_threshold = float(np.median(last_score[edge_index[0], edge_index[1]]))
        last_adj = (last_score > last_threshold) + 0
        last_adj = pd.DataFrame(last_adj, columns=data.node_names, index=data.node_names)
        last_adj.to_csv(os.path.join(log_dir, "last_all_adj.csv"))

        metrics = {}
        metrics.update(config)
        for key, value in test_metrics.items():
            metrics[f'test/{key}'] = value
        for key, value in all_metrics.items():
            metrics[f'all/{key}'] = value
        metrics['last_threshold'] = last_threshold
        metric_file = os.path.join(log_dir, "metrics.json")
        test_curve_file = os.path.join(log_dir, "last_test_curve.json")
        all_curve_file = os.path.join(log_dir, "last_all_curve.json")
        last_score_matrix_file = os.path.join(log_dir, "last_score_matrix.txt")
        if data.masked is not None:
            data.masked.to_csv(os.path.join(log_dir, "data_masked.csv"), index=False)
        with open(metric_file, "w") as f:
            json.dump(metrics, f, indent=2)
        with open(test_curve_file, "w") as f:
            json.dump(test_curve, f)
        with open(all_curve_file, "w") as f:
            json.dump(all_curve, f)
        np.savetxt(last_score_matrix_file, last_score)
        print("log dir:", log_dir)
        pprint.pprint(test_metrics)
        pprint.pprint(all_metrics)

        return log_dir

    @classmethod
    def run_all_fold(cls, **config):
        dirs = []
        for split_id in range(1, config['n_splits'] + 1):
            config['split_id'] = split_id
            print(f"begin split {split_id}/{config['n_splits']}")
            log_dir = cls.run_single_fold(**config)
            dirs.append(log_dir)
        ans = cls.collect_result(dirs)
        save_dir = os.path.join(os.path.dirname(cls.get_root_dir(config)), "configs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_stamp = "_".join(time.asctime().split()).replace(":", "-")
        file = os.path.join(save_dir, f"seed-{config['model_seed']}-{config['dataset_seed']}_{time_stamp}")
        # ans['best_adj'].to_csv(f"{file}_best_adj.csv")
        ans['last_all_adj'].to_csv(f'{file}_last_all_adj.csv')
        ans['configs'].to_csv(f'{file}_config.csv', index=False)
        return ans

    @classmethod
    def load_metric_file(cls, file):
        with open(file) as f:
            metrics = json.load(f)
        return metrics

    @classmethod
    def run_with_seeds(cls, model_seeds=(666,), dataset_seeds=(666,), **config):
        assert len(model_seeds) == len(dataset_seeds) or len(model_seeds) == 1 or len(dataset_seeds) == 1
        if len(model_seeds) == 1:
            model_seeds = list(model_seeds) * len(dataset_seeds)
        elif len(dataset_seeds) == 1:
            dataset_seeds = list(dataset_seeds) * len(model_seeds)
        ans = []
        for model_seed, dataset_seed in zip(model_seeds, dataset_seeds):
            config['model_seed'] = model_seed
            config['dataset_seed'] = dataset_seed
            ans.append(cls.run_all_fold(**config))

        save_dir = os.path.join(os.path.dirname(cls.get_root_dir(config)), "configs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # best_adj = sum([item['best_adj'] for item in ans])
        last_adj = sum([item['last_all_adj'] for item in ans])

        # best_adj = (best_adj>=(len(model_seeds)-2)).astype(int)
        last_adj = (last_adj >= (len(model_seeds) - 2)).astype(int)

        configs = pd.concat([item['configs'] for item in ans])
        time_stamp = "_".join(time.asctime().split()).replace(":", "-")
        file = os.path.join(save_dir,
                            f"merged_model_seed-{len(model_seeds)}-dataset_seed-{len(dataset_seeds)}_{time_stamp}")
        # best_adj.to_csv(f"best_adj_{file}.csv")
        last_adj.to_csv(f'last_all_adj_{file}.csv')
        configs.to_csv(f'config_{file}.csv', index=False)
        return {"last_all_adj": last_adj,
                # "best_adj": best_adj,
                "configs": configs}

    @classmethod
    def collect_result(cls, dirs):
        best_adj = []
        last_adj = []
        configs = []
        for dir in dirs:
            # best_adj.append(pd.read_csv(os.path.join(dir, "best_adj.csv"), index_col=0))
            last_adj.append(pd.read_csv(os.path.join(dir, "last_all_adj.csv"), index_col=0))
            config = cls.load_metric_file(os.path.join(dir, 'metrics.json'))
            # metric['best_adj_connection'] = (best_adj[-1].values-np.diag(np.diag(best_adj[-1].values))).sum()
            config['last_adj_connection'] = (last_adj[-1].values - np.diag(np.diag(last_adj[-1].values))).sum()
            configs.append(config)
        # best_adj = sum(best_adj)
        last_adj = sum(last_adj)
        # best_adj_matrix = (best_adj>=(0.6*len(dirs))).astype(int)
        last_adj_matrix = (last_adj >= (0.6 * len(dirs))).astype(int)
        configs = pd.DataFrame(configs)
        return {'last_all_adj': last_adj_matrix,
                # 'best_adj': best_adj_matrix,
                'configs': configs}