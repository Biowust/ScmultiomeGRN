import os
import sys
import argparse
import json
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
try:
    import pytorch_lightning as pl
except:
    import lightning.pytorch as pl


from src.models.deepTFni import DeepTFni
from src.models.model import scMultiomeGRN, scMultiomeGRNVariant
from src.dataset import GraphDataModule, DeepTFniDataModule


class CustomProgressBar(pl.callbacks.TQDMProgressBar):
    def get_metrics(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"):
        # don't show the version number
        items = super().get_metrics(trainer=trainer, pl_module=pl_module)
        items["v_num"] = trainer.logger.version
        return items

    def init_train_tqdm(self):
        """Override this to customize the tqdm bar for training."""
        bar = pl.callbacks.progress.tqdm_progress.Tqdm(
            desc=self.train_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=False,
            file=sys.stdout,
            smoothing=0,
            ncols=120,
        )
        return bar


class Trainer():

    @classmethod
    def add_argparse_args(cls, **kwargs):
        parser = argparse.ArgumentParser()
        parser = GraphDataModule.add_argparse_args(parser)
        parser.add_argument("--max_epochs", type=int, default=2000)
        parser.add_argument("--patience", type=int, default=100)
        parser.add_argument("--model_cls", type=str, default="DeepTFni")
        parser.add_argument("--model_seed", type=int, default=666)
        parser.add_argument("--comment", type=str, default="debug")
        parser.add_argument("--accelerator", type=str, default='auto')
        parser.set_defaults(**kwargs)
        return parser

    @classmethod
    def get_root_dir(cls, config):
        return os.path.join("lightning_logs", config['name'], config['model_cls'], config['comment'], f"split_{config['split_id']}")

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

        pl.seed_everything(model_seed)
        model = model_cls(**config)
        monitor = "val/loss/dataloader_idx_0"
        early_stop_callback = pl.callbacks.EarlyStopping(monitor=monitor, patience=config['patience'], verbose=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor, save_last=True, save_weights_only=True)
        callbacks = [early_stop_callback, CustomProgressBar(), checkpoint_callback]
        default_root_dir = cls.get_root_dir(config)
        trainer = pl.Trainer(max_epochs=config['max_epochs'], callbacks=callbacks, accelerator=config['accelerator'], default_root_dir=default_root_dir)
        trainer.fit(model, train_dataloaders=data.train_dataloader(), val_dataloaders=[data.val_dataloader(), data.test_dataloader()])
        model.set_tag("last_test")
        model.set_threshold(None)
        ans = trainer.test(model, data.test_dataloader())
        model.set_tag("last_all")
        model.set_threshold(ans[-1]['test/last_test/threshold'])
        trainer.test(model, data.all_dataloader())
        model.set_tag("best_test")
        model.set_threshold(None)
        ans = trainer.test(model, data.test_dataloader(), ckpt_path='best')
        model.set_tag("best_all")
        model.set_threshold(ans[-1]['test/best_test/threshold'])
        trainer.test(model, data.all_dataloader(), ckpt_path='best')

        edge_index = data.test_data.data['edge_index']
        best_score = np.loadtxt(os.path.join(trainer.log_dir, "best_all_score_matrix.txt"))
        best_threshold = np.median(best_score[edge_index[0], edge_index[1]])
        best_adj = (best_score>best_threshold)+0
        best_adj = pd.DataFrame(best_adj, columns=data.node_names, index=data.node_names)
        best_adj.to_csv(os.path.join(trainer.log_dir, "best_all_adj.csv"))

        last_score = np.loadtxt(os.path.join(trainer.log_dir, "last_all_score_matrix.txt"))
        last_threshold = np.median(last_score[edge_index[0], edge_index[1]])
        last_adj = (last_score>last_threshold)+0
        last_adj = pd.DataFrame(last_adj, columns=data.node_names, index=data.node_names)
        last_adj.to_csv(os.path.join(trainer.log_dir, "last_all_adj.csv"))

        torch.cuda.empty_cache()
        config['log_dir'] = os.path.abspath(trainer.log_dir)
        config['train_edge_file'] = os.path.abspath(data.train_edge_file)
        config['test_edge_file'] = os.path.abspath(data.test_edge_file)
        config['val_edge_file'] = os.path.abspath(data.val_edge_file)
        if data.masked is not None:
            data.masked.to_csv(os.path.join(trainer.log_dir, "data_masked.csv"), index=False)
        with open(os.path.join(trainer.log_dir, "config.json"), "w") as f:
            json.dump(config, f)
        print(checkpoint_callback.best_model_path)
        # model.load_state_dict(torch.load(checkpoint_callback.best_model_path)['state_dict'])
        # trainer.test(model, [data.test_dataloader(), data.all_dataloader()], ckpt_path='best')
        return trainer.log_dir


    @classmethod
    def run_all_fold(cls, **config):
        dirs = []
        for split_id in range(1, config['n_splits']+1):
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
        ans['best_all_adj'].to_csv(f"{file}_best_all_adj.csv")
        ans['last_all_adj'].to_csv(f'{file}_last_all_adj.csv')
        ans['configs'].to_csv(f'{file}_config.csv', index=False)
        return ans

    @classmethod
    def load_metric_file(cls, file):
        metric = pd.read_csv(file)
        ans = {}
        for col in metric.columns:
            ans[col] = metric[col].dropna().iloc[-1]
        return ans

    @classmethod
    def run_with_seeds(cls, model_seeds=(666, ), dataset_seeds=(666, ), **config):
        assert len(model_seeds)==len(dataset_seeds) or len(model_seeds)==1 or len(dataset_seeds)==1
        if len(model_seeds)==1:
            model_seeds = list(model_seeds)*len(dataset_seeds)
        elif len(dataset_seeds)==1:
            dataset_seeds = list(dataset_seeds)*len(model_seeds)
        ans = []
        for model_seed, dataset_seed in zip(model_seeds, dataset_seeds):
            config['model_seed'] = model_seed
            config['dataset_seed'] = dataset_seed
            ans.append(cls.run_all_fold(**config))

        save_dir = os.path.join(os.path.dirname(cls.get_root_dir(config)), "configs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        best_adj = sum([item['best_all_adj'] for item in ans])
        last_adj = sum([item['last_all_adj'] for item in ans])

        best_adj = (best_adj>=(len(model_seeds)-2)).astype(int)
        last_adj = (last_adj>=(len(model_seeds)-2)).astype(int)

        configs = pd.concat([item['configs'] for item in ans])
        time_stamp = "_".join(time.asctime().split()).replace(":", "-")
        file = os.path.join(save_dir, f"merged_model_seed-{len(model_seeds)}-dataset_seed-{len(dataset_seeds)}_{time_stamp}")
        best_adj.to_csv(f"best_all_adj_{file}.csv")
        last_adj.to_csv(f'last_all_adj_{file}.csv')
        configs.to_csv(f'config_{file}.csv', index=False)
        return {"best_all_adj": best_adj,
                "last_all_adj": last_adj,
                "configs": configs}

    @classmethod
    def collect_result(cls, dirs):
        best_adj = []
        last_adj = []
        configs = []
        for dir in dirs:
            best_adj.append(pd.read_csv(os.path.join(dir, "best_all_adj.csv"), index_col=0))
            last_adj.append(pd.read_csv(os.path.join(dir, "last_all_adj.csv"), index_col=0))
            metric = cls.load_metric_file(os.path.join(dir, 'metrics.csv'))
            metric['best_adj_connection'] = (best_adj[-1].values-np.diag(np.diag(best_adj[-1].values))).sum()
            metric['last_adj_connection'] = (last_adj[-1].values-np.diag(np.diag(last_adj[-1].values))).sum()
            with open(os.path.join(dir, "config.json")) as f:
                config = json.load(f)
            config.update(metric)
            configs.append(config)
        best_adj = sum(best_adj)
        last_adj = sum(last_adj)
        best_adj_matrix = (best_adj>=(0.6*len(dirs))).astype(int)
        last_adj_matrix = (last_adj>=(0.6*len(dirs))).astype(int)
        configs = pd.DataFrame(configs)
        return {'best_all_adj': best_adj_matrix,
                'last_all_adj': last_adj_matrix,
                'configs': configs}


class OldEvaluator():

    @classmethod
    def add_argparse_args(cls, **kwargs):
        parser = argparse.ArgumentParser()
        parser = GraphDataModule.add_argparse_args(parser)
        parser.add_argument("--max_epochs", type=int, default=2000)
        parser.add_argument("--patience", type=int, default=100)
        parser.add_argument("--model_cls", type=str, default="DeepTFni")
        parser.add_argument("--model_seed", type=int, default=666)
        parser.add_argument("--comment", type=str, default="debug")
        parser.set_defaults(**kwargs)
        return parser

    @classmethod
    def get_root_dir(cls, config):
        return os.path.join("lightning_logs", config['name'], config['model_cls'], config['comment'], f"split_{config['split_id']}")

    @classmethod
    def eval_one_fold(cls, model=None, data_module=None, **config):
        """
        val dataloader_idx_0: val data
        val dataloader_idx_1: test data
        test dataloader_idx_0: test data
        test_dataloader_idx_1: all data
        test_dataloader_idx: eval best model on all data
        """
        if data_module is None:
            data_module = DeepTFniDataModule(**config)

        data = data_module
        data.prepare_data()

        config['atac_feat_dim'] = data.atac_feat_dim
        config['scrna_feat_dim'] = data.scrna_feat_dim
        if model is None:
            model_cls = globals()[config['model_cls']]
            model = model_cls(**config)
            checkpoint_file = os.path.join(config['checkpoint_dir'],
                                           f"checkpoint_k_{config['split_id']}.pt")
            checkpoint = torch.load(checkpoint_file, map_location='cpu')
            model.model.load_state_dict(checkpoint, strict=True)
            print("load model from", checkpoint_file)

        monitor = "val/loss/dataloader_idx_0"
        early_stop_callback = pl.callbacks.EarlyStopping(monitor=monitor, patience=config['patience'], verbose=True)
        checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor=monitor, save_last=True, save_weights_only=True)
        callbacks = [early_stop_callback, CustomProgressBar(), checkpoint_callback]
        default_root_dir = cls.get_root_dir(config)
        trainer = pl.Trainer(max_epochs=config['max_epochs'], callbacks=callbacks, accelerator="auto",
                             default_root_dir=default_root_dir)

        trainer.test(model, [data.test_dataloader(), data.all_dataloader(), data.train_dataloader()])
        torch.cuda.empty_cache()

        config['log_dir'] = os.path.abspath(trainer.log_dir)
        with open(os.path.join(trainer.log_dir, "config.json"), "w") as f:
            json.dump(config, f)

        edge_index = data.test_data.data['edge_index']
        last_score = np.loadtxt(os.path.join(trainer.log_dir, "last_score_matrix.txt"))
        last_threshold = np.median(last_score[edge_index[0], edge_index[1]])
        last_adj = (last_score > last_threshold) + 0
        last_adj = pd.DataFrame(last_adj, columns=data.node_names, index=data.node_names)
        last_adj.to_csv(os.path.join(trainer.log_dir, "last_adj.csv"))
        return trainer.log_dir

    @classmethod
    def run_all_fold(cls, **config):
        dirs = []
        for split_id in range(1, config['n_splits']+1):
            config['split_id'] = split_id
            print(f"begin split {split_id}/{config['n_splits']}")
            log_dir = cls.eval_one_fold(**config)
            dirs.append(log_dir)
        ans = cls.collect_result(dirs)
        save_dir = os.path.join(os.path.dirname(cls.get_root_dir(config)), "configs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        time_stamp = "_".join(time.asctime().split()).replace(":", "-")
        file = os.path.join(save_dir, f"seed_{config['round']}_{time_stamp}")
        ans['last_adj'].to_csv(f'{file}_last_adj.csv')
        ans['configs'].to_csv(f'{file}_config.csv', index=False)
        return ans

    @classmethod
    def collect_result(cls, dirs):
        last_adj = []
        configs = []
        for dir in dirs:
            last_adj.append(pd.read_csv(os.path.join(dir, "last_adj.csv"), index_col=0))
            metric = cls.load_metric_file(os.path.join(dir, 'metrics.csv'))
            metric['last_adj_connection'] = (last_adj[-1].values-np.diag(np.diag(last_adj[-1].values))).sum()
            with open(os.path.join(dir, "config.json")) as f:
                config = json.load(f)
            config.update(metric)
            configs.append(config)
        last_adj = sum(last_adj)
        last_adj_matrix = (last_adj>=(0.6*len(dirs))).astype(int)
        configs = pd.DataFrame(configs)
        return {'last_adj': last_adj_matrix,
                'configs': configs}

    @classmethod
    def load_metric_file(cls, file):
        metric = pd.read_csv(file)
        ans = {}
        for col in metric.columns:
            ans[col] = metric[col].dropna().iloc[-1]
        return ans


if __name__=="__main__":
    # adj_file = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_Adjacency_matrix/CD14_Mono.txt"
    # node_file = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_Adjacency_matrix/CD14_Mono.TF.list.txt"
    # scrna_feature_file = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_CD14_Mono/10_CD14_Mono_genes_genie3_score.txt"
    # atac_feature_file = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_CD14_Mono/10_CD14_Mono_TF_rp_score10000.txt"
    # data_dir = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_CD14_Mono/train_info/data_k_10_r_1"
    # scrna_file = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/ginie3_input/ginie3_input CD14_Mono.csv"
    #
    # parser = Trainer.add_argparse_args(name="CD14_Mono", adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
    #                                    atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file)
    # config = parser.parse_args()
    # config.n_splits = 10
    # config.comment = "test"
    # model_seeds = list(range(20))
    # Trainer.run_with_seeds(model_seeds=model_seeds, **vars(config))


    target = 'CD14_Mono'
    target = 'Stromalr'

    model_cls = 'GraFRankAE'
    model_cls = 'DeepTFni'

    # root_dir = "/root/autodl-tmp/DeepTFniData"
    # root_dir = "/Users/lcc/PycharmProjects/DeepTFni/example_data"
    root_dir = "/home/jm/PycharmProjects/yss/lcc/pycharm_project_821"

    adj_file = os.path.join(root_dir, f"Intersection_Adjacency_matrix/{target}.txt")
    node_file = os.path.join(root_dir, f"Intersection_Adjacency_matrix/{target}.TF.list.txt")
    scrna_feature_file = os.path.join(root_dir, f"Intersection_{target}/10_{target}_genes_genie3_score.txt")
    atac_feature_file = os.path.join(root_dir, f"Intersection_{target}/10_{target}_TF_rp_score4000.txt")
    data_dir = os.path.join(root_dir, f"Intersection_{target}/train_info/data_k_10_r_1")
    scrna_file = os.path.join(root_dir, f"ginie3_input/ginie3_input {target}.csv")

    parser = Trainer.add_argparse_args(name=target, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                       atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file, model_cls=model_cls)
    config = parser.parse_args()
    # Trainer.run_single_fold(**vars(config))


    data_dir = f"/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_{target}/train_info/data_k_10_r_1"
    checkpoint_dir = f"/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_{target}/trainres/result_k_10_r_1"
    checkpoint_dir = f"/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_{target}/New Folder/result_k_10_r_1"
    # model_cls = 'GraFRankAE'
    parser = Trainer.add_argparse_args(name=target, adj_file=adj_file, node_file=node_file, scrna_file=scrna_file,
                                       atac_feature_file=atac_feature_file, scrna_feature_file=scrna_feature_file,
                                       model_cls=model_cls, split_id=1, data_dir=data_dir, checkpoint_dir=checkpoint_dir)
    config = parser.parse_args([])
    config.model_cls = "GraFRankAE"
    OldEvaluator.eval_one_fold(**vars(config))
    "trainres_varients_only_atac"
    checkpoint_dir = f"/home/jm/PycharmProjects/yss/lcc/pycharm_project_821/Intersection_{target}/trainres_varients_only_atac/result_k_10_r_1/checkpoint_k_1.pt"
    checkpoint = torch.load(checkpoint_dir)