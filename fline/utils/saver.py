import torch
import os
import typing as tp


class MetricStrategies:
    low = 'low'
    high = 'high'


class Modes:
    train = 'train'
    val = 'val'
    test = 'test'


class Saver:
    def __init__(
            self,
            save_dir: str = './romakhin_utils_results',
            project_name: str = 'test_project',
            experiment_name: str = 'test_experiment',
            metric_name: str = None,
            metric_strategy: MetricStrategies = MetricStrategies.high,
            by_mode=Modes.val,
            save_by_ind: int = None,
    ):
        self.save_dir = save_dir
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.experiment_dir = self.get_experiment_dir()

        self.metric_name = metric_name
        self.metric_strategy = metric_strategy

        self.last_metric = None

        self.by_mode = by_mode
        self.save_by_ind = save_by_ind

        self.prepare_experiment_dir()

    def get_experiment_dir(self):
        return os.path.join(
            self.save_dir,
            self.project_name,
            self.experiment_name,
        )

    def get_config_path(self):
        return os.path.join(
            self.experiment_dir,
            'config.json',
        )

    def get_model_path(self, model_name):
        return os.path.join(
            self.experiment_dir,
            f'{model_name}_model.pt',
        )

    def get_weights_path(self, model_name):
        return os.path.join(
            self.experiment_dir,
            f'{model_name}_weights.pth',
        )

    def get_traced_model_path(self, model_name):
        return os.path.join(
            self.experiment_dir,
            f'{model_name}_traced_model.pt',
        )

    def prepare_experiment_dir(self):
        os.makedirs(self.experiment_dir, exist_ok=True)

    def get_folder_size(self):
        return get_folder_size(self.experiment_dir)

    def _get_save_metric(self, train_data: dict, val_data: dict):
        cur_data = None
        if self.by_mode == Modes.val:
            cur_data = val_data
        if self.by_mode == Modes.train:
            cur_data = train_data
        if cur_data is not None:
            if self.metric_name in cur_data.keys():
                return cur_data[self.metric_name].mean().detach().cpu().numpy()

    def save_best(self, models, train_data, val_data):
        cur_metric = self._get_save_metric(train_data, val_data)
        if cur_metric is not None:
            if self.last_metric is None:
                self.save_models(models)
                self.last_metric = cur_metric
            elif cur_metric > self.last_metric and self.metric_strategy == MetricStrategies.high:
                self.save_models(models)
                self.last_metric = cur_metric
            elif cur_metric < self.last_metric and self.metric_strategy == MetricStrategies.low:
                self.save_models(models)
                self.last_metric = cur_metric

    def save_best_by_ind(self, models, train_data, val_data, ind):
        if self.save_by_ind is not None:
            if ind % self.save_by_ind == 0:
                self.save_best(models, train_data, val_data)

    def save_models(self, models):
        for model_name, model in models.items():
            torch.save(model, self.get_model_path(model_name))

    def load_models(self, models, device):
        for model_name, model in models.items():
            models[model_name].model = torch.load(self.get_model_path(model_name), map_location=device).model.to(device)
        return models


def get_folder_size(folder):
    total_size = os.path.getsize(folder)
    for item in os.listdir(folder):
        itempath = os.path.join(folder, item)
        if os.path.isfile(itempath):
            total_size += os.path.getsize(itempath)
        elif os.path.isdir(itempath):
            total_size += get_folder_size(itempath)
    return total_size
