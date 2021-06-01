import typing as tp
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import numpy as np

from fline.utils.wrappers import DataWrapper, StrategyWrapper, ModelWrapper
from fline.utils.logging.base import BaseLogger
from fline.utils.saver import Saver, Modes
from fline.strategies.base import BaseStrategy


class BasePipeline:
    def __init__(
            self,
            train_loader: DataLoader,
            val_loader: DataLoader = None,
            test_loader: DataLoader = None,

            models: tp.Mapping[str, ModelWrapper] = None,
            strategy: BaseStrategy = BaseStrategy,
            losses: tp.List[DataWrapper] = None,
            metrics: tp.List[DataWrapper] = None,
            loss_reduce: bool = True,

            continue_train: bool = False,
            device: str = 'cpu',
            n_epochs: int = 1,

            logger: BaseLogger = None,
            saver: Saver = None,
    ):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.models = models
        self.losses = losses
        self.metrics = metrics
        self.loss_reduce = loss_reduce

        self.continue_train = continue_train
        self.device = device

        self.saver = saver
        self.logger = logger

        self.n_epochs = n_epochs

        self.strategy = strategy
        self.strategy.init_args(
            models=self.models,
            metrics=self.metrics,
            losses=self.losses,
            loss_reduce=self.loss_reduce,
        )

    def _prepare_models(self):
        if self.continue_train:
            self.models = self.saver.load_models(self.models, device=self.device)
        for model_name, model_wrapper in self.models.items():
            model_wrapper.model.to(self.device)
            model_wrapper.model.train()

    def _train_epoch(self, epoch):
        pbar = tqdm(self.train_loader)
        for i, data in enumerate(pbar, 0):
            for strategy in self.strategies:
                for data_name, data_val in data.items():
                    if isinstance(data_val, torch.Tensor):
                        data[data_name] = data[data_name].to(self.device)
                data = strategy(
                    data=data,
                    models=self.models,
                    losses=self.losses,
                    metrics=self.metrics,
                    device=self.device,
                    is_train=True,
                    loss_reduce=self.loss_reduce,
                )
                pbar.set_postfix({'loss': data['loss'].item()})
                if self.logger is not None:
                    self.logger.push(
                        data=data,
                        ind=i,
                        epoch=epoch,
                        mode=Modes.train,
                    )
                self.saver.save_best(
                    models=self.models,
                    train_data=data,
                    val_data=None,
                )

    def _val_epoch(self, epoch):
        if self.val_loader is not None:
            pbar = tqdm(self.val_loader)
            for i, data in enumerate(pbar, 0):
                for strategy in self.strategies:
                    for data_name, data_val in data.items():
                        if isinstance(data_val, torch.Tensor):
                            data[data_name] = data[data_name].to(self.device)
                    data = strategy(
                        data=data,
                        models=self.models,
                        losses=self.losses,
                        metrics=self.metrics,
                        device=self.device,
                        is_train=False,
                        loss_reduce=False,
                    )
                    pbar.set_postfix({'loss': data['loss'].item()})
                if self.logger is not None:
                    self.logger.push(
                        data=data,
                        ind=i,
                        epoch=epoch,
                        mode=Modes.val,
                    )
                self.saver.save_best(
                    models=self.models,
                    train_data=None,
                    val_data=data,
                )

    def run(self):
        self._prepare_models()
        for epoch in range(self.n_epochs):
            self._train_epoch(epoch)
            self._val_epoch(epoch)


