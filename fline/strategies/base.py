import torch
import typing as tp


def base_strategy(data, models, losses, metrics, is_train=True, loss_reduce=False):
    for model_name, model in models.items():
        if is_train:
            model.optimizer.zero_grad()
            data = model(data)
        else:
            with torch.no_grad():
                data = model(data)

    if loss_reduce and is_train:
        summary_loss = None
    for loss in losses:
        data = loss(data)
        for in_key, out_keys in loss.keys:
            for key in out_keys:
                if is_train:
                    if loss_reduce:
                        if summary_loss is None:
                            summary_loss = data[key].mean()
                        else:
                            summary_loss += data[key].mean()
                    else:
                        data[key].mean().backward(retain_graph=True)
    if loss_reduce and is_train:
        summary_loss.backward()

    for metric in metrics:
        data = metric(data)
    if is_train:
        for model_name, model in models.items():
            model.optimizer.step()
    return data


def filter_strategy(models_keys=None, losses_idx=None, metrics_idx=None):
    def _filter_strategy(data, models, losses, metrics, device, is_train=True, loss_reduce=False):
        if models_keys is not None:
            models = {model_name: model for model_name, model in models.items() if model_name in models_keys}
        if losses_idx is not None:
            losses = [loss for loss_idx, loss in enumerate(losses) if loss_idx in losses_idx]
        if metrics_idx is not None:
            metrics = [metric for metric_idx, metric in enumerate(metrics) if metric_idx in metrics_idx]
        return base_strategy(data, models, losses, metrics, is_train=is_train, loss_reduce=loss_reduce)
    return _filter_strategy


class BaseStrategy:
    def __init__(
            self,
    ):
        self.models = None
        self.losses = None
        self.metrics = None
        self.loss_reduce = None

    def init_args(
            self,
            models,
            losses,
            metrics,
            loss_reduce,
    ):
        self.models = models
        self.losses = losses
        self.metrics = metrics
        self.loss_reduce = loss_reduce

    def forward(self, data, is_train=True):
        return base_strategy(
            data,
            is_train=is_train,
            models=self.models,
            losses=self.losses,
            metrics=self.metrics,
            loss_reduce=self.loss_reduce,
        )


class StageStrategy(BaseStrategy):
    def __init__(
            self,
            stages: tp.List[tp.Mapping[str, tp.Tuple[tp.List, tp.List]]],
    ):
        super(StageStrategy, self).__init__()
        self.stages = stages

    def forward(self, data, is_train=True):
        data = None
        for stage in self.stages:
            cdata = base_strategy(
                data,
                is_train=is_train,
                models=self.models,
                losses=self.losses,
                metrics=self.metrics,
                loss_reduce=self.loss_reduce,
            )
            if data is None:
                data = cdata
            else:
                data.update(cdata)
        return data
