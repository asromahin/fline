import wandb
import typing as tp

from fline.utils.logging.base import BaseLogger, EnumLogDataTypes
from fline.utils.data.image.convert import convert_tensor_image_to_numpy


class WandbLogger(BaseLogger):
    def __init__(
            self,
            config: dict,
            project_name: str,
            task_name: str,
            log_each_epoch: bool = True,
            log_each_ind: int = 0,
            log_keys: tp.Mapping[str, EnumLogDataTypes] = None,

    ):
        super(WandbLogger, self).__init__(
            config=config,
            log_each_epoch=log_each_epoch,
            log_each_ind=log_each_ind,
            log_keys=log_keys,
        )

        self.task = wandb.init(
            config=config,
            project=project_name,
            name=task_name,
        )

    def push_metric(self, metric, metric_name, batch_ind, ind, mode):
        wandb.log({
            '_'.join([mode, metric_name]): metric,
        })

    def push_loss(self, loss, loss_name, batch_ind, ind, mode):
        wandb.log({
            '_'.join([mode, loss_name]): loss,
        })

    def push_image(self, image, image_name, batch_ind, ind, mode):
        wandb.log({
            '_'.join([mode, image_name]): wandb.Image(image),
        })
