from trains import Task, Logger
import typing as tp

from fline.utils.logging.base import BaseLogger, EnumLogDataTypes
from fline.utils.data.image.convert import convert_tensor_image_to_numpy


class TrainsLogger(BaseLogger):
    def __init__(
            self,
            config: dict,
            project_name: str,
            task_name: str,
            log_each_epoch: bool = True,
            log_each_ind: int = 0,
            log_keys: tp.Mapping[str, EnumLogDataTypes] = None,

    ):
        super(TrainsLogger, self).__init__(
            config=config,
            log_each_epoch=log_each_epoch,
            log_each_ind=log_each_ind,
            log_keys=log_keys,
        )

        self.task = Task.init(
            project_name=project_name,
            task_name=task_name,
        )
        self.task.connect(config)
        self.logger = self.task.get_logger()

    def push_metric(self, metric, metric_name, batch_ind, ind, mode):
        self.logger.report_scalar(
            title='_'.join([metric_name]),
            series='_'.join([mode, metric_name]),
            value=metric,
            iteration=ind,
        )

    def push_loss(self, loss, loss_name, batch_ind, ind, mode):
        self.logger.report_scalar(
            title='_'.join([loss_name]),
            series='_'.join([mode, loss_name]),
            value=loss,
            iteration=ind,
        )

    def push_image(self, image, image_name, batch_ind, ind, mode):
        image = convert_tensor_image_to_numpy(image)
        if image.shape[-1] == 1:
            image = image[:, :, 0]
        self.logger.report_image(
            title='_'.join([mode]),
            series=image_name+'_'+str(batch_ind),
            iteration=ind,
            image=image,
        )
