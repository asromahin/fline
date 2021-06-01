from collections import defaultdict
import numpy as np
import typing as tp


from fline.utils.saver import Modes
from fline.utils.logging.base_group import BaseGroup


class EnumLogDataTypes:
    image = 'log_image'
    metric = 'log_metric'
    loss = 'log_loss'


class BaseLogger:
    def __init__(
            self,
            config: dict,
            log_each_epoch: bool = True,
            log_each_ind: int = 0,
            log_keys: tp.Mapping[str, tp.Union[EnumLogDataTypes, BaseGroup]] = None,
    ):
        self.config = config
        self._cur_ind = 0
        self._last_epoch = None
        self._last_ind = None
        self.log_each_epoch = log_each_epoch
        self.log_each_ind = log_each_ind

        self.log_keys = log_keys

        self._acumulate_losses = defaultdict(lambda:  defaultdict(list))
        self._acumulate_metrics = defaultdict(lambda:  defaultdict(list))
        self._accumulate_images = defaultdict(lambda:  defaultdict(list))

    def push_metric(self, metrics, metric_name, batch_ind, ind, mode):
        return NotImplementedError()

    def push_loss(self, losses, loss_name, batch_ind, ind, mode):
        return NotImplementedError()

    def push_image(self, image, image_name, batch_ind,  ind, mode):
        return NotImplementedError()

    def push(self, data, ind, epoch, mode: Modes = Modes.train):
        for log_key, log_type in self.log_keys.items():
            if isinstance(log_type, BaseGroup):
                batch_d = log_type(data)
                log_type = log_type.log_type
            else:
                batch_d = data.get(log_key)
            if batch_d is not None:
                if log_type == EnumLogDataTypes.loss:
                    d = batch_d.mean().detach().cpu().numpy()
                    if self.log_each_ind != 0 and ind % self.log_each_ind == 0:
                        self.push_loss(d, log_key+'_ind', 0, ind, mode=mode)
                    if self.log_each_epoch:
                        self._acumulate_losses[log_key][mode].append(d)
                if log_type == EnumLogDataTypes.metric:
                    d = batch_d.mean().detach().cpu().numpy()
                    if self.log_each_ind != 0 and ind % self.log_each_ind == 0:
                        self.push_metric(d, log_key+'_ind', 0, ind, mode=mode)
                    if self.log_each_epoch:
                        self._acumulate_metrics[log_key][mode].append(d)
                if log_type == EnumLogDataTypes.image:
                    for batch_ind, d in enumerate(batch_d):
                        self.push_image(d, log_key+'_ind', batch_ind, self._cur_ind, mode=mode)

        if self._last_epoch is not None:
            if epoch > self._last_epoch and self.log_each_epoch:
                self._log_accumulate()
        self._cur_ind += 1
        self._last_ind = ind
        self._last_epoch = epoch

    def _acumulate(self, losses, metrics, images):
        for loss_name, loss in losses.items():
            for mode, mode_data in loss.items():
                self._acumulate_losses[loss_name][mode].append(mode_data)
        for metric_name, metric in metrics.items():
            for mode, mode_data in metric.items():
                self._acumulate_metrics[metric_name][mode].append(mode_data)

    def _log_accumulate(self):
        for loss_name, loss_dict in self._acumulate_losses.items():
            for mode, loss in loss_dict.items():
                self.push_metric(np.mean(loss), loss_name, 0, self._last_epoch,
                                 mode=mode)
        for metric_name, metric_dict in self._acumulate_metrics.items():
            for mode, metric in metric_dict.items():
                self.push_metric(np.mean(metric), metric_name, 0, self._last_epoch,
                                 mode=mode)
        self._acumulate_losses.clear()
        self._acumulate_metrics.clear()



"""    def push(self, metrics, losses, images, ind, epoch, strategy_name=''):
        if self.log_each_ind != 0 and ind % self.log_each_ind == 0:
            self.push_metrics(metrics, self._cur_ind, strategy_name=strategy_name+'_ind')
            self.push_losses(losses, self._cur_ind, strategy_name=strategy_name+'_ind')
            self.push_images(images, self._cur_ind, strategy_name=strategy_name+'_ind')
        if self._last_epoch is not None:
            if epoch > self._last_epoch and self.log_each_epoch:
                reduce_losses, reduce_metrics = self._get_accumulate()
                self.push_metrics(reduce_metrics, self._last_epoch, strategy_name=strategy_name)
                self.push_losses(reduce_losses, self._last_epoch, strategy_name=strategy_name)
                self.push_images(images, self._last_epoch, strategy_name=strategy_name)
                self._acumulate_losses.clear()
                self._acumulate_metrics.clear()
                self._accumulate_images.clear()
        if self.log_each_epoch:
            self._acumulate(losses=losses, metrics=metrics, images=images)
        self._cur_ind += 1
        self._last_ind = ind
        self._last_epoch = epoch"""