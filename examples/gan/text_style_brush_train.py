import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
from torchvision.transforms import ToTensor
import cv2
import numpy as np

import albumentations as alb

from fline.pipelines.base_pipeline import BasePipeline

from fline.utils.logging.trains import TrainsLogger
from fline.utils.saver import Saver, MetricStrategies
from fline.utils.data.dataset import BaseDataset

from fline.utils.wrappers import ModelWrapper, DataWrapper
from fline.utils.logging.base import EnumLogDataTypes

from fline.utils.logging.groups import ProcessImage

from fline.models.models.research.textstylebrush import TextStyleBrush

from fline.utils.data.image.io import imread_padding
from fline.losses.segmentation.dice import BCEDiceLoss


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={
        'textstylebrush': ModelWrapper(
            model=TextStyleBrush(
                original_encoder_backbone_name='efficientnet_b0',
                gen_encoder_backbone_name='efficientnet_b0',
            ),
            keys=[
                ([IMAGE, GEN1, GEN2], ['res_gen1', 'res_gen2']),
            ],
            optimizer=torch.optim.Adam,
            optimizer_kwargs={
                'lr': 1e-4,
            },
        ),
        'discriminator': ModelWrapper(
            model=TextStyleBrush(
                original_encoder_backbone_name='efficientnet_b0',
                gen_encoder_backbone_name='efficientnet_b0',
            ),
            keys=[
                ([IMAGE, 'true'], ['image_discrloss']),
                ([GEN1, 'false'], ['gen1_discrloss']),
                ([GEN2, 'false'], ['gen2_discrloss']),
            ],
            optimizer=torch.optim.Adam,
            optimizer_kwargs={
                'lr': 1e-4,
            },
        )
    },
    losses=[
        DataWrapper(
            torch.nn.BCELoss(),
            keys=[
                (['conns', CONNECTIONS], ['loss'])
            ],
        ),
    ],
    metrics=[
    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=0,
        log_each_epoch=True,
        log_keys={
            CONNECTIONS: ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=gen_log_connected(CONNECTIONS),
            ),
            'conns': ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=gen_log_connected('conns'),
            ),
            'loss': EnumLogDataTypes.loss,
            IMAGE1: EnumLogDataTypes.image,
            IMAGE2: EnumLogDataTypes.image,
        },
        project_name=PROECT_NAME,
        task_name=TASK_NAME,
    ),
    saver=Saver(
        save_dir='/data/checkpoints',
        project_name=PROECT_NAME,
        experiment_name=TASK_NAME,
        metric_name='loss',
        metric_strategy=MetricStrategies.low,
    ),
    device=DEVICE,
    n_epochs=1000,
    loss_reduce=False,
    #continue_train=True,
)
print('pipeline run')
pipeline.run()
