import sys

import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
from torchvision.transforms import ToTensor
import cv2
import numpy as np
import json

import segmentation_models_pytorch as smp
import albumentations as alb
import timm
from torch.utils.data._utils.collate import default_collate

from fline.pipelines.base_pipeline import BasePipeline

from fline.utils.logging.trains import TrainsLogger
from fline.utils.saver import Saver, MetricStrategies
from fline.utils.data.dataset import BaseDataset
from fline.utils.data.callbacks.base import generate_imread_callback

from fline.utils.wrappers import ModelWrapper, DataWrapper
from fline.utils.logging.base import EnumLogDataTypes

from fline.utils.logging.groups import ConcatImage, ProcessImage

from fline.utils.data.image.io import imread_padding, imread_resize, imread, padding_q, padding_qxy

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.models.segmentation.unet import TimmUnet

from sklearn.model_selection import train_test_split

from fline.losses.segmentation.dice import BCEDiceLoss
from fline.metrics.segmentation.segmentation import iou
from fline.utils.data.dict import OcrDict


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


#SHAPE = (256, 256)
PROECT_NAME = '[RESEARCH]OCRSegmentation'
TASK_NAME = 'test_v1'
SEG_MODEL = 'seg_model'
MASK = 'mask'
IMAGE = 'IMAGE'
DEVICE = 'cuda:1'

CHARS_SEG_PATH = '/data/SARA/data/generated_images/chars_segmentation'

df = pd.read_csv(os.path.join(CHARS_SEG_PATH, 'data.csv'))
ocr_dict = OcrDict(df['text'].values)
train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

train_augs_geometric = alb.Compose([
    #alb.RandomCrop(256, 256),
    alb.ShiftScaleRotate(border_mode=0),
    #alb.Flip(),
    #alb.ElasticTransform(border_mode=0),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(),
    alb.RandomGamma(),
    alb.RandomShadow(),
    #alb.HueSaturationValue(),
    #alb.Blur(),
])


def generate_get_images(mode='train'):
    def get_images(row, res_dict):
        im = cv2.imread(row['image'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row['mask_path'], cv2.IMREAD_GRAYSCALE)
        cntrs, _ = cv2.findContours(mask, 0, 1)
        sort_cntrs = sorted([(cv2.boundingRect(cnt), cnt) for cnt in cntrs])
        sort_cntrs = [cnt[1] for cnt in sort_cntrs]
        sort_chars = ocr_dict.text_to_code(row['text'])
        res_mask = np.zeros((mask.shape[0], mask.shape[1], ocr_dict.count_letters))
        for char, cnt in zip(sort_chars, sort_cntrs):
            buf = res_mask[:, :, char].copy()
            buf = cv2.drawContours(buf, [cnt], -1, 1, -1)
            res_mask[:, :, char] = buf
        res_mask = res_mask * np.expand_dims(mask, -1)
        res_mask = res_mask
        im = padding_q(im, 64, 0)
        res_mask = padding_q(res_mask, 64, 0)
        #print(crop.shape, crop_mask.shape)
        #print(crop.shape, crop_mask.shape)
        res_dict[IMAGE] = im.astype('float32')
        res_dict[MASK] = res_mask.astype('float32')
        return res_dict
    return get_images


def collate_fn(batch):
    max_size_y = max([b[IMAGE].shape[0] for b in batch])
    max_size_x = max([b[IMAGE].shape[1] for b in batch])
    for i, b in enumerate(batch):
        batch[i][IMAGE] = padding_qxy(batch[i][IMAGE], max_size_x, max_size_y, 0)
        batch[i][IMAGE] = ToTensor()(batch[i][IMAGE])
        batch[i][MASK] = padding_qxy(batch[i][MASK], max_size_x, max_size_y, 0)
        batch[i][MASK] = ToTensor()(batch[i][MASK])
        #print(batch[i][IMAGE].shape, batch[i][MASK].shape)
    res_batch = default_collate(batch)
    #[print(k, res_batch[k].shape) for k in res_batch.keys()]
    return res_batch


train_dataset = BaseDataset(
    df=train_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_images('train'),
    ],
)
val_dataset = BaseDataset(
    df=val_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_images('val'),
    ],
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    num_workers=8,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    num_workers=8,
    shuffle=False,
    collate_fn=collate_fn,
)


def concat_images(x):
    return torch.cat([
        x[IMAGE],
        (x[MASK]>0.5).to(dtype=torch.long).argmax(dim=1).unsqueeze(dim=1).repeat(1,3,1,1),
        (x['out']>0.5).to(dtype=torch.long).argmax(dim=1).unsqueeze(dim=1).repeat(1,3,1,1),

    ], dim=2)


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=smp.FPN(encoder_name='efficientnet-b0', classes=ocr_dict.count_letters, activation='sigmoid'),
        # model=TimmFPN(
        #     'efficientnet_b0',
        #     classes=1,
        #     activation=torch.nn.Sigmoid(),
        #     return_dict=False,
        #     features_block='each',
        # ),
        keys=[
            ([IMAGE], ['out']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        DataWrapper(
            BCEDiceLoss(activation=None),
            keys=[
                (['out', MASK], ['loss'])
            ],
            # processing={
            #     MASK: lambda x: x[MASK].to(dtype=torch.long)
            # }
        ),
    ],
    metrics=[
        DataWrapper(
            iou,
            keys=[
                (['out', MASK], ['iou'])
            ],
        ),
    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=0,
        log_each_epoch=True,
        log_keys={
            'loss': EnumLogDataTypes.loss,
            'out': ProcessImage(
                EnumLogDataTypes.image,
                process_func=concat_images
            ),
            #'iou': EnumLogDataTypes.metric,
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
)
print('pipeline run')
pipeline.run()
