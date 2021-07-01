DATA_DIR = '/data/KAGGLE/GlobalWheatDetection/GWDData/'

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


from fline.pipelines.base_pipeline import BasePipeline

from fline.utils.logging.trains import TrainsLogger
from fline.utils.saver import Saver, MetricStrategies
from fline.utils.data.dataset import BaseDataset
from fline.utils.data.callbacks.base import generate_imread_callback

from fline.utils.wrappers import ModelWrapper, DataWrapper
from fline.utils.logging.base import EnumLogDataTypes

from fline.utils.logging.groups import ConcatImage, ProcessImage

from fline.utils.data.image.io import imread_padding, imread_resize, imread, padding_q

from fline.models.models.segmentation.fpn import TimmFPN
from fline.models.models.segmentation.unet import TimmUnet

from sklearn.model_selection import train_test_split

from fline.losses.segmentation.dice import BCEDiceLoss
from fline.metrics.segmentation.segmentation import iou


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


#SHAPE = (256, 256)
PROECT_NAME = '[KAGGLE]GWD'
TASK_NAME = 'segmentation_each_all_pspnet_densenet121_V2'
SEG_MODEL = 'seg_model'
MASK = 'mask'
IMAGE = 'IMAGE'
DEVICE = 'cuda:3'

df = pd.read_csv(os.path.join(DATA_DIR, 'train.csv'))
df[IMAGE] = DATA_DIR + df['filename']

#groups = df.groupby([IMAGE])
#groups_names = pd.DataFrame(list(df[IMAGE].unique()), columns=[IMAGE])
train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

train_augs_geometric = alb.Compose([
    #alb.RandomCrop(256, 256),
    alb.ShiftScaleRotate(border_mode=0),
    alb.Flip(),
    #alb.ElasticTransform(border_mode=0),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(),
    alb.RandomGamma(),
    alb.RandomShadow(),
    #alb.HueSaturationValue(),
    #alb.Blur(),
])


def read_cnt(row):
    rsa = json.loads(row['region_shape_attributes'])
    all_x = rsa['all_points_x']
    all_y = rsa['all_points_y']
    cnt = np.zeros((len(all_x), 1, 2), dtype='int')
    cnt[:, 0, 0] = all_x
    cnt[:, 0, 1] = all_y
    return cnt


def generate_get_images(mode='train'):
    def get_images(trow, res_dict):
        im = cv2.imread(trow[IMAGE])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cnt = read_cnt(trow)
        x, y, w, h = cv2.boundingRect(cnt)
        if np.random.randint(0, 2) == 0 and mode == 'train':
            #print(x, y, w, h)
            w_shift = (0.2 * np.random.rand()) * w
            h_shift = (0.2 * np.random.rand()) * h
            x_shift = 0.5 * np.random.rand() * w_shift
            y_shift = 0.5 * np.random.rand() * h_shift
            x -= x_shift
            y -= y_shift
            w += w_shift
            h += h_shift
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            #print(x,y,w,h)

        mask = np.zeros((im.shape[0], im.shape[1], 1))
        mask = cv2.drawContours(mask, [cnt], -1, 1, -1)
        crop = (im[max(y, 0):min(y+h, im.shape[0]), max(x, 0):min(x+w, im.shape[1])]/255).astype('float32')
        crop_mask = mask[max(y, 0):min(y+h, im.shape[0]), max(x, 0):min(x+w, im.shape[1])].astype('float32')
        if mode == 'train':
            a = train_augs_geometric(image=crop, mask=crop_mask)
            crop = a['image']
            crop_mask = a['mask']
            a = train_augs_values(image=crop, mask=crop_mask)
            crop = a['image']
            crop_mask = a['mask']
        crop = padding_q(crop, 64, 0)
        crop_mask = padding_q(crop_mask, 64, 0)
        #print(crop.shape, crop_mask.shape)
        crop = ToTensor()(crop)
        crop_mask = ToTensor()(crop_mask)
        #print(crop.shape, crop_mask.shape)
        #print(crop.shape, crop_mask.shape)
        res_dict[IMAGE] = crop
        res_dict[MASK] = crop_mask
        return res_dict
    return get_images


def generate_get_images_v2(mode='train'):
    def get_images(trow, res_dict):
        im = cv2.imread(trow[IMAGE])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        group = groups.get_group(trow[IMAGE])
        cnt = read_cnt(group.sample(1).iloc[0])
        x, y, w, h = cv2.boundingRect(cnt)
        if np.random.randint(0, 2) == 0 and mode == 'train':
            #print(x, y, w, h)
            w_shift = (1.0 * np.random.rand()) * w
            h_shift = (1.0 * np.random.rand()) * h
            x_shift = 0.5 * np.random.rand() * w_shift
            y_shift = 0.5 * np.random.rand() * h_shift
            x -= x_shift
            y -= y_shift
            w += w_shift
            h += h_shift
            x = int(x)
            y = int(y)
            w = int(w)
            h = int(h)
            #print(x,y,w,h)

        mask = np.zeros((im.shape[0], im.shape[1], 1))
        #mask = cv2.drawContours(mask, [cnt], -1, 1, -1)
        cntrs = []
        for i in range(len(group)):
            crow = group.iloc[i]
            ccnt = read_cnt(crow)
            cntrs.append(ccnt)
        mask = cv2.drawContours(mask, cntrs, -1, 1, -1)
        crop = (im[max(y, 0):min(y+h, im.shape[0]), max(x, 0):min(x+w, im.shape[1])]/255).astype('float32')
        crop_mask = mask[max(y, 0):min(y+h, im.shape[0]), max(x, 0):min(x+w, im.shape[1])].astype('float32')
        if mode == 'train':
            a = train_augs_geometric(image=crop, mask=crop_mask)
            crop = a['image']
            crop_mask = a['mask']
            a = train_augs_values(image=crop, mask=crop_mask)
            crop = a['image']
            crop_mask = a['mask']
        crop = padding_q(crop, 64, 0)
        crop_mask = padding_q(crop_mask, 64, 0)
        #print(crop.shape, crop_mask.shape)
        crop = ToTensor()(crop)
        crop_mask = ToTensor()(crop_mask)
        #print(crop.shape, crop_mask.shape)
        #print(crop.shape, crop_mask.shape)
        res_dict[IMAGE] = crop
        res_dict[MASK] = crop_mask
        return res_dict
    return get_images


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
    batch_size=1,
    num_workers=1,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=1,
    shuffle=False,
)


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        #model=smp.PSPNet(encoder_name='densenet121', classes=1, activation='sigmoid'),
        model=TimmFPN(
            'efficientnet_b0',
            classes=1,
            activation=torch.nn.Sigmoid(),
            return_dict=False,
            features_block='each',
        ),
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
            IMAGE: EnumLogDataTypes.image,
            MASK: EnumLogDataTypes.image,
            'out': EnumLogDataTypes.image,
            'iou': EnumLogDataTypes.metric,
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
