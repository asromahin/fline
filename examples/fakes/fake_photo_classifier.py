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

from fline.utils.data.image.io import imread_padding, imread_resize, imread


from sklearn.model_selection import train_test_split


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[SARA]DocFake'
TASK_NAME = 'fake_photo_classifier_v5'
SEG_MODEL = 'seg_model'
MASK = 'mask'
FAKE_IMAGE = 'fake_image'
DEVICE = 'cuda:1'
IMAGE = 'IMAGE'



def read_annot(annotation_path):
    with open(annotation_path, 'r') as f:
        data = json.loads(f.read())
    regions = data['regions']
    res = None, None, None, None
    for reg in regions:
        shape_attributes = reg['shape_attributes']
        if shape_attributes['name'] == 'rect':
            x = shape_attributes['x']
            y = shape_attributes['y']
            w = shape_attributes['width']
            h = shape_attributes['height']
            return x,y,w,h
    return res


df = pd.read_csv('/data/DOCS_ANTISPOOFING/data/datasets/dataset_v01.csv')
#df = df.iloc[500:700]

df = df[~df['image'].str.contains('|'.join(os.listdir('/data/loans/visual_fraud/')))]
df['bbox_data'] = df['annotation'].apply(read_annot)
df['x'] = df['bbox_data'].str[0]
df['y'] = df['bbox_data'].str[1]
df['w'] = df['bbox_data'].str[2]
df['h'] = df['bbox_data'].str[3]
df['x'] = df['x'].clip(lower=0)
df['y'] = df['y'].clip(lower=0)

df = df.dropna()

train_augs_geometric = alb.Compose([
    alb.ShiftScaleRotate(),
    alb.ElasticTransform(),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(),
    alb.RandomGamma(),
    alb.RandomShadow(),
    alb.Blur(),
])


def generate_get_images(cur_df, mode='train'):
    def get_images(row, res_dict):
        im = cv2.imread(row['image'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #im = cv2.resize(im, 512, 512)
        if np.random.randint(0, 2) == 0:
            another_row = cur_df.iloc[np.random.randint(0, len(cur_df))]
            another_im = cv2.imread(another_row['image'])
            another_im = cv2.cvtColor(another_im, cv2.COLOR_BGR2RGB)
            crop = another_im[
                   another_row['y']:another_row['y'] + another_row['h'],
                   another_row['x']:another_row['x'] + another_row['w'],
                   ]
            #print(crop.shape, another_row[['x','y','w','h']])
            im[row['y']:row['y']+row['h'], row['x']:row['x']+row['w']] = cv2.resize(
                crop,
                (row['w'], row['h']),
            )
            res_dict['label'] = 1
        else:
            res_dict['label'] = 0
        if mode == 'train':
            im = train_augs_geometric(image=im)['image']
            im = train_augs_values(image=im)['image']
        im = cv2.resize(im, SHAPE)
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        return res_dict
    return get_images

train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

train_dataset = BaseDataset(
    df=train_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_images(train_df, 'val'),
    ],
)
val_dataset = BaseDataset(
    df=val_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_images(val_df, 'val'),
    ],
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=8,
    num_workers=4,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=8,
    num_workers=4,
    shuffle=False,
)


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=timm.create_model('densenet121', num_classes=2),
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
            torch.nn.CrossEntropyLoss(),
            keys=[
                (['out', 'label'], ['loss'])
            ],
            processing={
                'out': lambda x: torch.softmax(x['out'], dim=1),
                'label': lambda x: x['label'].to(dtype=torch.long),
            },
        ),
    ],
    metrics=[
        # DataWrapper(
        #     lambda out, true: (out.argmax(dim=-1) == true).sum()/out.shape[0],
        #     keys=[
        #         (['out1', 'true'], ['acc1'])
        #     ],
        # ),
        DataWrapper(
            lambda out, true: (out.argmax(dim=-1, keepdim=True) == true).sum()/out.shape[0],
            #lambda out, true: ((out > 0.5) == true.to(dtype=torch.bool)).sum()/out.shape[0],
            keys=[
                (['out', 'label'], ['acc'])
            ],
            processing={
                'out': lambda x: torch.softmax(x['out'], dim=1),
                'label': lambda x: x['label'].to(dtype=torch.long),
            },
        ),
    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=0,
        log_each_epoch=True,
        log_keys={
            'loss': EnumLogDataTypes.loss,
            IMAGE: EnumLogDataTypes.image,
            'acc': EnumLogDataTypes.metric,
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
