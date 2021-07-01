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

from fline.utils.data.image.io import imread_padding, imread_resize, imread, padding_q


from sklearn.model_selection import train_test_split


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[KAGGLE]GWD'
TASK_NAME = 'box_verify_test_v6'
SEG_MODEL = 'seg_model'
MASK = 'mask'
DEVICE = 'cuda:2'
IMAGE = 'IMAGE'


df = pd.read_csv('/data/KAGGLE/GlobalWheatDetection/2021/crop_df_v2.csv')
df = df.dropna()
df = df[~(df['iou'].str.contains('index') == True)]
df['iou'] = df['iou'].astype('float32')
df['size_left'] = df['x1'] - df['crop_x1']
df['size_right'] = df['crop_x2'] - df['x2']
df['size_down'] = df['y1'] - df['crop_y1']
df['size_up'] = df['crop_y2'] - df['y2']
groups = df[df['iou'] == 0].groupby(['im_path'])
im_paths = list(df[df['iou'] == 0]['im_path'].unique())
# rdf = pd.DataFrame([{'im_path': im} for im in im_paths])

train_augs_geometric = alb.Compose([
    alb.ShiftScaleRotate(),
    #alb.ElasticTransform(),
    alb.Flip(),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(),
    alb.RandomGamma(),
    alb.RandomShadow(),
    alb.Blur(),
])


def collate_fn(batch):
    max_size = max([b[IMAGE].shape[0] for b in batch] + [b[IMAGE].shape[1] for b in batch])
    for i, b in enumerate(batch):
        batch[i][IMAGE] = padding_q(batch[i][IMAGE], max_size, 0)
        #print(batch[i][IMAGE].shape, max_size)
        batch[i][IMAGE] = ToTensor()(batch[i][IMAGE])
    res_batch = default_collate(batch)
    #[print(k, res_batch[k].shape) for k in res_batch.keys()]
    return res_batch


def generate_get_images(mode='train'):
    def get_images(row, res_dict):
        im = cv2.imread(row['im_path'])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        x1, y1, x2, y2 = (row[['x1', 'y1', 'x2', 'y2']]*1024).astype('int')
        label = 1
        if np.random.rand() < 0.5 and row['im_path'] in im_paths:
            rows = groups.get_group(row['im_path'])
            if len(rows) > 0:
                crow = rows.sample(1).iloc[0]
                t = np.random.randint(0, 4)     #   left,right,down,up
                if t == 0:
                    new_x1, new_y1, new_x2, new_y2 = (crow[['crop_x1', 'crop_y1', 'x1', 'crop_y2']]*1024).astype('int')
                if t == 1:
                    new_x1, new_y1, new_x2, new_y2 = (crow[['x2', 'crop_y1', 'crop_x2', 'crop_y2']]*1024).astype('int')
                if t == 2:
                    new_x1, new_y1, new_x2, new_y2 = (crow[['crop_x1', 'crop_y1', 'crop_x2', 'y1']]*1024).astype('int')
                if t == 3:
                    new_x1, new_y1, new_x2, new_y2 = (crow[['crop_x1', 'y2', 'crop_x2', 'crop_y2']]*1024).astype('int')

                dy = new_y2-new_y1
                dx = new_x2-new_x1
                #print(dx,dy)
                if dy > 1 and dx > 1 and dy<512 and dx<512:
                    x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2
                    label = 0
                    #print(label)
        crop = im[y1:y2, x1:x2]
        #print(crop.shape,x1,y1,x2,y2)
        if mode == 'train':
            crop = train_augs_geometric(image=crop)['image']
            crop = train_augs_values(image=crop)['image']
        crop = padding_q(crop, 64, 0)
        #im = cv2.resize(im, SHAPE)
        res_dict[IMAGE] = ((crop / 255).astype('float32'))
        res_dict['label'] = label
        #print(label)
        return res_dict
    return get_images

train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

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
    num_workers=4,
    shuffle=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn,
)


def softmax_acc(pr, gt):
    pr_argmax = pr.argmax(dim=-1, keepdim=False)
    s = (pr_argmax == gt).sum()
    #print(pr_argmax.shape, gt.shape, s, pr.shape[0])
    return s/pr.shape[0]


def add_mark(x):
    label = x['out']
    target = x['label']
    #print(label.shape)
    labels = label.argmax(dim=1)
    for i, (l, t) in enumerate(zip(labels, target)):
        if l == 1:
            x[IMAGE][i, :, :10, :10] = 1
        if t == 1:
            x[IMAGE][i, :, :10, -10:] = 1
    return x[IMAGE]

pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=timm.create_model('efficientnet_b0', num_classes=2),
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
            softmax_acc,
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
            IMAGE: ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=add_mark,
            ),
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