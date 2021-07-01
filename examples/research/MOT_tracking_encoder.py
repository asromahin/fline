DATASET_DIR = '/home/ds/Projects/SARA/datasets/fields_segmentation'
SRC_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/document_crops_rotated/'
DST_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_640_v1/'
SRC_MASKS_PREFIX = '/home/ds/Projects/SARA/annotations/fields_segmentation/'
DST_MASKS_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_640_v1_masks/'

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

from fline.models.models.research.connection_model import (
    TrackConnectionsSMP,
    TrackConnectionsSMPV2,
    TrackConnectionsTimm,
    TrackConnectionsSMPV3,
    TrackConnectionsTimmV2,
    TrackEncoderModel,
)

from fline.utils.data.image.io import imread_padding
from fline.losses.segmentation.dice import BCEDiceLoss

os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[RESEARCH]ConnectionsModels'
TASK_NAME = 'MOT-20_test_encoder_features'
SEG_MODEL = 'main'
MASK = 'mask'
BBOX1 = 'bbox1'
BBOX2 = 'bbox2'
IMAGE1 = 'IMAGE1'
IMAGE2 = 'IMAGE2'
CONNECTIONS = 'connections'

DEVICE = 'cuda:2'


df = pd.read_csv('/data/research/mot/MOT20/train_df.csv')
video_list = list(df['video'].unique())
groups = df.groupby(['video', 'frame_id'])
frame_list = list(groups.size().keys())

train_video = video_list[:-1]
val_video = video_list[-1:]

keys = ['video', 'frame_id']
train_frames = [{keys[i]: v for i, v in enumerate(frame)} for frame in frame_list if frame[0] in train_video]
val_frames = [{keys[i]: v for i, v in enumerate(frame)} for frame in frame_list if frame[0] in val_video]


train_df = pd.DataFrame(train_frames)
val_df = pd.DataFrame(val_frames)

train_augs_geometric = alb.Compose([
    # alb.ShiftScaleRotate(),
    # alb.ElasticTransform(),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(),
    alb.RandomGamma(),
    alb.RandomShadow(),
    alb.Blur(),
])

def get_crop(im1,im2,cur_df,another_df):
    x = np.random.randint(im1.shape[1] - SHAPE[0])
    y = np.random.randint(im1.shape[0] - SHAPE[1])
    x2 = x + SHAPE[0]
    y2 = y + SHAPE[1]
    mask_x_cur = (cur_df['x'] > x) & (cur_df['x'] + cur_df['w'] < x2)
    mask_y_cur = (cur_df['y'] > y) & (cur_df['y'] + cur_df['h'] < y2)
    mask_x_another = (another_df['x'] > x) & (another_df['x'] + another_df['w'] < x2)
    mask_y_another = (another_df['y'] > y) & (another_df['y'] + another_df['h'] < y2)
    cur_df = cur_df[mask_x_cur & mask_y_cur]
    another_df = another_df[mask_x_another & mask_y_another]
    cur_df = cur_df.copy()
    another_df = another_df.copy()
    cur_df['x'] = cur_df['x'] - x
    cur_df['y'] = cur_df['y'] - y
    another_df['x'] = another_df['x'] - x
    another_df['y'] = another_df['y'] - y
    im1 = im1[y:y2, x:x2]
    im2 = im2[y:y2, x:x2]
    return im1, im2, cur_df, another_df


def generate_get_bboxes(mode='train'):
    def get_bboxes(row, res_dict):
        cur_df = groups.get_group((row['video'], row['frame_id']))
        #im1 = imread_padding(cur_df.iloc[0]['frame_path'], 64, 0)
        im1 = cv2.imread(cur_df.iloc[0]['frame_path'])
        im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
        #im1 = cv2.resize(im1, SHAPE)
        new_id = str(int(row['frame_id']) + 1)
        if new_id not in list(df[df['video'] == row['video']]['frame_id'].values):
            new_id = row['frame_id']
        another_df = groups.get_group((row['video'], new_id))
        #im2 = imread_padding(another_df.iloc[0]['frame_path'], 64, 0)
        im2 = cv2.imread(another_df.iloc[0]['frame_path'])
        im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
        #im2 = cv2.resize(im2, SHAPE)
        #print(len(cur_df), len(another_df))
        crop_im1, crop_im2, crop_cur_df, crop_another_df = get_crop(im1, im2, cur_df, another_df)
        while len(crop_cur_df) == 0 or len(crop_another_df) == 0:
            crop_im1, crop_im2, crop_cur_df, crop_another_df = get_crop(im1, im2, cur_df, another_df)
        im1 = crop_im1
        im2 = crop_im2
        cur_df = crop_cur_df
        another_df = crop_another_df

        #cur_df = cur_df.sample(min(10, len(cur_df)))
        #another_df = another_df.sample(min(10, len(another_df)))
        cur_df = cur_df.sample(10, replace=True)
        another_df = another_df.sample(10, replace=True)
        #print(len(cur_df), len(another_df))
        #print(crop_state)
        # print(cur_df['x'].max(), cur_df['x'].min())
        # print(cur_df['y'].max(), cur_df['y'].min())
        # print(another_df['x'].max(), another_df['x'].min())
        # print(another_df['y'].max(), another_df['y'].min())
        # print('-'*60)
        cur_bboxes = np.zeros((len(cur_df), 4))
        another_bboxes = np.zeros((len(another_df), 4))
        connections = np.zeros((1, len(cur_df), len(another_df)), dtype='float32')
        for i in range(len(cur_df)):
            cur_row = cur_df.iloc[i]
            bbox = (cur_row['x'], cur_row['y'], cur_row['w'], cur_row['h'])
            cur_bboxes[i] = bbox
            for j in range(len(another_df)):
                another_row = another_df.iloc[j]
                bbox = (another_row['x'], another_row['y'], another_row['w'], another_row['h'])
                another_bboxes[i] = bbox
                #print(i,j,cur_row['id'],another_row['id'] )
                #print('-'*60)
                connections[0, i, j] = (cur_row['id'] == another_row['id'])
        res_dict[IMAGE1] = ToTensor()((im1 / 255).astype('float32'))
        res_dict[IMAGE2] = ToTensor()((im2 / 255).astype('float32'))

        res_dict[BBOX1] = torch.tensor(cur_bboxes)
        res_dict[BBOX2] = torch.tensor(another_bboxes)

        res_dict[CONNECTIONS] = torch.tensor(connections)
        #print(connections.mean())
        #print(res_dict[CONNECTIONS].shape, res_dict[BBOX1].shape, res_dict[BBOX2].shape)
        return res_dict
    return get_bboxes


train_dataset = BaseDataset(
    df=train_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_bboxes('train'),
    ],
)
val_dataset = BaseDataset(
    df=val_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_bboxes('val'),
    ],
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=2,
    num_workers=8,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=2,
    num_workers=8,
    shuffle=False,
)


def gen_log_connected(key):
    def log_connected(data):
        res = None
        d = data[key]
        for i in range(d.shape[1]):
            if res is None:
                res = d[:, i:i+1, :, :]
            else:
                res = torch.cat([res, d[:, i:i+1, :, :]], dim=2)
        return res
    return log_connected


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=TrackEncoderModel(
            'efficientnet_b0',
           # features_block='each',
            connections_classes=1,
            n_layers=1,
        ),
        keys=[
            ([IMAGE1, IMAGE2, BBOX1, BBOX2], ['vectors1', 'vectors2', 'conns']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        DataWrapper(
            BCEDiceLoss(),
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
)
print('pipeline run')
pipeline.run()
