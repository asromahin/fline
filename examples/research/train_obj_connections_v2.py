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

from fline.models.models.research.history.object_detection_model import (
    ConnectionsSMPV4,
)

os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (256, 256)
PROECT_NAME = '[RESEARCH]ObjectDetection'
TASK_NAME = 'test_tv_human_interaction_v3_test'
SEG_MODEL = 'seg_model'
MASK = 'mask'
BBOX = 'bbox'
DEVICE = 'cuda:2'
IMAGE = 'IMAGE'
CONNECTIONS = 'connections'


df = pd.read_csv('/data/research/tv_human_interaction/prepare_df.csv')
df[IMAGE] = df['frame_path']
df = df[~df['frame_path'].str.contains('negative')]
#df = df.groupby('original_video').sample(n=1, random_state=1)
videos = list(df['original_video'].unique())
train_videos = videos[:len(videos)*2//3]
val_videos = videos[len(videos)*2//3:]

train_df = df[df['original_video'].str.contains('|'.join(train_videos))]
val_df = df[df['original_video'].str.contains('|'.join(val_videos))]

classes = [l for l in list(df['interaction_type'].unique()) if 'no_interaction' not in l]
class_to_code = {cl: i for i, cl in enumerate(classes)}
#print(classes)
print(class_to_code)
groups = df.groupby('frame_path')
#frame_paths = list(df['frame_path'].unique())
train_df = pd.DataFrame({'frame_path': list(train_df['frame_path'].unique())})
val_df = pd.DataFrame({'frame_path': list(val_df['frame_path'].unique())})
#train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)
#train_df = train_df.iloc[1400:]
#train_df = train_df.drop(list(range(1400, 1500)))

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


def generate_get_bboxes(mode='train'):
    def get_bboxes(row, res_dict):
        im = cv2.imread(row['frame_path']).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        cur_df = groups.get_group(row['frame_path']).copy()
        cur_df['id_cat'] = cur_df['id'].astype("category").cat.codes
        mask = np.zeros((*SHAPE, len(cur_df)+1), dtype='int')
        connections_mask = np.zeros((len(classes), len(cur_df), len(cur_df)), dtype='float32')
       # print(cur_df['id'].unique())
        for i in range(len(cur_df)):
            crow = cur_df.iloc[i]
            x = int(crow['x'] / im.shape[1] * SHAPE[0])
            y = int(crow['y'] / im.shape[0]*SHAPE[1])
            x2 = int((crow['x']+crow['size'])/im.shape[1]*SHAPE[0])
            y2 = int((crow['y']+crow['size'])/im.shape[0]*SHAPE[1])
            #other_mask = mask > 0

            #print(x,y,x2,y2)
            mask[:, :, i+1] = cv2.rectangle(
                mask[:, :, i+1].astype('uint8'),
                (x, y),
                (x2, y2),
                1,
                -1,
            )
            #intersection = other_mask[y:y2, x:x2].sum()
            #if intersection == 1 and other_mask.sum() > intersection:
            #    np.putmask(mask, ~other_mask, buf)
            #else:
            #    mask = buf
            if pd.notna(crow['interact_id']):
                another_row = cur_df[cur_df['id'] == crow['interact_id']].iloc[0]
                connections_mask[int(class_to_code[crow['interaction_type']]), int(crow['id_cat']), int(another_row['id_cat'])] = 1

        im = cv2.resize(im, SHAPE)
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[MASK] = ToTensor()(mask.astype('int'))
        res_dict[CONNECTIONS] = torch.tensor(connections_mask)
        #print(res_dict[IMAGE].shape, res_dict[MASK].shape, res_dict[CONNECTIONS].shape)
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
    batch_size=1,
    num_workers=4,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
)


def gen_log_connected(key):
    def log_connected(data):
        res = None
        d = data[key]
        for i in range(d.shape[1]):
            if res is None:
                res = d[:,i:i+1,:,:]
            else:
                res = torch.cat([res, d[:,i:i+1,:,:]], dim=2)
        return res
    return log_connected


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=ConnectionsSMPV4(
            'efficientnet-b0',
           # features_block='each',
            connections_classes=len(classes),

        ),
        keys=[
            ([IMAGE, MASK], ['vectors', 'conns']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
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
            IMAGE: EnumLogDataTypes.image,
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
