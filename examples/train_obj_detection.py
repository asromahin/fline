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
import json

import albumentations as alb

from cftcvpipeline.pipelines.base_pipeline import BasePipeline

from cftcvpipeline.utils.logging.trains.logger import TrainsLogger
from cftcvpipeline.utils.saver import Saver, MetricStrategies
from cftcvpipeline.utils.data.dataset import BaseDataset

from cftcvpipeline.constants.base import IMAGE

from cftcvpipeline.utils.wrappers import ModelWrapper, DataWrapper
from cftcvpipeline.utils.logging.base import EnumLogDataTypes

from cftcvpipeline.utils.logging.trains.groups import ProcessImage

from cftcvpipeline.utils.data.image.io import imread_padding

from cftcvpipeline.models.research.connected_model import ConnectLossV3
from cftcvpipeline.models.research.object_detection_model import BboxLoss, ObjectDetectionModelV3

from sklearn.model_selection import train_test_split


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[RESEARCH]ObjectDetection'
TASK_NAME = 'test_deep_vector'
SEG_MODEL = 'seg_model'
MASK = 'mask'
BBOX = 'bbox'
DEVICE = 'cuda:1'


df = pd.read_csv(os.path.join(DATASET_DIR, 'dataset_v93.csv'))
df[IMAGE] = df['image'].str.replace(SRC_IMAGES_PREFIX, DST_IMAGES_PREFIX)
df[MASK] = df['annotation'].str.replace(SRC_MASKS_PREFIX, DST_MASKS_PREFIX) + '.png'

df['original_image'] = df['image'].str.split('/').str[-3]
#df = df.iloc[500:700]


def add_classes(df):
    cdf = pd.read_csv('/home/ds/Projects/SARA/datasets/classification/dataset_v75.csv')
    def extract_class(annot_path):
        with open(annot_path) as f:
            annot = json.loads(f.read())
            document_types = annot['file_attributes']['document_type']
            if len(document_types.keys()) == 1:
                return list(document_types.keys())[0]
    cdf['class'] = cdf['annotation'].apply(extract_class)
    cdf['original_image'] = cdf['image'].str.split('/').str[-1]
    cdf = cdf.dropna(subset=['class'])
    del(cdf['annotation'])
    df = pd.merge(
        left=df,
        right=cdf,
        left_on='original_image',
        right_on='original_image',
        how='inner',
    )
    return df

df = add_classes(df)
df = df[~df['class'].str.contains('играц')]

df['valid_mask'] = df[MASK].apply(lambda x: cv2.imread(x) is not None)

df = df[df['valid_mask']]

train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

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


def generate_get_mask(mode='train'):
    def get_mask(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        mask = mask[:, :, 2:3]
        r_mask = np.zeros(mask.shape, dtype='int')
        last_i = 1
        for i, m in enumerate(np.unique(mask)):
            if m != 0:
                c_mask = (mask == m).astype('uint8')[:, :, 0]
                conn_max, conn = cv2.connectedComponents(c_mask)
                #print(conn_max, np.unique(conn), m)
                c_mask = c_mask * last_i
                conn = conn + c_mask
                r_mask[:, :, 0] += conn
                last_i += conn_max
        mask = r_mask
        #print(mask.max())
        if mode == 'train':
            aug = train_augs_geometric(image=im, mask=mask)
            im, mask = aug['image'], aug['mask']
        if mode == 'train':
            im = train_augs_values(image=im.astype('uint8'))['image']
        im = im.astype('float32')
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[MASK] = ToTensor()(mask.astype('float32'))
        res_dict[MASK+'bin'] = ToTensor()((mask > 0).astype('float32'))
        return res_dict
    return get_mask


def generate_get_bboxes(mode='train'):
    def get_bboxes(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        mask = mask[:, :, 2:3]
        r_mask = np.zeros(mask.shape, dtype='int')
        last_i = 1
        bboxes = []
        for i, m in enumerate(np.unique(mask)):
            if m != 0:
                c_mask = (mask == m).astype('uint8')[:, :, 0]
                cntrs, _ = cv2.findContours(c_mask, 0, 1)
                for cnt in cntrs:
                    x,y,w,h = cv2.boundingRect(cnt)
                    x = (x + w/2) / im.shape[1]
                    y = (y + h/2) / im.shape[0]
                    w = w / im.shape[1]
                    h = h / im.shape[0]
                    bboxes.append((x,y,w,h))
                conn_max, conn = cv2.connectedComponents(c_mask)
                c_mask = c_mask * last_i
                conn = conn + c_mask
                r_mask[:, :, 0] += conn
                last_i += conn_max
        mask = r_mask
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        im = im.astype('float32')
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[BBOX] = bboxes
        res_dict[MASK] = ToTensor()(mask.astype('float32'))
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


def log_connected(data):
    res = None
    d = data['conn']
    for i in range(d.shape[1]):
        if res is None:
            res = d[:,i:i+1,:,:]
        else:
            res = torch.cat([res, d[:,i:i+1,:,:]], dim=2)
    return res

class AgregateLoss(torch.nn.Module):
    def __init__(self, device):
        super(AgregateLoss, self).__init__()
        self.bbox_loss = BboxLoss(device)
        self.connect_loss = ConnectLossV3(device)
    def forward(self, bbox_pred, bbox_target, mask_pred, mask_target):
        return self.bbox_loss(bbox_pred, bbox_target) + self.connect_loss(mask_pred, mask_target)

pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=ObjectDetectionModelV3('efficientnet_b0', last_features=64, connected_channels=32),
        keys=[
            ([IMAGE], ['bboxes', 'conn']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        DataWrapper(
            AgregateLoss(DEVICE),
            keys=[
                (['bboxes', BBOX, 'conn', MASK], ['loss'])
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
            'conn': ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=log_connected,
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
