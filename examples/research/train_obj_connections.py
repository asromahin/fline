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

from fline.pipelines.base_pipeline import BasePipeline

from fline.utils.logging.trains import TrainsLogger
from fline.utils.saver import Saver, MetricStrategies
from fline.utils.data.dataset import BaseDataset

from fline.utils.wrappers import ModelWrapper, DataWrapper
from fline.utils.logging.base import EnumLogDataTypes

from fline.utils.logging.groups import ProcessImage

from fline.utils.data.image.io import imread_padding

from fline.models.models.research.history.object_detection_model import (
    ConnectionsSMPV3,
)

from sklearn.model_selection import train_test_split
from torch.utils.data._utils.collate import default_collate

os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (256, 256)
PROECT_NAME = '[RESEARCH]ObjectDetection'
TASK_NAME = 'test_new_connection_with_shuffle'
SEG_MODEL = 'seg_model'
MASK = 'mask'
BBOX = 'bbox'
DEVICE = 'cuda:1'
IMAGE = 'IMAGE'
CONNECTIONS = 'connections'


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
        #im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        #mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        im = cv2.imread(row[IMAGE]).astype('uint8')
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row[MASK]).astype('uint8')
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, 2:3]
        r_mask = np.zeros((*SHAPE, 50), dtype='int')
        last_i = 0
        bboxes = np.zeros((50, 4), dtype='float32')
        is_one_field = np.zeros((50,), dtype='int')
        for i, m in enumerate(np.unique(mask)):
            if m != 0:
                c_mask = (mask == m).astype('uint8')[:, :, 0]
                c_mask = cv2.resize(
                    c_mask.astype('uint8'),
                    SHAPE,
                    interpolation=cv2.INTER_NEAREST,
                ).astype('uint8')
                cntrs, _ = cv2.findContours(c_mask, 0, 1)
                for t, cnt in enumerate(cntrs):
                    x, y, w, h = cv2.boundingRect(cnt)
                    #print(x,y,w,h)
                    if w <= 1 and h <= 1:
                        w = w + 2
                        h = h + 2

                    x = (x + w/2) / c_mask.shape[1]
                    y = (y + h/2) / c_mask.shape[0]
                    w = w / c_mask.shape[1]
                    h = h / c_mask.shape[0]
                    bboxes[last_i+t] = (x, y, w, h)
                    if len(cntrs) > 1:
                        is_one_field[last_i+t] = last_i+1
                conn_max, conn = cv2.connectedComponents(c_mask)
                for t in range(1, conn_max):
                    r_mask[:, :, last_i+t] = (conn == t)
                    #print(last_i+t)
                last_i += conn_max-1
                #print('last_i=', last_i)
        bboxes = bboxes[:last_i]
        idx = np.arange(len(bboxes))
        np.random.shuffle(idx)
        bboxes = bboxes[idx]
        is_one_field = is_one_field[:last_i]
        is_one_field = is_one_field[idx]
        #np.random.choice(bboxes, len(bboxes))
        idx = [i+1 for i in idx]
        idx.insert(0, 0)
        r_mask = r_mask[:, :, idx]
        r_mask = r_mask.argmax(axis=2)
        connections_mask = np.zeros((3, len(bboxes), len(bboxes)), dtype='float32')
        for i, box1 in enumerate(bboxes):
            dif_up_y_min = np.inf
            select_up = None
            dif_down_y_min = np.inf
            select_down = None
            for j, box2 in enumerate(bboxes):
                if i != j:
                    #connections_mask[0, i, j] = box1[0] - box2[0] > 0.05
                    #connections_mask[1, i, j] = box2[0] - box1[0] > 0.05
                    #connections_mask[2, i, j] = box1[1] - box2[1] > 0.05
                    #connections_mask[3, i, j] = box2[1] - box1[1] > 0.05
                    cur_dif_down = box1[1] - box2[1]
                    cur_dif_up = - cur_dif_down
                    if cur_dif_down > 0.05 and cur_dif_down < dif_down_y_min:
                        dif_down_y_min = cur_dif_down
                        select_down = j
                    if cur_dif_up > 0.05 and cur_dif_up < dif_up_y_min:
                        dif_up_y_min = cur_dif_up
                        select_up = j
            if select_up is not None:
                connections_mask[0, i, select_up] = 1
            if select_down is not None:
                connections_mask[1, i, select_down] = 1
        #print(is_one_field)
        #print(r_mask.max())
        for u in np.unique(is_one_field):
            if u != 0:
                cur_field_mask = (is_one_field == u)
                cur_indexes = np.where(cur_field_mask)[0]
                #print(cur_indexes)
                for i in cur_indexes:
                    for j in cur_indexes:
                        if i != j:
                            connections_mask[2, i, j] = 1

        mask = r_mask
        #a = len(np.unique(mask))
        im = cv2.resize(im.astype('uint8'), SHAPE).astype('uint8')
        #mask = cv2.resize(mask.astype('uint8'), SHAPE, interpolation=cv2.INTER_NEAREST).astype('uint8')
        #print(a, len(np.unique(mask)))
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        im = im.astype('float32')
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[BBOX] = bboxes
        res_dict[MASK] = ToTensor()(mask.astype('float32'))
        res_dict[CONNECTIONS] = torch.tensor(connections_mask)
        return res_dict
    return get_bboxes

def collate_fn(batch):
    max_size = max([b[BBOX].shape[0] for b in batch])
    for i, b in enumerate(batch):
        old_size = b[BBOX].shape[0]
        if old_size != max_size:
            buf = torch.zeros((max_size, b[BBOX].shape[1]))
            buf[:b[BBOX].shape[0]] = b[BBOX]
            batch[i][BBOX] = buf
    res_batch = default_collate(batch)
    #[print(k, res_batch[k].shape) for k in res_batch.keys()]
    return res_batch




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
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=4,
    shuffle=False,
    collate_fn=collate_fn,
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
        model=ConnectionsSMPV3(
            'efficientnet-b0',
           # features_block='each',
            connections_classes=3,

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
