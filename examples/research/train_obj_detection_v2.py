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

from fline.models.models.research.history.object_detection_model import (
    ObjectDetectionSMP,
    ObjectDetectionSMPV2,
)

from sklearn.model_selection import train_test_split
from torch.utils.data._utils.collate import default_collate

from fline.losses.segmentation.dice import BCEDiceLoss
from fline.utils.data.image.io import imread_padding


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


#SHAPE = (256, 256)
PROECT_NAME = '[RESEARCH]ObjectDetection'
TASK_NAME = 'test_corner_net'
SEG_MODEL = 'seg_model'
MASK = 'mask'
BBOX = 'bbox'
IMAGE = 'IMAGE'
CONNECTIONS = 'connections'
CORNERS = 'CORNERS'

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


def generate_get_bboxes(mode='train'):
    def get_bboxes(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        #im = cv2.imread(row[IMAGE]).astype('uint8')
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(row[MASK]).astype('uint8')
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, 2]
        SHAPE = im.shape
        #print(np.unique(mask))
        r_mask = np.zeros((*SHAPE, 50), dtype='int')
        last_i = 0
        bboxes = np.zeros((50, 4), dtype='float32')
        is_one_field = np.zeros((50,), dtype='int')
        bbox_corners = np.zeros((*SHAPE,))
        bbox_corners_mask = np.zeros((*SHAPE, 3))
        bbox_corners_labels = np.zeros((*SHAPE, 50))
        #lt_corner = np.zeros((50,), dtype='int')
        #rb_corner = np.zeros((50,), dtype='int')
        for i, m in enumerate(np.unique(mask)):
            if m != 0:
                c_mask = (mask == m).astype('uint8')
                c_mask = cv2.resize(
                    c_mask.astype('uint8'),
                    SHAPE,
                    interpolation=cv2.INTER_NEAREST,
                ).astype('uint8')
                cntrs, _ = cv2.findContours(c_mask, 1, 2)
                max_it = 0
                for t, cnt in enumerate(cntrs):
                    x, y, w, h = cv2.boundingRect(cnt)
                    #print(x,y,w,h, y+h, x+w)
                    if w <= 1 and h <= 1:
                        w = w + 2
                        h = h + 2
                    if y+h >= SHAPE[1]:
                        y = SHAPE[1]-h-1
                    if x+w >= SHAPE[0]:
                        x = SHAPE[0]-w-1
                    if y-h <= 0:
                        y = 1
                    if x-w >= 0:
                        x = 1
                    if bbox_corners[y, x] == 1 or bbox_corners[y+h, x+w] == 2:
                        continue
                    if bbox_corners_mask[y, x, 1] == 1 or bbox_corners_mask[y+h, x+w, 2] == 1:
                        continue
                    bbox_corners[y, x] = 1
                    bbox_corners[y+h, x+w] = 2
                    bbox_corners_mask[y, x, 1] = 1
                    bbox_corners_mask[y+h, x+w, 2] = 1
                    #print(y,x, y+h, x+w, bbox_corners_mask[:, :, 1].sum(), bbox_corners_mask[:, :, 2].sum(), len(cntrs))
                    bbox_corners_labels[y, x, last_i + t] = 1
                    bbox_corners_labels[y+h, x+w, last_i + t] = 1
                    #lt_corner[last_i+t] = last_i+1
                    #rb_corner[last_i+t] = last_i+1
                    # buf = np.zeros(SHAPE)
                    # buf = cv2.drawContours(
                    #     buf,
                    #     [cnt], -1, 1, -1,
                    # )
                    # r_mask[:, :, last_i + t] = buf
                    x = (x + w/2) / c_mask.shape[1]
                    y = (y + h/2) / c_mask.shape[0]
                    w = w / c_mask.shape[1]
                    h = h / c_mask.shape[0]
                    bboxes[last_i+t] = (x, y, w, h)
                    max_it += 1
                    if len(cntrs) > 1:
                        is_one_field[last_i+t] = last_i+1
                # conn_max, conn = cv2.connectedComponents(c_mask)
                # for t in range(1, max_it):
                #     r_mask[:, :, last_i+t] = (conn == t)
                #     #print(last_i+t)
                last_i += max_it
                #print('last_i=', last_i)
        #print('-'*60)
        bboxes = bboxes[:last_i]
        #bbox_corners = bbox_corners[:last_i]
        #idx = np.arange(len(bboxes))
        #np.random.shuffle(idx)
        #bboxes = bboxes[idx]
        #is_one_field = is_one_field[:last_i]
       #is_one_field = is_one_field[idx]
        #np.random.choice(bboxes, len(bboxes))
       # idx = [i+1 for i in idx]
        #idx.insert(0, 0)
        #r_mask = r_mask[:, :, idx]
        bbox_corners_labels = bbox_corners_labels[:,:,:last_i].argmax(axis=2)
        lt_mask = bbox_corners_mask[:,:,1] == 1
        rb_mask = bbox_corners_mask[:,:,2] == 1
        #print(lt_mask.sum(), rb_mask.sum())
        #print(bbox_corners.shape, lt_mask.shape, bbox_corners_labels.shape)
        lt = bbox_corners_labels[lt_mask[:,:]]
        rb = bbox_corners_labels[rb_mask[:,:]]
        bbox_corners_links = np.zeros((1, len(lt), len(rb)), dtype='float32')
        for i, p1 in enumerate(lt):
            for j, p2 in enumerate(rb):
                if p1 == p2:
                    #print(i,j)
                    bbox_corners_links[0, i, j] = 1
        #r_mask = r_mask.argmax(axis=2)
        # connections_mask = np.zeros((3, len(bboxes), len(bboxes)), dtype='float32')
        # for i, box1 in enumerate(bboxes):
        #     dif_up_y_min = np.inf
        #     select_up = None
        #     dif_down_y_min = np.inf
        #     select_down = None
        #     for j, box2 in enumerate(bboxes):
        #         if i != j:
        #             #connections_mask[0, i, j] = box1[0] - box2[0] > 0.05
        #             #connections_mask[1, i, j] = box2[0] - box1[0] > 0.05
        #             #connections_mask[2, i, j] = box1[1] - box2[1] > 0.05
        #             #connections_mask[3, i, j] = box2[1] - box1[1] > 0.05
        #             cur_dif_down = box1[1] - box2[1]
        #             cur_dif_up = - cur_dif_down
        #             if cur_dif_down > 0.05 and cur_dif_down < dif_down_y_min:
        #                 dif_down_y_min = cur_dif_down
        #                 select_down = j
        #             if cur_dif_up > 0.05 and cur_dif_up < dif_up_y_min:
        #                 dif_up_y_min = cur_dif_up
        #                 select_up = j
        #     if select_up is not None:
        #         connections_mask[0, i, select_up] = 1
        #     if select_down is not None:
        #         connections_mask[1, i, select_down] = 1
        #print(is_one_field)
        #print(r_mask.max())
        # for u in np.unique(is_one_field):
        #     if u != 0:
        #         cur_field_mask = (is_one_field == u)
        #         cur_indexes = np.where(cur_field_mask)[0]
        #         #print(cur_indexes)
        #         for i in cur_indexes:
        #             for j in cur_indexes:
        #                 if i != j:
        #                     connections_mask[2, i, j] = 1

        #mask = r_mask
        #a = len(np.unique(mask))
        im = cv2.resize(im.astype('uint8'), SHAPE).astype('uint8')
        #mask = cv2.resize(mask.astype('uint8'), SHAPE, interpolation=cv2.INTER_NEAREST).astype('uint8')
        #print(a, len(np.unique(mask)))
        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        im = im.astype('float32')
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[BBOX] = bboxes
        res_dict[CORNERS] = torch.tensor(bbox_corners)
        res_dict[MASK] = ToTensor()(bbox_corners_mask)
        #res_dict[MASK] = ToTensor()(mask.astype('float32'))
        res_dict[CONNECTIONS] = torch.tensor(bbox_corners_links)
        return res_dict
    return get_bboxes



def generate_get_bboxes(mode='train'):
    def get_bboxes(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        #im = cv2.imread(row[IMAGE]).astype('uint8')
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #mask = cv2.imread(row[MASK]).astype('uint8')
        #mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
        mask = mask[:, :, 2]
        SHAPE = (im.shape[0], im.shape[1])
        left_top_points = np.zeros((*SHAPE, 50))
        right_bottom_points = np.zeros((*SHAPE, 50))
        pairs_counter = 0

        for i, m in enumerate(np.unique(mask)):
            if m != 0:
                c_mask = (mask == m).astype('uint8')
                # c_mask = cv2.resize(
                #     c_mask.astype('uint8'),
                #     SHAPE,
                #     interpolation=cv2.INTER_NEAREST,
                # ).astype('uint8')
                cntrs, _ = cv2.findContours(c_mask, 1, 2)
                for cnt in cntrs:
                    x, y, w, h = cv2.boundingRect(cnt)
                    left_top_points[y, x, pairs_counter] = 1
                    right_bottom_points[y+h-1, x+w-1, pairs_counter] = 1
                    pairs_counter += 1
        left_top_points = left_top_points[:, :, :pairs_counter]
        right_bottom_points = right_bottom_points[:, :, :pairs_counter]
        idx = np.arange(pairs_counter)
        np.random.shuffle(idx)
        right_bottom_points = right_bottom_points[:, :, idx]
        bbox_corners_links = np.zeros((1, pairs_counter, pairs_counter), dtype='float32')
        for p in range(pairs_counter):
            bbox_corners_links[0, p, idx[p]] = 1
        left_top_mask = np.max(left_top_points, axis=2)
        right_bottom_mask = np.max(right_bottom_points, axis=2)
        res_mask = np.zeros((*SHAPE, 2), dtype='float32')
        res_mask[:, :, 0] = left_top_mask
        res_mask[:, :, 1] = right_bottom_mask
        #im = cv2.resize(im.astype('uint8'), SHAPE).astype('uint8')
        im = im.astype('float32')
        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[MASK] = ToTensor()(res_mask)
        res_dict[CONNECTIONS] = torch.tensor(bbox_corners_links)
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
    num_workers=2,
    shuffle=True,
    #collate_fn=collate_fn,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=1,
    num_workers=2,
    shuffle=False,
    #collate_fn=collate_fn,
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


class SumLoss(torch.nn.Module):
    def __init__(self):
        super(SumLoss, self).__init__()
        self.bce = torch.nn.BCELoss()
        self.dice = BCEDiceLoss(activation=None)
    def forward(self, matrix, connection, out_seg, mask):
        return self.bce(matrix, connection) + self.bce(out_seg, mask)


pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=ObjectDetectionSMPV2(
            'efficientnet-b0',
           # features_block='each',
            #connections_classes=1,

        ),
        keys=[
            ([IMAGE, MASK], ['out_matrix', 'out_seg']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        DataWrapper(
            SumLoss(),
            keys=[
                (['out_matrix', CONNECTIONS, 'out_seg', MASK], ['loss'])
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
            'out_matrix': ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=gen_log_connected('out_matrix'),
            ),
            'loss': EnumLogDataTypes.loss,
            IMAGE: EnumLogDataTypes.image,
            'out_seg': ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=lambda x: torch.cat([x['out_seg'][:,0:1,:,:], x['out_seg'][:,1:2,:,:]], dim=2),
            ),
            MASK: ProcessImage(
                log_type=EnumLogDataTypes.image,
                process_func=lambda x: torch.cat([x[MASK][:,0:1,:,:], x[MASK][:,1:2,:,:]], dim=2),
            ),
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
