DATASET_DIR = '/home/ds/Projects/SARA/datasets/fields_segmentation'
SRC_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/document_crops_rotated/'
DST_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_512/'
SRC_MASKS_PREFIX = '/home/ds/Projects/SARA/annotations/fields_segmentation/'
DST_MASKS_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_512_masks/'

import sys
sys.path.insert(0,'/home/romakhin/rep/cftcv')
from cftcv.doc_generator.doc_generator import DocGenerator

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
TASK_NAME = 'fake_classifier_v2'
SEG_MODEL = 'seg_model'
MASK = 'mask'
FAKE_IMAGE = 'fake_image'
DEVICE = 'cuda:2'
IMAGE = 'IMAGE'


df = pd.read_csv(os.path.join(DATASET_DIR, 'dataset_v97.csv'))
df['original_image'] = df['image'].str.split('/').str[-3]
df[IMAGE] = df['image'].str.replace(SRC_IMAGES_PREFIX, DST_IMAGES_PREFIX)
df[MASK] = df['annotation'].str.replace(SRC_MASKS_PREFIX, DST_MASKS_PREFIX) + '.png'
df['segmentation_name'] = df['image'].str.split('/').str[-1]

df = df.iloc[500:700]

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


def parse_text(annot_path):
    with open(annot_path, 'r') as f:
        annot = json.loads(f.read())
        return annot['file_attributes'].get('value')


ocr_df = pd.read_csv('/home/ds/Projects/SARA/datasets/ocr/dataset_v60.csv')
ocr_df['text'] = ocr_df['annotation'].apply(parse_text)
ocr_df['segmentation_name'] = ocr_df['image'].str.split('__').str[-2]
ocr_df['original_image'] = ocr_df['image'].str.split('/').str[-3]
ocr_df['field_page'] = ocr_df['image'].str.split('/').str[-2]
ocr_df['field'] = ocr_df['field_page'].str.split('__').str[1]
ocr_df['page'] = ocr_df['field_page'].str.split('__').str[0]
ocr_df['field'] = ocr_df['field'].str.replace('_rus', '_nat')
ocr_df['field'] = ocr_df['field'].str.replace('_lat', '_eng')
ocr_df = ocr_df.dropna()
ocr_df = ocr_df[~ocr_df['field'].str.contains('gender')]
m = ocr_df['field'] == 'authority'
ocr_df['field'].loc[m] = 'authority_eng'
m = ocr_df['field'] == 'authority_1'
ocr_df['field'].loc[m] = 'authority_eng_1'
m = ocr_df['field'] == 'authority_2'
ocr_df['field'].loc[m] = 'authority_eng_2'
m = ocr_df['field'] == 'authority_3'
ocr_df['field'].loc[m] = 'authority_eng_3'
m = ocr_df['field'] == 'surname'
ocr_df['field'].loc[m] = 'surname_eng'
m = ocr_df['field'] == 'surname_1'
ocr_df['field'].loc[m] = 'surname_eng_1'
m = ocr_df['field'] == 'surname_2'
ocr_df['field'].loc[m] = 'surname_eng_2'
m = ocr_df['field'] == 'surname_3'
ocr_df['field'].loc[m] = 'surname_eng_3'
#ocr_groups = ocr_df.groupby('segmentation_name')
m = pd.merge(df, ocr_df, how='right', on='segmentation_name')
m = m.dropna().reset_index()
m[IMAGE] = m['image_x']
m['annotation'] = m['annotation_x']
segmentation_names = list(m['segmentation_name'].unique())
ocr_groups = m.groupby('segmentation_name')
rdf = [{'name': name} for name in segmentation_names]
rdf = pd.DataFrame(rdf)


def get_mapping_mask(mask, ocr_rows):
    res = {}
    for m in np.unique(mask):
        if m != 0:
            cur_mask = (mask == m).astype('uint8')
            conn_max, conn = cv2.connectedComponents(cur_mask)
            key = CODE_TO_LABEL[m - 1]
            if not 'gender' in key:
                for i, c in enumerate(np.unique(conn)):
                    if i != 0:

                        cur_key = key
                        if conn_max > 2:
                            cur_key += '_' + str(i - 1)
                        res_mask = (conn == c).astype('uint8')
                        cntrs, _ = cv2.findContours(res_mask, 0, 1)
                        x, y, w, h = cv2.boundingRect(cntrs[0])
                        if 1.2 * w < h:
                            continue
                        text = ocr_rows[ocr_rows['field'] == cur_key]
                        if len(text) == 1:
                            text = text.iloc[0]['text']
                            res[cur_key] = (text, res_mask)
    return res


LABELS = [
    'separator',
    'mrz',
    'surname_eng',
    'surname_nat',
    'name_eng',
    'name_nat',
    'patronymic_eng',
    'patronymic_nat',
    'birth_date',
    'gender',
    'number',
    'issue_date',
    'authority_eng',
    'authority_nat',
    'expiration_date',
    'birth_place_eng',
    'birth_place_nat',
    'nationality_eng',
    'nationality_nat',
    'issue_code',
    'mc_passport_number',
    'series',
    'personal_number',
    'registration_address',
]

LABEL_TO_CODE = {cls: i for i, cls in enumerate(LABELS)}
CODE_TO_LABEL = {i: cls for i, cls in enumerate(LABELS)}

train_df, val_df = train_test_split(rdf, test_size=0.25, random_state=42)

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


def get_random_color():
    w = np.random.randint(5)
    if w == 0:
        return (
                np.random.randint(0, 50),
                np.random.randint(0, 50),
                np.random.randint(0, 50),
        ),
    else:
        return (
            0,
            0,
            0,
        )


dg = DocGenerator(
            iou_thresh=-1,
            color=(0, 0, 0),
            # padding=lambda: (
            #         np.random.randint(0, 5),
            #         np.random.randint(0, 5),
            #         np.random.randint(0, 5),
            #         np.random.randint(0, 5),
            # ),
            padding=(0, 0, 0, 0),
)


def generate_get_images(mode='train'):
    def get_images(trow, res_dict):
        ocr_rows = ocr_groups.get_group(trow['name'])
        row = ocr_rows.iloc[0]
        im = imread(row[IMAGE]).astype('uint8')
        mask = imread(row[MASK]).astype('uint8')
        im = cv2.resize(im, SHAPE)
        mask = cv2.resize(mask, SHAPE)
        mask = mask[:, :, 2:3]

        r = dg.generate(im, get_mapping_mask(mask, ocr_rows))
        res_dict['label'] = 1
        if np.abs(im - r).max() == 0:
            res_dict['label'] = 0
        if mode == 'train':
            r = train_augs_geometric(image=r)['image']
            r = train_augs_values(image=r)['image']
        #print(res_dict['label'])
        res_dict[IMAGE] = ToTensor()((r / 255).astype('float32'))
        return res_dict
    return get_images


class AgregateLoss(torch.nn.Module):
    def __init__(self):
        super(AgregateLoss, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, out1, label1, out2, label2):
        return self.loss(torch.nn.functional.softmax(out1, dim=1), label1) + self.loss(torch.nn.functional.softmax(out2, dim=1), label2)


train_dataset = BaseDataset(
    df=train_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_images('val'),
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
        model=timm.create_model('resnet18', num_classes=2),
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
