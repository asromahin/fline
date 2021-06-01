DATASET_DIR = '/home/ds/Projects/SARA/datasets/fields_segmentation'
SRC_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/document_crops_rotated/'
DST_IMAGES_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_640_v1/'
SRC_MASKS_PREFIX = '/home/ds/Projects/SARA/annotations/fields_segmentation/'
DST_MASKS_PREFIX = '/home/ds/Projects/SARA/data/generated_images/fields_segm_resize_640_v1_masks/'

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

from cftcvpipeline.pipelines.base_pipeline import BasePipeline

from cftcvpipeline.utils.logging.trains.logger import TrainsLogger
from cftcvpipeline.utils.saver import Saver, MetricStrategies
from cftcvpipeline.utils.data.dataset import BaseDataset

from cftcvpipeline.constants.base import IMAGE

from cftcvpipeline.utils.wrappers import ModelWrapper, DataWrapper
from cftcvpipeline.utils.logging.base import EnumLogDataTypes

from cftcvpipeline.utils.data.image.io import imread

from sklearn.model_selection import train_test_split


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[SARA]DocFake'
TASK_NAME = 'segmentation_fake'
SEG_MODEL = 'seg_model'
MASK = 'mask'
FAKE_IMAGE = 'fake_image'
DEVICE = 'cuda:2'


def iou(pr, gt, eps=1e-7, threshold=None, activation=None):
    """
    Source:
        https://github.com/catalyst-team/catalyst/
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    gt = gt.view(gt.shape[0], gt.shape[1], -1)
    pr = pr.view(pr.shape[0], pr.shape[1], -1)

    intersection = torch.sum(gt * pr, dim=2)
    union = torch.sum(gt, dim=2) + torch.sum(pr, dim=2) - intersection + eps
    return (intersection + eps) / union


def f_score(pr, gt, beta=1, eps=1e-7, threshold=None, activation='sigmoid'):
    """
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    if activation is None or activation == "none":
        activation_fn = lambda x: x
    elif activation == "sigmoid":
        activation_fn = torch.nn.Sigmoid()
    elif activation == "softmax2d":
        activation_fn = torch.nn.Softmax2d()
    else:
        raise NotImplementedError(
            "Activation implemented for sigmoid and softmax2d"
        )

    pr = activation_fn(pr)

    if threshold is not None:
        pr = (pr > threshold).float()

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta ** 2) * tp + eps) \
            / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + eps)

    return score


class DiceLoss(torch.nn.Module):
    __name__ = 'dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__()
        self.activation = activation
        self.eps = eps

    def forward(self, y_pr, y_gt):
        return 1 - f_score(y_pr, y_gt, beta=1., eps=self.eps, threshold=None, activation=self.activation)


class BCELoss(torch.nn.Module):
    __name__ = 'bce_loss'

    def __init__(self):
        super().__init__()
        self.bce = torch.nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, y_pr, y_gt):
        bce = self.bce(y_pr, y_gt)
        return bce


class BCEDiceLoss(DiceLoss):
    __name__ = 'bce_dice_loss'

    def __init__(self, eps=1e-7, activation='sigmoid'):
        super().__init__(eps, activation)
        self.bce = BCELoss()

    def forward(self, y_pr, y_gt):
        dice = super().forward(y_pr, y_gt)
        bce = self.bce(y_pr, y_gt)
        return dice + bce

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

ocr_df = pd.read_csv('/home/ds/Projects/SARA/datasets/ocr/dataset_v59.csv')


def parse_text(annot_path):
    with open(annot_path, 'r') as f:
        annot = json.loads(f.read())
        return annot['file_attributes'].get('value')


ocr_df['text'] = ocr_df['annotation'].apply(parse_text)
ocr_df['original_image'] = ocr_df['image'].str.split('/').str[-3]
ocr_df['field_page'] = ocr_df['image'].str.split('/').str[-2]
ocr_df['field'] = ocr_df['field_page'].str.split('__').str[1]
ocr_df['page'] = ocr_df['field_page'].str.split('__').str[0]

ocr_df['field'] = ocr_df['field'].str.replace('_rus', '_nat')
ocr_df['field'] = ocr_df['field'].str.replace('_lat', '_eng')

ocr_df = ocr_df.dropna()

ocr_groups = ocr_df.groupby('original_image')

valid_images = list(ocr_df['original_image'].unique())

df = df[df['original_image'].str.contains('|'.join(valid_images))]

LABELS = [
    "void",
    "mrz",
    "surname_eng",
    "surname_nat",
    "name_eng",
    "name_nat",
    "patronymic_eng",
    "patronymic_nat",
    "birth_date",
    "gender",
    "number",
    "issue_date",
    "authority_eng",
    "authority_nat",
    "expiration_date",
    "birth_place_eng",
    "birth_place_nat",
    "nationality_eng",
    "nationality_nat",
    "issue_code",
    "mc_passport_number",
    'separator',
]
LABEL_TO_CODE = {cls: i for i, cls in enumerate(LABELS)}
CODE_TO_LABEL = {i: cls for i, cls in enumerate(LABELS)}

train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

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


def get_mapping_mask(mask, ocr_rows):
    res = {}
    for m in np.unique(mask):
        if m != 0:
            cur_mask = (mask == m).astype('uint8')
            conn_max, conn = cv2.connectedComponents(cur_mask)
            key = CODE_TO_LABEL[m]
            if not 'gender' in key:
                for i, c in enumerate(np.unique(conn)):
                    if i != 0:
                        cur_key = key
                        if conn_max > 2:
                            cur_key += '_' + str(i - 1)
                        res_mask = (conn == c).astype('uint8')
                        cntrs, _ = cv2.findContours(res_mask, 0, 1)
                        text = ocr_rows[ocr_rows['field'] == cur_key]

                        if len(text) == 1 and len(cntrs) == 1:
                            x,y,w,h = cv2.boundingRect(cntrs[0])
                            if w > 20 and h > 20:
                                text = text.iloc[0]['text']
                                #print(text)
                                res[cur_key] = (text, res_mask)
    return res


dg = DocGenerator(-1)


def generate_get_images(mode='train'):
    def get_images(row, res_dict):
        im = imread(row[IMAGE]).astype('uint8')
        mask = imread(row[MASK]).astype('uint8')
        im = cv2.resize(im, SHAPE)
        mask = cv2.resize(mask, SHAPE)
        mask = mask[:, :, 2:3]

        ocr_rows = ocr_groups.get_group(row['original_image'])
        map_masks = get_mapping_mask(mask, ocr_rows)
        seg_mask = None
        try:
            r = dg.generate(im, map_masks)
            for key, (text, m) in map_masks.items():
                #print(m.shape)
                #print(m.max(), m.min())
                if seg_mask is None:
                    seg_mask = m
                else:
                    seg_mask = np.logical_or(seg_mask, m)
        except:
            #print('error')
            r = im
        if seg_mask is None:
            #print('none')
            r = im
        if np.abs(im - r).max() == 0:
            seg_mask = np.zeros(im.shape[:2], dtype='float32')
        seg_mask = np.expand_dims(seg_mask, axis=-1)
        #print(seg_mask.max(), seg_mask.min())
        res_dict[IMAGE] = ToTensor()((r / 255).astype('float32'))
        #print(seg_mask.shape)
        res_dict[MASK] = ToTensor()(seg_mask.astype('float32'))
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
        model=smp.Unet(encoder_name='efficientnet-b0', decoder_attention_type='scse', classes=1, activation='sigmoid'),
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
