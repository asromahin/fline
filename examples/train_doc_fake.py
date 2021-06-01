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

import segmentation_models_pytorch as smp
import albumentations as alb


from cftcvpipeline.pipelines.base_pipeline import BasePipeline

from cftcvpipeline.utils.logging.trains.logger import TrainsLogger
from cftcvpipeline.utils.saver import Saver, MetricStrategies
from cftcvpipeline.utils.data.dataset import BaseDataset
from cftcvpipeline.utils.data.callbacks.base import generate_imread_callback

from cftcvpipeline.constants.base import IMAGE

from cftcvpipeline.utils.wrappers import ModelWrapper, DataWrapper
from cftcvpipeline.utils.logging.base import EnumLogDataTypes

from cftcvpipeline.utils.logging.trains.groups import ConcatImage, ProcessImage

from cftcvpipeline.utils.data.image.io import imread_padding, imread_resize

from sklearn.model_selection import train_test_split


os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


SHAPE = (512, 512)
PROECT_NAME = '[SARA]DocFake'
TASK_NAME = 'segmentation_fields_more_aug_finetune_with_model'
SEG_MODEL = 'seg_model'
MASK = 'mask'
DEVICE = 'cuda:2'

df = pd.read_csv(os.path.join(DATASET_DIR, 'dataset_v93.csv'))
df[IMAGE] = df['image'].str.replace(SRC_IMAGES_PREFIX, DST_IMAGES_PREFIX)
df[MASK] = df['annotation'].str.replace(SRC_MASKS_PREFIX, DST_MASKS_PREFIX) + '.png'

df['original_image'] = df['image'].str.split('/').str[-3]
#df = df.iloc[1200:1300]


def add_classes(df):
    cdf = pd.read_csv('/home/ds/Projects/SARA/datasets/classification/dataset_v75.csv')
    def extract_class(annot_path):
        with open(annot_path) as f:
            annot = json.loads(f.read())
            document_types = annot['file_attributes']['document_type']
            if len(document_types.keys())==1:
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


def get_text_mask(image, mask):
    reconstruct = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)
    dif = (reconstruct - image).astype('float32') ** 2
    dif = dif.max(axis=-1)
    text_mask = np.logical_and(dif > 3000, dif < 40000)
    #print(dif.max(), text_mask.sum(), mask.sum())
    m = (mask*text_mask).sum()/mask.sum()
    return text_mask, m


def remove_text(image, mask, iou_thresh=0.2):
    text_mask, m = get_text_mask(image, mask)
    if m < iou_thresh:
        return image, text_mask, False
    else:
        return image, text_mask, True


train_augs_geometric = alb.Compose([
    # alb.ShiftScaleRotate(),
    # alb.OneOf([
    #     alb.ElasticTransform(),
    #     alb.OpticalDistortion(),
    # ]),
])

train_augs_values = alb.Compose([
    alb.RandomBrightnessContrast(0.4, 0.4),
    #alb.RandomGamma(),
    alb.ChannelShuffle(),
    alb.RandomShadow(),
    alb.RandomSunFlare(),
    alb.CoarseDropout(),
    alb.OneOf([
        alb.GlassBlur(),
        alb.MotionBlur(),
        alb.Blur(),
    ]),
    alb.ImageCompression(),
])


def generate_get_mask(mode='train'):
    def get_mask(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        if mode == 'train':
            aug = train_augs_geometric(image=im, mask=mask)
            im, mask = aug['image'], aug['mask']
        mask = mask[:, :, 2]
        dif_mask = np.zeros(im.shape[:2])
        sum_mask = np.zeros(im.shape[:2])
        for n in np.unique(mask):
            if n > 0:
                cur_mask = (mask == n).astype('uint8')
                im, text_mask, is_correct = remove_text(im, cur_mask)
                if is_correct or mode != 'train':
                    dif_mask = np.logical_or(dif_mask, text_mask)
                    sum_mask = np.logical_or(sum_mask, cur_mask)

        if mode == 'train':
            im = train_augs_values(image=im.astype('uint8'))['image']
        im = im.astype('float32')

        im[:, :, 0] = im[:, :, 0] * sum_mask
        im[:, :, 1] = im[:, :, 1] * sum_mask
        im[:, :, 2] = im[:, :, 2] * sum_mask
        dif_mask = torch.tensor(dif_mask.astype('float32')).unsqueeze(0)

        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[MASK] = dif_mask
        return res_dict
    return get_mask


def generate_get_mask_v2(mode='train'):
    def get_mask(row, res_dict):
        im = imread_padding(row[IMAGE], 64, 0).astype('uint8')
        mask = imread_padding(row[MASK], 64, 0).astype('uint8')
        if mode == 'train':
            aug = train_augs_geometric(image=im, mask=mask)
            im, mask = aug['image'], aug['mask']
        mask = mask[:, :, 2]
        dif_mask = np.zeros(im.shape[:2])
        sum_mask = np.zeros(im.shape[:2])
        for n in np.unique(mask):
            if n > 0:
                cur_mask = (mask == n).astype('uint8')
                im, text_mask, is_correct = remove_text(im, cur_mask)
                if is_correct or mode != 'train':
                    dif_mask = np.logical_or(dif_mask, text_mask)
                    sum_mask = np.logical_or(sum_mask, cur_mask)

        if mode == 'train':
            im = train_augs_values(image=im.astype('uint8'))['image']
        im = im.astype('float32')

        dif_mask = torch.tensor(dif_mask.astype('float32')).unsqueeze(0)
        sum_mask = torch.tensor(sum_mask.astype('float32')).unsqueeze(0)
        all_mask = torch.tensor((mask > 0).astype('float32')).unsqueeze(0)

        res_dict[IMAGE] = ToTensor()((im / 255).astype('float32'))
        res_dict[MASK] = dif_mask
        res_dict[MASK+'sum'] = sum_mask
        res_dict[MASK + 'all'] = all_mask
        return res_dict
    return get_mask


def inference(model, im, device):
    t = ToTensor()((im / 255).astype('float32')).unsqueeze(dim=0).to(device)
    with torch.no_grad():
        out = model(t)
    return out[0].detach().cpu().numpy()[0]


def cut_im(data):
    im = data[IMAGE]
    with torch.no_grad():
        out = model(im)
    cut_mask = torch.logical_or(out > 0.5, data[MASK+'all'])
    cut_mask = torch.logical_or(torch.logical_not(cut_mask), data[MASK+'sum'])
    return im*cut_mask


model = torch.load('/data/checkpoints/[SARA]DocFake/segmentation_fields_more_aug/seg_model_model.pt').model


train_df, val_df = train_test_split(df, test_size=0.25, random_state=42)

train_dataset = BaseDataset(
    df=train_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_mask('train'),
    ],
)
val_dataset = BaseDataset(
    df=val_df,
    callbacks=[
        #generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_get_mask('val'),
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

pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={SEG_MODEL: ModelWrapper(
        model=model,
        keys=[
            ([IMAGE], [SEG_MODEL]),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
        # processing={
        #     IMAGE: cut_im,
        # },
    )},
    losses=[
        DataWrapper(
            BCEDiceLoss(activation='none'),
            keys=[
                ([SEG_MODEL, MASK], ['loss'])
            ],
        ),
    ],
    metrics=[
        DataWrapper(
            iou,
            keys=[
                ([SEG_MODEL, MASK], ['iou'])
            ],
        ),
    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=0,
        log_each_epoch=True,
        log_keys={
            'concat_image': ConcatImage(
                log_type=EnumLogDataTypes.image,
                keys=[SEG_MODEL, MASK],
                dim=2,
            ),
            'loss': EnumLogDataTypes.loss,
            'iou': EnumLogDataTypes.metric,
            IMAGE: ProcessImage(
                log_type=EnumLogDataTypes.image,
                #process_func=cut_im,
                process_func=lambda x: x[IMAGE],
            )
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
