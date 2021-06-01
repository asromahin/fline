import pandas as pd
from torch.utils.data import DataLoader
import os
import torch
import json
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity


from cftcvpipeline.pipelines.base_pipeline import BasePipeline

from cftcvpipeline.utils.logging.trains.logger import TrainsLogger
from cftcvpipeline.utils.saver import Saver, MetricStrategies
from cftcvpipeline.utils.data.dataset import BaseDataset
from cftcvpipeline.utils.data.callbacks.base import generate_imread_callback

from cftcvpipeline.constants.base import IMAGE

from cftcvpipeline.utils.wrappers import ModelWrapper, DataWrapper
from cftcvpipeline.utils.logging.base import EnumLogDataTypes

from cftcvpipeline.models.autoencoders.simple_ae import AutoEncoder

from cftcvpipeline.utils.logging.trains.groups import ConcatImage

os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


def get_original_image_for_df(df: pd.DataFrame):
    df['original_image'] = df['image'].str.split('/').str[-3]
    return df


def get_class_for_df(df: pd.DataFrame):
    cdf = pd.read_csv('/home/ds/Projects/SARA/datasets/classification/dataset_v75.csv')

    def extract_class(annot_path):
        with open(annot_path) as f:
            annot = json.loads(f.read())
            document_types = annot['file_attributes']['document_type']
            # print(document_types)
            if len(document_types.keys()) == 1:
                return list(document_types.keys())[0]

    cdf['class'] = cdf['annotation'].apply(extract_class)
    cdf = cdf.dropna(subset=['class'])
    cdf['original_image'] = cdf['image'].str.split('/').str[-1]
    #print(len(cdf), len(df))
    df = pd.merge(left=df, right=cdf, on='original_image')
    #print(len(df))
    return df


SHAPE = (512, 512)
PROECT_NAME = '[SARA]FieldsCompare'
TASK_NAME = 'test_ae_densenet121_encoder_with_circleloss_full_image'
AE_MODEL = 'ae_model'

DESCR_MODEl = 'descr_model'
LABELS = 'labels'

df = pd.read_csv('/home/ds/Projects/SARA/datasets/fields_segmentation/dataset_v93.csv')

df = get_original_image_for_df(df)
df = get_class_for_df(df)
df = df.dropna(subset=['class'])

df[IMAGE] = df['image_x']
df[LABELS] = df['class']
groups_dict = {cls: i for i, cls in enumerate(list(df['class'].unique()))}


def get_class(df_row, d):
    d[LABELS] = groups_dict[df_row[LABELS]]
    return d


class Flatten(torch.nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

#df = df[:1000]
train_dataset = BaseDataset(
    df=df[:-len(df)//4],
    callbacks=[
        generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        get_class,
    ],
)
val_dataset = BaseDataset(
    df=df[-len(df)//4:],
    callbacks=[
        generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        get_class,
    ],
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=4,
    num_workers=4,
    shuffle=False,
)

pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={AE_MODEL: ModelWrapper(
        model=AutoEncoder(
            features=[3, 64, 128, 256, 512, 1024]
        ),
        keys=[
            ([IMAGE], [AE_MODEL, AE_MODEL+'features']),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        DataWrapper(
            torch.nn.MSELoss(),
            keys=[
                ([AE_MODEL, IMAGE], ['mse'])
            ],
        ),
        DataWrapper(
            losses.CircleLoss(distance=CosineSimilarity()),
            keys=[
                ([AE_MODEL+'features', LABELS], ['circle_loss'])
            ],
            processing={AE_MODEL+'features': Flatten()},
        ),
    ],
    metrics=[

    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=0,
        log_each_epoch=True,
        log_keys={
            'concat_image': ConcatImage(
                log_type=EnumLogDataTypes.image,
                keys=[IMAGE, AE_MODEL],
                dim=2,
            ),
            'mse': EnumLogDataTypes.loss,
            'circle_loss': EnumLogDataTypes.loss,
        },
        project_name=PROECT_NAME,
        task_name=TASK_NAME,
    ),
    saver=Saver(
        save_dir='/data/checkpoints',
        project_name=PROECT_NAME,
        experiment_name=TASK_NAME,
        metric_name='circle_loss',
        metric_strategy=MetricStrategies.low,
    ),
    device='cuda:2',
    n_epochs=1000,
    loss_reduce=False,
)
print('pipeline run')
pipeline.run()
