import pandas as pd
from torch.utils.data import DataLoader
from torch.nn import CTCLoss
import os
import torch
from sklearn.model_selection import train_test_split


from cftcvpipeline.pipelines.base_pipeline import BasePipeline

from cftcvpipeline.utils.logging.trains.logger import TrainsLogger
from cftcvpipeline.utils.saver import Saver, MetricStrategies
from cftcvpipeline.utils.data.dataset import BaseDataset
from cftcvpipeline.utils.data.callbacks.base import generate_imread_callback
from cftcvpipeline.utils.data.callbacks.ocr import generate_text_callback

from cftcvpipeline.constants.ocr import (
    OCR_MODEL,
    OCR_TEXT,
    OCR_LENGTH,
    OCR_SEQUENCE_LENGTH,
    OCR_SEQUENCE,
)
from cftcvpipeline.constants.base import IMAGE

from cftcvpipeline.models.ocr.segocr import SegOcrModel
from cftcvpipeline.strategies.base import base_strategy
from cftcvpipeline.metrics.ocr import str_match
from cftcvpipeline.utils.data.dict import OcrDict

from cftcvpipeline.utils.wrappers import ModelWrapper, StrategyWrapper, DataWrapper
from cftcvpipeline.utils.logging.base import EnumLogDataTypes

os.environ['TORCH_HOME'] = '/home/ds'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['NO_PROXY'] = 'koala03.ftc.ru'
os.environ['no_proxy'] = 'koala03.ftc.ru'


df = pd.read_csv('/data/KAGGLE/BMS/train_labels.csv')
df = df.dropna(subset=['InChI'])

SHAPE = (256, 448)
RNN_SIZE = 32
PROJECT_NAME = '[KAGGLE]BMS'
TASK_NAME = 'test_cuda'


def get_image_path(image_id):
    p1 = image_id[0]
    p2 = image_id[1]
    p3 = image_id[2]
    return os.path.join('/data/KAGGLE/BMS/train', p1, p2, p3, image_id) + '.png'


def convert_ocr(x: torch.Tensor):
    #print(x.shape)
    x = x.permute(1, 2, 0)
    #print(x.shape)
    return x


df[IMAGE] = df['image_id'].apply(get_image_path)
df[OCR_TEXT] = df['InChI'].str.split('=').str[1].str.split('/').str[1]
df = df.groupby(OCR_TEXT).sample(1)
ocr_dict = OcrDict(list(df[OCR_TEXT].values))
#df = df[:1000]
#print(ocr_dict.max_len)
ocr_dict.max_len = RNN_SIZE
train_df, val_df = train_test_split(df, test_size=0.25, random_state=2021)

def one_hot_encode(row, res_dict):
    cd = res_dict[OCR_SEQUENCE]
    d = torch.zeros((cd.shape[0], ocr_dict.count_letters))
    for i in range(cd.shape[0]):
        d[i, cd[i]] = 1
    res_dict[OCR_SEQUENCE] = d
    return res_dict


train_dataset = BaseDataset(
    df=df[:-len(df)//4],
    callbacks=[
        generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_text_callback(text_key=OCR_TEXT, ocr_dict=ocr_dict, seq_length=RNN_SIZE),
        #one_hot_encode,
    ],
)
val_dataset = BaseDataset(
    df=df[-len(df)//4:],
    callbacks=[
        generate_imread_callback(image_key=IMAGE, shape=SHAPE),
        generate_text_callback(text_key=OCR_TEXT, ocr_dict=ocr_dict, seq_length=RNN_SIZE),
        #one_hot_encode,
    ],
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=True,
)
val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=16,
    num_workers=4,
    shuffle=False,
)

pipeline = BasePipeline(
    train_loader=train_loader,
    val_loader=val_loader,
    models={OCR_MODEL: ModelWrapper(
        model=SegOcrModel(
            backbone='efficientnet-b0',
            input_size=SHAPE,
            rnn_size=RNN_SIZE,
            letters_count=ocr_dict.count_letters,
        ),
        keys=[
            ([IMAGE], [OCR_MODEL]),
        ],
        optimizer=torch.optim.Adam,
        optimizer_kwargs={
            'lr': 1e-4,
        },
    )},
    losses=[
        # DataWrapper(
        #     CTCLoss(blank=0, zero_infinity=False, reduction='none'),
        #     keys=[
        #         ([OCR_MODEL, OCR_SEQUENCE, OCR_SEQUENCE_LENGTH, OCR_LENGTH], ['ctc'])
        #     ],
        # ),
        DataWrapper(
            torch.nn.CrossEntropyLoss(reduction='none'),
            keys=[
                ([OCR_MODEL, OCR_SEQUENCE], ['bce'])
            ],
            processing={OCR_MODEL: convert_ocr},
        ),
    ],
    metrics=[
        DataWrapper(
            str_match,
            keys=[
                ([OCR_MODEL, OCR_SEQUENCE], ['str_match']),
            ],
        ),
    ],
    logger=TrainsLogger(
        config={},
        log_each_ind=30,
        log_each_epoch=True,
        log_keys={
            IMAGE: EnumLogDataTypes.image,
            'str_match': EnumLogDataTypes.metric,
            'bce': EnumLogDataTypes.loss,
        },
        project_name=PROJECT_NAME,
        task_name=TASK_NAME,
    ),
    saver=Saver(
        save_dir='/data/checkpoints',
        project_name=PROJECT_NAME,
        experiment_name=TASK_NAME,
        metric_name='bce',
        metric_strategy=MetricStrategies.low,
    ),
    device='cuda:2',
    n_epochs=100,
    loss_reduce=False,
)
print('pipeline run')
pipeline.run()
