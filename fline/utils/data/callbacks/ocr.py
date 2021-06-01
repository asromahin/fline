import torch

from fline.utils.data.dict import OcrDict
from fline.constants.ocr import OCR_TEXT, OCR_SEQUENCE, OCR_LENGTH, OCR_SEQUENCE_LENGTH


def generate_text_callback(text_key, ocr_dict: OcrDict, seq_length):
    def _text_callback(df_row, res_dict):
        res_dict[OCR_TEXT] = df_row[text_key]
        res_dict[OCR_SEQUENCE] = ocr_dict.text_to_code(res_dict[OCR_TEXT])
        #print('-'*50)
        #print(ocr_dict.max_len)
        #print(len(res_dict[OCR_SEQUENCE]))
        res_dict[OCR_SEQUENCE] = torch.tensor(ocr_dict.fill_code(res_dict[OCR_SEQUENCE]))
        #print(len(res_dict[OCR_SEQUENCE]))
        res_dict[OCR_LENGTH] = len(res_dict[OCR_TEXT])
        res_dict[OCR_SEQUENCE_LENGTH] = seq_length
        return res_dict
    return _text_callback
