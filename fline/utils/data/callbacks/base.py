from torchvision.transforms import ToTensor

from fline.utils.data.image.io import imread_resize, imread_padding


def generate_imread_callback(image_key, shape):
    def _imread_callback(df_row, res_dict):
        res_dict[image_key] = ToTensor()((imread_resize(df_row[image_key], shape)/255).astype('float32'))
        return res_dict
    return _imread_callback


def generate_imread_padding_callback(image_key, q, fill_value):
    def _imread_callback(df_row, res_dict):
        res_dict[image_key] = ToTensor()((imread_padding(df_row[image_key], q, fill_value)/255).astype('float32'))
        return res_dict
    return _imread_callback
