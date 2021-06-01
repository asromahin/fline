import cv2
import numpy as np


SHAPE_ATTRIBUTES_KEYS = [
    'name',
    'cx',
    'cy',
    'all_points_x',
    'all_points_y',
    'x',
    'y',
    'width',
    'height',
]

"""
def read_annot(annot_path):
    with(open(annot_path, 'r')) as f:
        return json.loads(f.read())


def extract_regions(annot, region_attributes_keys=[]):
    res = []
    for region in annot['regions']:
        r = {
            'original_width': annot['metadata']['width'],
            'original_height': annot['metadata']['height'],
        }
        for shape_attr in SHAPE_ATTRIBUTES_KEYS:
            r[shape_attr] = region['shape_attributes'].get(shape_attr)
        for region_attr in region_attributes_keys:
            r[region_attr] = region['region_attributes'].get(region_attr)
        res.append(r)
    return res


def extract_regions_from_path(annot_path, region_attributes_keys=[]):
    return extract_regions(read_annot(annot_path), region_attributes_keys=region_attributes_keys)


def extract_regions_from_df(df, annotation_key, region_attributes_keys=[]):
    res = []
    for annotation_path in df[annotation_key]:
        res += extract_regions_from_path(annotation_path, region_attributes_keys=region_attributes_keys)
    return res
"""

def get_mask(mask, df_row, channel=None, mask_val=1, get_mask_points=True, points_radius=7):
    if df_row['name'] == 'rect':
        x, y, width, height = df_row[['x', 'y', 'width', 'height']]
        if channel is None:
            mask[y:y+height, x:x+width] = mask_val
        else:
            mask[y:y + height, x:x + width, channel] = mask_val
    if df_row['name'] == 'polygon':
        all_points_x, all_points_y = df_row[['all_points_x', 'all_points_y']]
        p = np.zeros((len(all_points_x), 1, 2), dtype='int')
        p[:, 0, 0] = all_points_x
        p[:, 0, 1] = all_points_y
        if channel is None:
            mask = cv2.drawContours(mask, [p], -1, mask_val, -1)
        else:
            mask[:, :, channel] = cv2.drawContours(mask[:, :, channel], [p], -1, mask_val, -1)
    if (df_row['name'] == 'point') & get_mask_points:
        cx, cy = df_row[['cx', 'cy']]
        if channel is None:
            mask = cv2.circle(mask, (cx, cy), points_radius, mask_val, -1)
        else:
            mask[:, :, channel] = cv2.circle(mask[:, :, channel], (cx, cy), points_radius, mask_val, -1)
    return mask


