import torch


class VectorsFromMask(torch.nn.Module):
    def __init__(self, skip_zero=True):
        super(VectorsFromMask, self).__init__()
        self.skip_zero = skip_zero

    def forward(self, encoded, masks):
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in range(masks.shape[1]):
            if i == 0 and self.skip_zero:
                continue
            cur_mask = masks[:, i, :, :].to(dtype=torch.bool)
            vector_encoded = []
            #print(i, cur_mask.sum())
            for i, cm in enumerate(cur_mask):
                v_encoded = encoded[:, i, cm]
                v_encoded = v_encoded.max(dim=1)[0]
                vector_encoded.append(v_encoded.unsqueeze(0))
            vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
            vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1)  # (b,n,f)
        vectors = vectors.transpose(1, 2).unsqueeze(dim=3)
        return vectors


class VectorsFromMaskV2(torch.nn.Module):
    def __init__(self):
        super(VectorsFromMaskV2, self).__init__()

    def forward(self, encoded, masks):
        vectors = []
        #encoded = torch.transpose(encoded, 0, 1)
        for b in range(masks.shape[0]):
            v_masks = torch.unique(masks[b])
            batch_vectors = []
            for v in v_masks:
                if v != 0:
                    cur_mask = (masks[b, 0, :, :] == v).to(dtype=torch.bool)
                    #print(encoded.shape, cur_mask.shape)
                    v_encoded = encoded[b, :, cur_mask]
                    v_encoded = v_encoded.max(dim=1)[0]
                    batch_vectors.append(v_encoded.unsqueeze(0))
            batch_vectors = torch.cat(batch_vectors, dim=0)
            vectors.append(batch_vectors.unsqueeze(0))
        vectors = torch.cat(vectors, dim=0)  # (b,n,f)
        print(vectors.shape)
        vectors = vectors.transpose(1, 2).unsqueeze(dim=3)
        return vectors


class VectorsFromBbox(torch.nn.Module):
    def __init__(self, merge_type='max'):
        super(VectorsFromBbox, self).__init__()
        self.merge_type = merge_type

    def forward(self, encoded, bboxes):
        vectors = []
        encoded = torch.transpose(encoded, 0, 1)
        for i in range(bboxes.shape[1]):
            cur_bboxes = bboxes[:, i].to(dtype=torch.long)
            #print(cur_bboxes)
            vector_encoded = []
            for i, bbox in enumerate(cur_bboxes):
                v_encoded = encoded[:, i, bbox[1]:bbox[1]+bbox[3], bbox[0]:bbox[0]+bbox[2]]
                #print(bbox)
                #print(v_encoded.shape)
                if self.merge_type == 'max':
                    v_encoded = v_encoded.max(dim=1)[0].max(dim=1)[0]
                if self.merge_type == 'mean':
                    v_encoded = v_encoded.mean(dim=1).mean(dim=1)
                if self.merge_type == 'matmul':
                    v_encoded = v_encoded.reshape(v_encoded.shape[0], -1)
                    v_encoded = torch.matmul(v_encoded, torch.transpose(v_encoded, 0, 1))
                    #print(v_encoded.shape)
                    v_encoded = v_encoded.reshape(-1)
                vector_encoded.append(v_encoded.unsqueeze(0))
            vector_encoded = torch.cat(vector_encoded, dim=0).unsqueeze(1)
            vectors.append(vector_encoded)
        vectors = torch.cat(vectors, dim=1)  # (b,n,f)
        vectors = vectors.transpose(1, 2).unsqueeze(dim=3)
        return vectors

