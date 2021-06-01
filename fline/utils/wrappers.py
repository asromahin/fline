import typing as tp
import torch


class DataWrapper:
    def __init__(
            self,
            callable_obj: tp.Callable = None,
            keys: tp.List[tp.Tuple[tp.List[str], tp.List[str]]] = None,
            processing: tp.Mapping[str, tp.Callable] = {},
    ):
        self.callable_obj = callable_obj
        self.keys = keys
        self.processing = processing

    def __call__(self, data: tp.Mapping, *args, **kwargs):
        #print(data.keys())
        for key_pairs in self.keys:
            in_keys, out_keys = key_pairs
            prepare_input = [
                data[k]
                if k not in self.processing.keys()
                else self.processing[k](data)
                for k in in_keys
            ]
            res = self.callable_obj(*prepare_input)
            if len(out_keys) == 1:
                out_key = out_keys[0]
                if out_key in self.processing.keys():
                    res = self.processing[out_key](res)
                data[out_key] = res
            else:
                for i, out_key in enumerate(out_keys):
                    if out_key in self.processing.keys():
                        res[i] = self.processing[out_key](res[i])
                    data[out_key] = res[i]
        return data


class ModelWrapper(DataWrapper):
    def __init__(
            self,
            model: tp.Callable,
            keys: tp.List[tp.Tuple[tp.List[str], tp.List[str]]] = None,
            processing: tp.Mapping[str, tp.Callable] = {},
            optimizer: tp.Callable = None,
            optimizer_kwargs: dict = {},
    ):
        super(ModelWrapper, self).__init__(
            model,
            keys,
            processing,
        )
        self.model = model
        self.optimizer = optimizer(self.model.parameters(), **optimizer_kwargs)



