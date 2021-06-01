import pandas as pd
import typing as tp


class BaseDataset:
    def __init__(
            self,
            df: pd.DataFrame,
            callbacks: tp.List[tp.Callable],
    ):
        self.df = df
        self.callbacks = callbacks

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        res_data = {}
        cur_row = self.df.iloc[idx]
        for callback in self.callbacks:
            res_data = callback(cur_row, res_data)
        return res_data
