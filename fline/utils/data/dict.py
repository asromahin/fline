import torch


class OcrDict:
    def __init__(self, list_texts):
        self.list_texts = list_texts
        self.unique_texts = list(set(list_texts))
        self.unique_chars, self.code_to_char, self.char_to_code = self.get_dicts()
        self.count_letters = len(self.unique_chars)
        self.max_len = max([len(text) for text in list_texts])

    def get_dicts(self):
        unique_chars = list(set("".join(self.list_texts)))
        code_to_char = {i: char for i, char in enumerate(unique_chars)}
        char_to_code = {char: i for i, char in enumerate(unique_chars)}
        return unique_chars, code_to_char, char_to_code

    def text_to_code(self, text):
        codes = [self.char_to_code[char] for char in text]
        return codes

    def fill_code(self, codes):
        codes += (self.max_len - len(codes)) * [0]
        return codes

    def timeseries_to_code(self, timeseries):
        result_code = []
        last_code = None
        for code in timeseries:
            if code is not None:
                if code != last_code:
                    result_code.append(code)
            last_code = code
        return result_code

    def prediction_to_code(self, pred: torch.Tensor):
        pred = pred.permute(1, 0, 2)
        pred = torch.Tensor.argmax(pred, dim=2).detach().cpu().numpy()
        #print(pred.shape)
