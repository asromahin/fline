class BaseGroup:
    def __init__(self, log_type):
        self.log_type = log_type

    def __call__(self, data):
        return data
