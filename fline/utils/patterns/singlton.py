def to_singlton_class(cls):
    class SingltonWrapper(cls):
        def __new__(cls, cls_instance=None, *args, **kwargs):
            if not hasattr(cls, 'instance') and cls_instance is not None:
                cls.instance = cls_instance
            return cls.instance
    return SingltonWrapper
