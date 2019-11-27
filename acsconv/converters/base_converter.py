import torch.nn as nn

class BaseConverter(object):
    """
    base class for converters
    """
    converter_attributes = []
    def __init__(self, model):
        pass

    def convert_module(self):
        raise NotImplementedError

    def __getattr__(self, attr):
        return getattr(self.model, attr)
        
    def __setattr__(self, name, value):
        if name in self.__class__.converter_attributes:
            return object.__setattr__(self, name, value)
        else:
            return setattr(self.model, name, value)

    def __call__(self, x):
        return self.model(x)

    def __repr__(self):
        return self.__class__.__name__ + '(\n' + self.model.__repr__() + '\n)'
