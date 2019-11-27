import torch.nn as nn
from .base_converter import BaseConverter
from ..operators import ACSConv

class ACSConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding acs version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import ACSConverter
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = ACSConverter(m)
        >>> # after converted, m is using ACSConv and capable of dealing with 3D data
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model']
    def __init__(self, model):
        preserve_state_dict = model.state_dict()
        model = self.convert_module(model)
        model.load_state_dict(preserve_state_dict,strict=False) # 
        self.model = model

    def convert_module(self, module):
        for child_name, child in module.named_children(): 
            if isinstance(child, nn.Conv2d):
                arguments = nn.Conv2d.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['bias'] = True if kwargs['bias'] is not None else False
                setattr(module, child_name, ACSConv(**kwargs))

            elif hasattr(nn, child.__class__.__name__) and \
                ('pool' in child.__class__.__name__.lower() or 
                'norm' in child.__class__.__name__.lower()):
                if hasattr(nn, child.__class__.__name__.replace('2d', '3d')):
                    TargetClass = getattr(nn, child.__class__.__name__.replace('2d', '3d'))
                    arguments = TargetClass.__init__.__code__.co_varnames[1:]
                    kwargs = {k: getattr(child, k) for k in arguments}
                    setattr(module, child_name, TargetClass(**kwargs))
                else:
                    raise Exception('No corresponding module in 3D for 2d module {}'.format(child.__class__.__name__))
            elif isinstance(child, nn.Upsample):
                arguments = nn.Upsample.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['mode'] = 'trilinear' if kwargs['mode']=='bilinear' else kwargs['mode']
                setattr(module, child_name, nn.Upsample(**kwargs))
            else:
                self.convert_module(child)
        return module
