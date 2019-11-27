import torch.nn as nn
from .base_converter import BaseConverter
from ..operators import Conv2_5d
from ..utils import _pair_same

class Conv2_5dConverter(BaseConverter):
    """
    Decorator class for converting 2d convolution modules
    to corresponding 3d version in any networks.
    
    Args:
        model (torch.nn.module): model that needs to be converted
    Warnings:
        Functions in torch.nn.functional involved in data dimension are not supported
    Examples:
        >>> import Conv2_5DWrapper
        >>> import torchvision
        >>> # m is a standard pytorch model
        >>> m = torchvision.models.resnet18(True)
        >>> m = Conv2_5DWrapper(m)
        >>> # after converted, m is using ACSConv and capable of dealing with 3D data
        >>> x = torch.rand(batch_size, in_channels, D, H, W)
        >>> out = m(x)
    """
    converter_attributes = ['model', 'unsqueeze_axis']
    def __init__(self, model, unsqueeze_axis=-3):
        preserve_state_dict = model.state_dict()
        self.model = model
        self.unsqueeze_axis = unsqueeze_axis
        self.model = self.convert_module(self.model)
        self.load_state_dict(preserve_state_dict, strict=True)
        

    def convert_module(self, module):
        for child_name, child in module.named_children(): 
            if isinstance(child, nn.Conv2d):
                arguments = nn.Conv2d.__init__.__code__.co_varnames[1:]
                kwargs = {k: getattr(child, k) for k in arguments}
                kwargs['bias'] = True if kwargs['bias'] is not None else False
                for k in ['kernel_size','stride','padding','dilation']:
                    kwargs[k] = _pair_same(kwargs[k])[0]
                kwargs['unsqueeze_axis'] = self.unsqueeze_axis
                setattr(module, child_name, Conv2_5d(**kwargs))

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

    def load_state_dict(self, state_dict, strict=True, unsqueeze_axis=-3):
        load_state_dict_from_2d_to_2_5d(self.model, state_dict, strict=strict, unsqueeze_axis=unsqueeze_axis)

def load_state_dict_from_2d_to_2_5d(model_2_5d, state_dict_2d, strict=True, unsqueeze_axis=-3):
    for key in list(state_dict_2d.keys()):
        if state_dict_2d[key].dim()==4:
            state_dict_2d[key] = state_dict_2d[key].unsqueeze(unsqueeze_axis)
    model_2_5d.load_state_dict(state_dict_2d, strict=strict)