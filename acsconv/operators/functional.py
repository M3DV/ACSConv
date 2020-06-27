import torch
import torch.nn.functional as F
import math
import os


from torch.utils.cpp_extension import load

abspath = os.path.abspath(__file__)
dname = os.path.dirname(abspath)
backward_cpp = load(name='backward_cpp', sources=[os.path.join(dname, 'backward_wrap.cpp')])


def conv3D_output_shape_f(i, input_shape, kernel_size, dilation, padding, stride):
    """
    Calculate the original output size assuming the convolution is nn.Conv3d based on 
    input size, kernel size, dilation, padding and stride.
    """
    return math.floor((input_shape[i]-kernel_size[i]-(dilation[i]-1)*
                                        (kernel_size[i]-1)+2*padding[i])
                                    /stride[i])+1
    
class ACSConvOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, acs_kernel_split):

        # ctx = ...
        ctx.save_for_backward(x, weight, bias)
        ctx.stride = stride
        ctx.padding = padding 
        ctx.dilation = dilation
        ctx.groups = groups
        ctx.acs_kernel_split = acs_kernel_split

        B, C_in, *input_shape = x.shape
        conv3D_output_shape = (conv3D_output_shape_f(0, input_shape, kernel_size, dilation, padding, stride), 
                                conv3D_output_shape_f(1, input_shape, kernel_size, dilation, padding, stride), 
                                conv3D_output_shape_f(2, input_shape, kernel_size, dilation, padding, stride))
                
        weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
        weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
        weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)
        f_out = []
        a_x = c_x = s_x = None
        if acs_kernel_split[0]>0:
            a_x = x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                                kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                                :,:]
            a = F.conv3d(a_x, 
                                                weight=weight_a, bias=None, 
                                                stride=stride,
                                                padding=(0,padding[1],padding[2]),
                                                dilation=dilation,
                                                groups=groups)                
            f_out.append(a)
        if acs_kernel_split[1]>0:
            c_x = x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                                kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                                :]
            c = F.conv3d(c_x, 
                                                weight=weight_c, bias=None,                                     
                                                stride=stride,
                                                padding=(padding[0],0,padding[2]),
                                                dilation=dilation,
                                                groups=groups)
            f_out.append(c)
        if acs_kernel_split[2]>0:
            s_x = x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                                kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                                ]
            s = F.conv3d(s_x, 
                                                weight=weight_s, 
                                                bias=None, 
                                                stride=stride,
                                                padding=(padding[0],padding[1],0),
                                                dilation=dilation,
                                                groups=groups)
            f_out.append(s)
        f = torch.cat(f_out, dim=1)
        if bias is not None:
            f += bias.view(1,out_channels,1,1,1)
        ctx.conv3D_output_shape = conv3D_output_shape
        return f


    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        stride = ctx.stride
        padding = ctx.padding 
        dilation = ctx.dilation
        groups = ctx.groups
        acs_kernel_split = ctx.acs_kernel_split
        conv3D_output_shape = ctx.conv3D_output_shape
        
        grad_input = grad_weight = grad_bias = None

        B, C_in, *input_shape = x.shape
        a_x = x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                                kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                                :,:]
        c_x = x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                                kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                                :]
        s_x = x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                                kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                                ]

        weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
        weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
        weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)

        grad_output_a = grad_output[:, 0:acs_kernel_split[0]]
        grad_output_c = grad_output[:, acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])]
        grad_output_s = grad_output[:, (acs_kernel_split[0]+acs_kernel_split[1]):]

        if ctx.needs_input_grad[0]:
            if a_x is not None:
                grad_input_a = backward_cpp.backward_input(
                    a_x.shape, weight_a, grad_output_a, stride, (0,padding[1],padding[2]), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                ) 
            if c_x is not None:
                grad_input_a = backward_cpp.backward_input(
                    c_x.shape, weight_c, grad_output_c, stride, (padding[0],0,padding[2]), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                ) 
            if s_x is not None:
                grad_input_a = backward_cpp.backward_input(
                    s_x.shape, weight_s, grad_output_s, stride, (padding[0],padding[1],0), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                )    
            grad_input = grad_input_a + grad_input_c + grad_input_s
        if ctx.needs_input_grad[1]:
            grad_weight = backward_cpp.backward_weight(
                x, weight.shape, grad_output, stride, padding, dilation, groups,
                torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
            )
            grad_weight_list = []
            if a_x is not None:
                grad_weight_a = backward_cpp.backward_weight(
                    a_x, weight_a.shape, grad_output_a, stride, (0,padding[1],padding[2]), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                )
                grad_weight_list.append(grad_weight_a)
            if c_x is not None:
                grad_weight_a = backward_cpp.backward_weight(
                    c_x, weight_c.shape, grad_output_c, stride, (padding[0],0,padding[2]), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                ) 
                grad_weight_list.append(grad_weight_c)
            if s_x is not None:
                grad_weight_a = backward_cpp.backward_weight(
                    s_x, weight_s.shape, grad_output_s, stride, (padding[0],padding[1],0), dilation, groups,
                    torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
                )    
                grad_weight_list.append(grad_weight_s)
            grad_weight = torch.cat(grad_weight_list, dim=0)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = backward_cpp.backward_bias(grad_output)

        return grad_input, grad_weight, grad_bias, None, None, None, None, None, None, None

acs_conv_f = ACSConvOp.apply

'''z
def acs_conv_f(x, weight, bias, kernel_size, dilation, padding, stride, groups, out_channels, acs_kernel_split):
    B, C_in, *input_shape = x.shape
    conv3D_output_shape = (conv3D_output_shape_f(0, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(1, input_shape, kernel_size, dilation, padding, stride), 
                            conv3D_output_shape_f(2, input_shape, kernel_size, dilation, padding, stride))
            
    weight_a = weight[0:acs_kernel_split[0]].unsqueeze(2)
    weight_c = weight[acs_kernel_split[0]:(acs_kernel_split[0]+acs_kernel_split[1])].unsqueeze(3)
    weight_s = weight[(acs_kernel_split[0]+acs_kernel_split[1]):].unsqueeze(4)
    f_out = []
    if acs_kernel_split[0]>0:
        a = F.conv3d(x if conv3D_output_shape[0]==input_shape[0] or 2*conv3D_output_shape[0]==input_shape[0] else F.pad(x, (0,0,0,0,padding[0],padding[0]),'constant',0)[:,:,
                                            kernel_size[0]//2:kernel_size[0]//2+(conv3D_output_shape[0]-1)*stride[0]+1,
                                            :,:], 
                                            weight=weight_a, bias=None, 
                                            stride=stride,
                                            padding=(0,padding[1],padding[2]),
                                            dilation=dilation,
                                            groups=groups)                
        f_out.append(a)
    if acs_kernel_split[1]>0:
        c = F.conv3d(x if conv3D_output_shape[1]==input_shape[1] or 2*conv3D_output_shape[1]==input_shape[1] else F.pad(x, (0,0,padding[1],padding[1]),'constant',0)[:,:,:,
                                            kernel_size[1]//2:kernel_size[1]//2+stride[1]*(conv3D_output_shape[1]-1)+1,
                                            :], 
                                            weight=weight_c, bias=None,                                     
                                            stride=stride,
                                            padding=(padding[0],0,padding[2]),
                                            dilation=dilation,
                                            groups=groups)
        f_out.append(c)
    if acs_kernel_split[2]>0:
        s = F.conv3d(x if conv3D_output_shape[2]==input_shape[2] or 2*conv3D_output_shape[2]==input_shape[2] else F.pad(x, (padding[2],padding[2]),'constant',0)[:,:,:,:,
                                            kernel_size[2]//2:kernel_size[2]//2+stride[2]*(conv3D_output_shape[2]-1)+1
                                            ], 
                                            weight=weight_s, 
                                            bias=None, 
                                            stride=stride,
                                            padding=(padding[0],padding[1],0),
                                            dilation=dilation,
                                            groups=groups)
        f_out.append(s)
    f = torch.cat(f_out, dim=1)
    if bias is not None:
        f += bias.view(1,out_channels,1,1,1)
    return f
'''