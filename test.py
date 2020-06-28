import torch
from acsconv.operators import ACSConv

conv = ACSConv(3, 10, 3, 1, 1).cuda()
x = torch.rand(2, 3, 8, 8, 8).cuda()
out = conv(x)
print(x.shape, out.shape)
out.sum().backward()
print(conv.weight.grad)