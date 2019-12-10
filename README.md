# ACSConv

Reinventing 2D Convolutions for 3D Medical Images ([arXiv](https://arxiv.org/abs/1911.10477))

**[WIP] More code is coming soon (around mid December), stay tuned!**

* [ ] More models
* [ ] More experiments
* [ ] More document
* [ ] Memory-efficient implementation
* [ ] More pretrained models (ours / other open source projects)

## Key contributions

* ACS convolution aims at a **plug-and-play replacement** of standard 3D convolution, for 3D medical images.
* ACS convolution enables **2D-to-3D transfer learning**, which consistently provides significant performance boost in our experiments.
* Even without pretraining, ACS convolution is **comparable to or even better than** 3D convolution, with **smaller model size** and **less computation**.

## Code structure

* ``acsconv``: the core implementation of ACS convolution, including the operators, models, and 2D-to-3D/ACS model converters. 
* ``experiments``: the scripts to run experiments.
* ``experiments/mylib``: the lib for running the experiments.
* ``experiments/poc``: the scripts to run proof-of-concept experiments.
* ``experiments/lidc``: the scripts to run LIDC-IDRI experiments.

## Convert a 2D model into 3D with a single line of code

```python
from torchvision.models import resnet18
from acsconv.converters import ACSConverter
# model_2d is a standard pytorch 2D model
model_2d = resnet18(pretrained=True)
B, C_in, H, W = (1, 3, 64, 64)
input_2d = torch.rand(B, C_in, H, W)
output_2d = model_2d(input_2d)

model_3d = ACSConverter(model_2d)
# once converted, model_3d is using ACSConv and capable of dealing with 3D data.
B, C_in, D, H, W = (1, 3, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = model_3d(input_3d)
```

## How to run the experiments

* [Proof-of-Concept Segmentation](./experiments/poc/README.md)
* [Lung Nodule Classification and Segmentation](./experiments/lidc/README.md)
* ...
