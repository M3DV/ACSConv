# ACSConv

Reinventing 2D Convolutions for 3D Images ([arXiv](https://arxiv.org/abs/1911.10477))

IEEE Journal of Biomedical and Health Informatics (IEEE JBHI), 2021 ([DOI](http://doi.org/10.1109/JBHI.2021.3049452))

**News**:
- 2022.01.26 - ACS [ConvNeXt](acsconv/models/convnext.py) supported.
- 2021.12.17 - torch 1.10 supported & pip installation supported.
- 2021.4.19 - torch 1.8 supported.

## Key contributions

* ACS convolution aims at a **plug-and-play replacement** of standard 3D convolution, for 3D medical images.
* ACS convolution enables **2D-to-3D transfer learning**, which consistently provides significant performance boost in our experiments.
* Even without pretraining, ACS convolution is **comparable to or even better than** 3D convolution, with **smaller model size** and **less computation**.


## Package Installation

If you want to use this class, you have two options:

A) Install ACSConv as a standard Python package from PyPI:

```bash
pip install ACSConv
```

B) Simply copy and paste it in your project;


You could run the `test.py` to validate the installation. (If you want to test the validity of pip installation, please move this `test.py` file outside of this git project directory, otherwise it is testing the code inside the project instead of pip installation.)

## Requirements
### PyTorch requirements

```python
torch>=1.0.0 and torch<=1.10.0
```

You can install it on the [official homepage](https://pytorch.org/docs/stable/index.html).

### Other requirements

All libraries needed to run the included experiments (base requirements included).

```python
fire
jupyterlab
matplotlib
pandas
tqdm
sklearn
tensorboardx
```

## Code structure

* ``acsconv``
  the core implementation of ACS convolution, including the operators, models, and 2D-to-3D/ACS model converters.
  * ``operators``: include ACSConv, SoftACSConv and Conv2_5d.
  * ``converters``: include converters which convert 2D models to 3d/ACS/Conv2_5d counterparts.
  * ``models``: Native ACS models.
* ``experiments``
  the scripts to run experiments.
  * ``mylib``: the lib for running the experiments.
  * ``poc``: the scripts to run proof-of-concept experiments.
  * ``lidc``: the scripts to run LIDC-IDRI experiments.

## Convert a 2D model into 3D with a single line of code

```python
import torch
from torchvision.models import resnet18
from acsconv.converters import ACSConverter
# model_2d is a standard pytorch 2D model
model_2d = resnet18(pretrained=True)
B, C_in, H, W = (1, 3, 64, 64)
input_2d = torch.rand(B, C_in, H, W)
output_2d = model_2d(input_2d)

model_3d = ACSConverter(model_2d)
# once converted, model_3d is using ACSConv and capable of processing 3D volumes.
B, C_in, D, H, W = (1, 3, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = model_3d(input_3d)
```

## Usage of ACS operators

```python
import torch
from acsconv.operators import ACSConv, SoftACSConv
B, C_in, D, H, W = (1, 3, 64, 64, 64)
x = torch.rand(B, C_in, D, H, W)
# ACSConv to process 3D volumnes
conv = ACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)
# SoftACSConv to process 3D volumnes
conv = SoftACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)
```

## Usage of native ACS models

```python
import torch
from acsconv.models.acsunet import ACSUNet
unet_3d = ACSUNet(num_classes=3)
B, C_in, D, H, W = (1, 1, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = unet_3d(input_3d)
```

## How to run the experiments

* [Proof-of-Concept Segmentation](./experiments/poc/README.md)
* [Lung Nodule Classification and Segmentation](./experiments/lidc/README.md)
* ...
