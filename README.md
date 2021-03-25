# ACSConv

Reinventing 2D Convolutions for 3D Images ([arXiv](https://arxiv.org/abs/1911.10477))

IEEE Journal of Biomedical and Health Informatics (IEEE JBHI), 2021 ([DOI](http://doi.org/10.1109/JBHI.2021.3049452))

## Key contributions

* ACS convolution aims at a **plug-and-play replacement** of standard 3D convolution, for 3D medical images.
* ACS convolution enables **2D-to-3D transfer learning**, which consistently provides significant performance boost in our experiments.
* Even without pretraining, ACS convolution is **comparable to or even better than** 3D convolution, with **smaller model size** and **less computation**.

## Requirements

### Base requirements

The bare minimum to run the ACSConv package.

```python
torch>=1.8.1
torchvision>=0.9.0
```

You can install them either manually or through the command:

``` bash
pip install -r requirements.txt
```

### Experimental requirements

All libraries needed to run the included experiments (base requirements included).

```python
fire==0.4.0
jupyterlab>=3.0.12
matplotlib==3.3.4
pandas==1.1.3
torch==1.8.0
torchvision==0.9.0
tqdm==4.59.0
scikit-image==0.17.2
scikit-learn==0.24.1
scipy==1.5.2
tensorboardx==2.1
```

You can install them either manually or through the command:

``` bash
pip install -r experimental_requirements.txt
```

## Package Installation

If you want to use this class, you have two options:

A) Simply copy and paste it in your project;

B) Or install it through `pip` following the command bellow:

``` bash
pip install git+git://github.com/M3DV/ACSConv.git#egg=ACSConv
```

> **Note 1**: As noted by [David Winterbottom](https://codeinthehole.com/tips/using-pip-and-requirementstxt-to-install-from-the-head-of-a-github-branch/), if you freeze the environment to export the dependencies, note that this will add the specific commit to your requirements, so it might be a good idea to delete the commit ID from it.
> ___
> **Note 2**: Due to the simplicity of this "package", this installation method was preferred over the more traditional [PyPI](https://pypi.org/).

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
from acsconv.operators import ACSConv, SoftACSConv
x = torch.rand(batch_size, 3, D, H, W)
# ACSConv to process 3D volumnes
conv = ACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)
# SoftACSConv to process 3D volumnes
conv = SoftACSConv(in_channels=3, out_channels=10, kernel_size=3, padding=1)
out = conv(x)
```

## Usage of native ACS models

```python
from acsconv.models.acsunet import ACSUnet
unet_3d = ACSUnet(num_classes=3)
B, C_in, D, H, W = (1, 3, 64, 64, 64)
input_3d = torch.rand(B, C_in, D, H, W)
output_3d = unet_3d(input_3d)
```

## How to run the experiments

* [Proof-of-Concept Segmentation](./experiments/poc/README.md)
* [Lung Nodule Classification and Segmentation](./experiments/lidc/README.md)
* ...
