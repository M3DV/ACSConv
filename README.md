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

## How to run the proof-of-concept experiment

```bash
cd poc_experiments
```

1. Generate the proof-of-concept dataset (2D and 3D)
```python
python generate_poc_data.py
```
2. Train the 2D UNet on 2D dataset
```python
python train_poc_shape.py
```
3. Locate where the 2D model checkpoint is saved (in ```./tmp/poc/noise0.5/shape/.../model.dat``` and then copy the path to ```POCVoxelEnv.shape_checkpoint``` in ```poc_config.py```

4. Train the 3D UNet on 3D dataset, with or without 2D pretraining
```python
python train_poc_voxel.py
```

The default 3D model is ACSUNet **p.**. To change ACSConv to Conv3d / Conv2_5d or random initialization, modify ```POCVoxelConfig.conv``` and ```POCVoxelConfig.pretrained``` in ```poc_config.py```.
