
# How to run the LIDC-IDRI experiment

```bash
cd experiments/lidc/
```

1. Download and prepare data
   Download LIDC data from this link (*coming soon*) and unzip it.
   Create data directory and move the downloaded data into the directory

   ```bash
   mkdir ./data
   mv <unzipped data path> ./data
   ```

2. Run training scripts
    for segmentation, run

    ```bash
    python train_segmentation.py
    ```

    for classification, run

    ```bash
    python train_classification.py
    ```

The default model is ACS ResNet18 **p.**. To change ACSConv to Conv3d / Conv2_5d or random initialization, or to change to other backbones (DenseNet or VGG16), modify ```LIDCSegConfig``` and ```LIDCClassConfig``` in ```lidc_config.py```, as instructed by the comments aside. Note that for ResNet18, the pretraining methods of Conv3d include **i3d**, **video** and **mednet**.
