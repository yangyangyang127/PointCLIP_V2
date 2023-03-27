## Introduction

This is the implementation of the zero-shot segmentation task. The code can re-produce the zero-shot part segmentation IoU on ShapeNet-Part dataset.

|  | Airplane | Bag | Cap | Chair | Earphone | Guitar | Knife | Lamp | Laptop | Moter | Mug | Pistol | Rocket | Skate | Table |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| IoU | 33.5| 60.4 | 52.8 | 51.5 | 56.5 | 71.5 | 66.7 | 44.6 | 61.6 | 31.5 | 48.0 | 46.0 | 49.6 | 43.9 | 61.1 |


## Dataset
Download the official ShapeNetPart dataset and put the folder under `data/`. Or you can directly download from this [Google Drive](https://drive.google.com/drive/folders/1TdC14kVjvNBLsb-QXIEXEHxVw5tCrOSB?usp=sharing).
After that, the directory structure should be:
```bash
│zeroshot_seg/
├──...
├──data/
│   ├──shapenet_part_seg_hdf5_data/
├──...
```

## Get Started

### Zero-shot Segmentation
Edit the running command in `zeroshot_cls.sh`, e.g. choice of category and data path. Then run zero-shot segmentation:
```bash
bash zeroshot_seg.sh
```
The prior category should be specified in line 7 of this .sh file, i.e.
```bash
CLASS=chair
```
The classnames can be selected from `[airplane, bag, cap, car, chair, earphone, guitar, knife, lamp, laptop, motorbike, mug, pistol, rocket, skateboard, table]`.

## Acknowlegment
This repo benefits from [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP), [CLIP](https://github.com/openai/CLIP), [CurveNet](https://github.com/tiangexiang/CurveNet), and [dgcnn](https://github.com/antao97/dgcnn.pytorch). Thanks for their wonderful works.
