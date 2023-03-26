## Introduction

This is the implementation of the zero-shot segmentation task.

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
The prior category should be specified in line 7 of this .sh file. 

## Acknowlegment
This repo benefits from [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP), [CLIP](https://github.com/openai/CLIP), [CurveNet](https://github.com/tiangexiang/CurveNet), and [dgcnn](https://github.com/antao97/dgcnn.pytorch). Thanks for their wonderful works.
