## Introduction

This is the implementation of the zero-shot classification task.

## Requirements

### Installation
Create a conda environment and install dependencies:
```bash
conda create -n clipoint python=3.7
conda activate clipoint

pip install -r requirements.txt

# Install the according versions of torch and torchvision
conda install pytorch torchvision cudatoolkit

# Install the modified dassl library (no need to re-build if the source code is changed)
# Under CLIPoint/zeroshot_fewshot_cls folder:
cd Dassl3D/
python setup.py develop

cd ..
```

### Dataset
Download the official ModelNet40 and ScanobjectNN dataset and put the folder under `data/`. Or you can directly download from this [Google Drive](https://drive.google.com/drive/folders/145flu-CtXPlhJ2nrSUUe7tmUj1DTts7t?usp=sharing). 
After download, the directory structure should be:
```bash
│zeroshot_cls/
├──...
├──data/
│   ├──modelnet40_ply_hdf5_2048/
│   ├──scanobjectnn/
├──...
```

## Get Started

### Zero-shot Classification
Edit the running command in `zeroshot_cls.sh`, e.g. config file and output directory. Then run Zero-shot classification:
```bash
bash zeroshot.sh
```
The dataset can be change by commenting or uncommenting line 4-5 or 6-7. 


## Acknowlegment
This repo benefits from [PointCLIP](https://github.com/ZrrSkywalker/PointCLIP), [CLIP](https://github.com/openai/CLIP), [SimpleView](https://github.com/princeton-vl/SimpleView) and the excellent codebase [Dassl](https://github.com/KaiyangZhou/Dassl.pytorch). Thanks for their wonderful works.
