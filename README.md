# PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning

Official implementation of [PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning](https://arxiv.org/abs/2211.11682).

[PointCLIP V1](https://openaccess.thecvf.com/content/CVPR2022/papers/Zhang_PointCLIP_Point_Cloud_Understanding_by_CLIP_CVPR_2022_paper.pdf) has been released at [repo](https://github.com/ZrrSkywalker/PointCLIP).

## Introduction
PointCLIP V2 is a powerful 3D open-world learner, which improves the performance of PointCLIP with significant margins. V2 utilizes a realistic shape projection module for depth map generation, and adopts the LLM-assisted 3D prompt to align visual and language representations. Besides classification, PointCLIP V2 also conducts zero-shot part segmentation and 3D object detection.


<!-- Examples of the synthesized depth map and attention map: -->
![Depth and Attention Map](figs/depth_attention_map.png)


<!-- The whole framework of PointCLIP V2: -->
<!-- ![Whole Framework](figs/whole_framework.png) -->


## Code

The 'zeroshot_cls' folder contains the code for zero-shot classification, and 'zeroshot_seg' contains code for zero-shot part segmentation.

## Contributors
[Xiangyang Zhu](https://github.com/yangyangyang127), [Renrui Zhang](https://github.com/ZrrSkywalker)


## Citation
Thanks for citing our paper:

```
@article{Zhu2022PointCLIPV2,
    title={PointCLIP V2: Adapting CLIP for Powerful 3D Open-world Learning},
    author={Zhu, Xiangyang and Zhang, Renrui and He, Bowei and Zeng, Ziyao and Zhang, Shanghang and Gao, Peng},
    journal={arXiv preprint arXiv:2211.11682},
    year={2022},
}
```

## Contact
If you have any question about this project, please feel free to contact xiangyzhu6-c@my.cityu.edu.hk and zhangrenrui@pjlab.org.cn.

