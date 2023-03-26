import os
import torch
import argparse
from dassl.engine import build_trainer
from dassl.config import get_cfg_default
from dassl.utils import setup_logger, set_random_seed, collect_env_info

import datasets.scanobjnn
import datasets.modelnet40
from trainers import best_param

from trainers import zeroshot
from trainers.post_search import search_weights_zs, search_prompt_zs

def print_args(args, cfg):
    print('***************')
    print('** Arguments **')
    print('***************')
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print('{}: {}'.format(key, args.__dict__[key]))
    print('************')
    print('** Config **')
    print('************')
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.seed:
        cfg.SEED = args.seed

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone


def extend_cfg(cfg):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN
    cfg.TRAINER.EXTRA = CN()


def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    # 3. From input arguments
    reset_cfg(cfg, args)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

def main(args):
    cfg = setup_cfg(args)
    
    # set random seed
    if cfg.SEED >= 0:
        print('Setting fixed seed: {}'.format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    print('Collecting env info ...')
    print('** System info **\n{}\n'.format(collect_env_info()))

    trainer = build_trainer(cfg)

    # zero-shot classification
    if args.zero_shot:
        trainer.test_zs()
        
    # view weight and prompt search
    vweights = best_param.best_prompt_weight['{}_{}_test_weights'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
    prompts = best_param.best_prompt_weight['{}_{}_test_prompts'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)]
    if args.post_search:
        if args.zero_shot:
            prompts, image_feature = search_prompt_zs(cfg, vweights, searched_prompt=prompts)
            #vweights = search_weights_zs(cfg, prompts, vweights, image_feature)
            return
            
                
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='', help='path to dataset')
    parser.add_argument('--output-dir', type=str, default='', help='output directory')
    parser.add_argument('--seed', type=int,default=2,help='only positive value enables a fixed seed')
    parser.add_argument('--transforms', type=str, nargs='+', help='data augmentation methods')
    parser.add_argument('--config-file', type=str, default='', help='path to config file')
    parser.add_argument('--dataset-config-file', type=str, default='', help='path to config file for dataset setup')
    parser.add_argument('--trainer', type=str, default='', help='name of trainer')
    parser.add_argument('--backbone', type=str, default='', help='name of CNN backbone')
    parser.add_argument('--head', type=str, default='', help='name of head')
    parser.add_argument('--zero-shot', action='store_true', help='zero-shot only')
    parser.add_argument('--post-search', default=True, action='store_true', help='post-search only')
    parser.add_argument('--model-dir', type=str, default='',help='load model from this directory for eval-only mode')
    parser.add_argument('--load-epoch', type=int, default=175, help='load model weights at this epoch for evaluation')
    parser.add_argument('--no-train', action='store_true', help='do not call trainer.train()')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER, help='modify config options using the command-line')
    args = parser.parse_args()
    main(args)
    
