import os
import torch
import random
import warnings
import argparse
from tqdm import tqdm
from clip import clip
from torch.utils.data import DataLoader
warnings.filterwarnings("ignore")

from best_param import *
from data import ShapeNetPart
from realistic_projection import Realistic_Projection
from post_search import search_prompt, search_vweight

PC_NUM = 2048

class Extractor(torch.nn.Module):
    def __init__(self, model):
        super(Extractor, self).__init__()

        self.model = model
        self.pc_views = Realistic_Projection()
        self.get_img = self.pc_views.get_img
        
    def mv_proj(self, pc):
        img, is_seen, point_loc_in_img = self.get_img(pc)
        img = img[:, :, 20:204, 20:204]
        point_loc_in_img = torch.ceil((point_loc_in_img - 20) * 224. / 184.)
        img = torch.nn.functional.interpolate(img, size=(224, 224), mode='bilinear', align_corners=True)
        return img, is_seen, point_loc_in_img

    def forward(self, pc):
        img, is_seen, point_loc_in_img = self.mv_proj(pc)
        _, x = self.model.encode_image(img)
        x = x / x.norm(dim=-1, keepdim=True)
        
        B, L, C = x.shape
        x = x.reshape(B, 14, 14, C).permute(0,3,1,2)
        return is_seen, point_loc_in_img, x


def extract_feature_maps(model_name, data_path, class_choice, device):
    model, _ = clip.load(model_name, device=device)
    model.to(device)

    segmentor = Extractor(model)
    segmentor = segmentor.to(device)
    segmentor.eval()
    
    output_path = 'output/{}/'.format(model_name.replace('/', '_'))
    mode = 'test'

    save_path = os.path.join(output_path, class_choice)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if os.path.exists(os.path.join(save_path, "{}_features.pt".format(mode))):
        return 
    
    print('\nStart to extract and save feature maps of class {}...'.format(class_choice))
    test_loader = DataLoader(ShapeNetPart(data_path, partition=mode, num_points=PC_NUM, class_choice=class_choice),
                            batch_size=1, shuffle=False, drop_last=False)
    feat_store, label_store, pc_store = [], [], []
    ifseen_store, pointloc_store = [], []
    for data in tqdm(test_loader):
        pc, cat, label = data
        pc, label = pc.cuda(), label.cuda()
        with torch.no_grad():
            is_seen, point_loc_in_img, feat = segmentor(pc)
            pc_store.append(pc)
            feat_store.append(feat[None,:,:,:])
            label_store.append(label.squeeze()[None,:])
            ifseen_store.append(is_seen[None,:,:])
            pointloc_store.append(point_loc_in_img[None,:,:,:])
    pc_store = torch.cat(pc_store, dim=0)
    feat_store = torch.cat(feat_store, dim=0)
    label_store = torch.cat(label_store, dim=0)
    ifseen_store = torch.cat(ifseen_store, dim=0)
    pointloc_store = torch.cat(pointloc_store, dim=0)
    
    # save features for post-search
    print('Save feature: ============================')
    print('Save labels: =============================')
    torch.save(pc_store,  os.path.join(save_path, "{}_pc.pt".format(mode)))
    torch.save(feat_store,  os.path.join(save_path, "{}_features.pt".format(mode)))
    torch.save(label_store, os.path.join(save_path, "{}_labels.pt".format(mode)))
    torch.save(ifseen_store,  os.path.join(save_path, "{}_ifseen.pt".format(mode)))
    torch.save(pointloc_store, os.path.join(save_path, "{}_pointloc.pt".format(mode)))


def main(args):

    random.seed(0)
    device = "cuda:0"
    model_name = args.modelname
    class_choice = args.classchoice
    data_path = args.datasetpath
    only_evaluate = args.onlyevaluate

    # extract and save feature maps, labels, point locations
    extract_feature_maps(model_name, data_path, class_choice, device)

    # test or post search prompt and view weights
    prompts = search_prompt(class_choice, model_name, only_evaluate=only_evaluate)
    if not only_evaluate:
        search_vweight(class_choice, model_name, prompts)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', default='ViT-B/16')
    parser.add_argument('--classchoice', default='table')
    parser.add_argument('--datasetpath', default='data/')
    parser.add_argument('--onlyevaluate', default=True)
    args = parser.parse_args()
    main(args)
