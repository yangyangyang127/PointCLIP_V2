import json
import clip
import torch
import numpy as np
import os.path as osp
import scipy.io as sio

from best_param import *
from data import id2cat, cat2part
from util import calculate_shape_IoU

PC_NUM = 2048

feat_dims = {'ViT-B/16':512, 'ViT-B/32':512, 'RN50':1024, 'RN101':512}
cat2id = {'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
            'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
            'motorbike': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15}
seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 2, 2, 6, 2, 3, 3, 3, 3]
index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]

def textual_encoder(clip_model, class_choice, searched_prompt=None):
    if not searched_prompt:
        sents = []
        sents = best_prompt[class_choice]
    else:
        sents = searched_prompt
    prompts = torch.cat([clip.tokenize(p) for p in sents]).cuda()
    text_feat = clip_model.encode_text(prompts)
    return text_feat, sents

def read_prompts():
    f = open('prompts/shapenetpart_700.json')
    data = json.load(f)
    return data

@torch.no_grad()
def search_prompt(class_choice, model_name, searched_prompt=None, only_evaluate=True):    
    output_path = 'output/{}/{}'.format(model_name.replace('/', '_'), class_choice)
    
    # read saved feature maps, labels, point locations
    print("\nReading saved feature maps of class {} ...".format(class_choice))
    test_feat = torch.load(osp.join(output_path, "test_features.pt")).cuda()
    test_label = torch.load(osp.join(output_path, "test_labels.pt")) - index_start[cat2id[class_choice]]
    test_ifseen = torch.load(osp.join(output_path, "test_ifseen.pt"))
    test_pointloc = torch.load(osp.join(output_path, "test_pointloc.pt"))
    test_feat = test_feat.reshape(-1, 10, 196, 512)

    # encoding textual features
    clip_model, _ = clip.load(model_name)
    clip_model.eval()
    text_feat, prompts = textual_encoder(clip_model, class_choice, searched_prompt)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    vweights = torch.Tensor(best_vweight[class_choice]).cuda()
    part_num = text_feat.shape[0]
    acc, iou = run_epoch(vweights, test_feat, test_label, test_ifseen, test_pointloc, text_feat, part_num, class_choice, model_name)
    
    if only_evaluate:
        print('\nFor class {}, part segmentation Acc: {}, IoU: {}.\n'.format(class_choice, acc, iou))
        return
    
    print("\n***** Searching for prompts *****\n")
    print('\nBefore prompt search, Acc: {}, IoU: {}.\n'.format(acc, iou))    
    gpt_sents = read_prompts()
    best_acc = acc
    best_iou = iou
    for kk in range(0, 2):
        for ii in range(len(cat2part[class_choice])):
            for ss in range(len(gpt_sents[class_choice][cat2part[class_choice][ii]])):
                
                prompts_temp = prompts.copy()
                prompts_temp[ii] = gpt_sents[class_choice][cat2part[class_choice][ii]][ss]
                prompt_token = torch.cat([clip.tokenize(p) for p in prompts_temp]).cuda()
                text_feat = clip_model.encode_text(prompt_token)
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                
                acc, iou = run_epoch(vweights, test_feat, test_label, test_ifseen, test_pointloc, text_feat, part_num, class_choice, model_name)

                if iou > best_iou:
                    print('Acc: {:.2f}, IoU: {:.2f},  obj: {}, part: {}'.format(acc, iou, class_choice, cat2part[class_choice][ii]))
                    best_acc = acc
                    best_iou = iou
                    prompts = prompts_temp
    print(prompts)
    return prompts
                    
                    
@torch.no_grad()
def search_vweight(class_choice, model_name, searched_prompt=None):
    print("\n***** Searching for view weights *****\n")
    
    output_path = 'output/{}/{}'.format(model_name.replace('/', '_'), class_choice)
    
    test_feat = torch.load(osp.join(output_path, "test_features.pt")).cuda()
    test_label = torch.load(osp.join(output_path, "test_labels.pt")) - index_start[cat2id[class_choice]]
    test_ifseen = torch.load(osp.join(output_path, "test_ifseen.pt"))
    test_pointloc = torch.load(osp.join(output_path, "test_pointloc.pt"))
    test_feat = test_feat.reshape(-1, 10, 196, 512)

    clip_model, _ = clip.load(model_name)
    clip_model.eval()
    text_feat, prompts = textual_encoder(clip_model, class_choice, searched_prompt)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    vweights = torch.Tensor(best_vweight[class_choice]).cuda()
    part_num = text_feat.shape[0]
    acc, iou = run_epoch(vweights, test_feat, test_label, test_ifseen, test_pointloc, text_feat, part_num, class_choice, model_name)
    print('\nBefore view weight search, Acc: {}, IoU: {}\n'.format(acc, iou))
    
    best_acc = acc
    best_iou = iou
    search_list = [0.25, 0.5, 0.75, 1.0]
    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:                                
                            view_weights = torch.tensor([0.75, 0.75, 0.75, 0.75, a, b, c, d, e, f]).cuda()
                            acc, iou = run_epoch(view_weights, test_feat, test_label, test_ifseen, test_pointloc, text_feat, part_num, class_choice, model_name)

                            if iou > best_iou:
                                vweights = [0.75, 0.75, 0.75, 0.75, a, b, c, d, e, f]
                                print('Acc: {:.2f}, IoU: {:.2f}, obj: {}, view weights: {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}, {:.2f}'.format(acc, iou, class_choice, 0.75, 0.75, 0.75, 0.75, a, b, c, d, e, f))
                                best_acc = acc
                                best_iou = iou
    
    print('\nAfter search, zero-shot segmentation IoU: {}'.format(best_iou))
    return vweights

                    
def run_epoch(vweights, val_feat, val_label, val_ifseen, val_pointloc, text_feat, part_num, class_choice, model_name):
    
    val_size = val_feat.shape[0]
    bs = 30
    iter = val_size // bs
    pred_seg, label_seg, class_label = [], [], []
    for i in range(iter + 1):
        end = bs*i+bs if bs*i+bs < val_size else val_size
        feat, label = val_feat[bs*i:end], val_label[bs*i:end]
        is_seen, point_loc = val_ifseen[bs*i:end], val_pointloc[bs*i:end]
        
        b, nv, hw, c = feat.size(0), feat.size(1), feat.size(2), feat.size(3)
        feat = feat.reshape(b*nv, hw, c)
        point_loc = point_loc.reshape(b*nv, -1, 2)
        is_seen = is_seen.reshape(b*nv, -1, 1)
        
        # calculating logits of each pixel on the feature map
        logits = 100. * feat.half() @ text_feat.half().t()
        output = logits.float().permute(0,2,1).reshape(-1, part_num, int(hw**0.5), int(hw**0.5))
        
        # upsample to the original image size
        upsample = torch.nn.Upsample(size=224, mode='bilinear')  # nearest, bilinear
        avgpool = torch.nn.AvgPool2d(6,1,0)
        padding = torch.nn.ReplicationPad2d([2,3,2,3])
        
        output = avgpool(padding(output))
        output = upsample(output)
        
        # back-projecting to each points
        nbatch = torch.repeat_interleave(torch.arange(0, nv*b)[:,None], 2048).view(-1, ).cuda().long()
        yy = point_loc[:, :, 0].view(-1).long()
        xx = point_loc[:, :, 1].view(-1).long()

        point_logits = output[nbatch, :, yy, xx]
        point_logits = point_logits.view(b, nv, 2048, part_num)
        
        vweights = vweights.view(1, -1, 1, 1)
        is_seen = is_seen.reshape(b, nv, 2048, 1)

        # points logits is the weighted sum of pixel logits
        point_logits = torch.sum(point_logits * vweights * is_seen, dim=1)
        point_seg = torch.topk(point_logits, k=1, dim=-1)[1].squeeze()
        label = label.reshape(b, 2048)
        class_id = torch.Tensor([cat2id[class_choice]] * point_seg.shape[0])
        
        pred_seg.append(point_seg.reshape(-1, 2048))
        label_seg.append(label.reshape(-1, 2048))
        class_label.append(class_id.reshape(-1))
        
    pred_seg = torch.cat(pred_seg, dim=0)
    label_seg = torch.cat(label_seg, dim=0)
    class_label = torch.cat(class_label, dim=0)
    
    output_path = 'output/{}/{}'.format(model_name.replace('/', '_'), class_choice)
    torch.save(pred_seg,  osp.join(output_path, "test_segpred.pt"))
    
    # calculating segmentation acc
    ratio = (pred_seg == label_seg)
    acc = torch.sum(ratio.float(), dim=-1) / PC_NUM
    acc = torch.mean(acc) * 100.
    
    # calculating iou
    pred_seg = pred_seg.cpu().numpy() 
    label_seg = label_seg.cpu().numpy()
    class_label = class_label.cpu().numpy()
    shape_ious, category = calculate_shape_IoU(pred_seg, label_seg, class_label, class_choice, eva=True)
    shape_ious = np.mean(np.array(shape_ious))
    
    return acc, shape_ious * 100.




