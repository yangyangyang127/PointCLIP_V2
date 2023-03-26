import json
import clip
import torch
import os.path as osp
import scipy.io as sio

class_names = {
    'ModelNet40': ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car', 'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand', 'vase', 'wardrobe', 'xbox'],
    'ScanObjectNN': ['bag', 'bin', 'box', 'cabinet', 'chair', 'desk', 'display', 'door', 'shelf', 'table', 'bed', 'pillow', 'sink', 'sofa', 'toilet']
}

def accuracy(output, target, topk=(1,)):
    pred = output.topk(max(topk), 1, True, True)[1].t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    return [float(correct[:k].reshape(-1).float().sum(0, keepdim=True).cpu().numpy()) for k in topk]


def textual_encoder(cfg, clip_model, searched_prompt=None):
    """Encoding prompts.
    """
    prompt = searched_prompt
    prompt_token = torch.cat([clip.tokenize(p) for p in prompt]).cuda()
    text_feat = clip_model.encode_text(prompt_token).repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
    return text_feat


def encode_prompt_lib(clip_model, cfg, dataset=''):
    """Encoding GPT-3 generated text to a feature library.
    """
    save_path = 'prompts/{}_{}_text_feat_lib.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2)
    if osp.exists(save_path):
        return
    
    print('Encoding prompt...')
    f = open('prompts/{}_1000.json'.format(dataset))
    gpt_sents = json.load(f)
    
    text_feat_lib = {}
    for key in gpt_sents.keys():
        temp_list = []
        for jj in range(len(gpt_sents[key])):
            prompt_token = clip.tokenize(gpt_sents[key][jj]).cuda()
            text_feat = clip_model.encode_text(prompt_token)
            temp_list.append(text_feat)
        text_feat_lib[key] = torch.cat(temp_list).cpu().numpy().squeeze()
    sio.savemat(save_path, text_feat_lib)
    print('End encoding prompt.')
    return


def random_textual_replace(cfg, prompt_feat=None, c_i=None, sent_feat=None):
    """Replace a category prompt from the pre-extracted text feature library.
    """
    temp_prompt_feat = prompt_feat.clone()
    temp_prompt_feat[c_i, :] = torch.Tensor(sent_feat).float().cuda().repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
    return temp_prompt_feat
    
    
def read_prompts(cfg, dataset='modelnet40'):
    f = open('prompts/{}_1000.json'.format(dataset))
    data = json.load(f)
    txt_feat = sio.loadmat('prompts/{}_{}_text_feat_lib.mat'.format(dataset, cfg.MODEL.BACKBONE.NAME2))
    return data, txt_feat


@torch.no_grad()
def search_prompt_zs(cfg, vweights, image_feature=None, searched_prompt=None, prompt_lib=None):
    print("\n***** Searching for prompts *****")
    
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()
    all_classes = class_names[cfg.DATASET.NAME]
    text_feat = textual_encoder(cfg, clip_model, searched_prompt=searched_prompt)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  
    
    # Encoding all GPT generated prompt.
    file = 'prompts/{}_{}_text_feat_lib.mat'.format(cfg.DATASET.NAME.lower(), cfg.MODEL.BACKBONE.NAME2)
    if not osp.exists(file):
        encode_prompt_lib(clip_model, cfg, dataset=cfg.DATASET.NAME.lower())
    
    if image_feature is None:
        image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
    else:
        image_feat = image_feature
    view_weights = torch.tensor(vweights).cuda()
    image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
    image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
    
    # Before search
    logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, zero-shot accuracy: {acc:.2f}")
    
    # Search for prompt for each category
    print("During search:")
    gpt_sents, text_feat_lib = read_prompts(cfg, dataset=cfg.DATASET.NAME.lower())
    prompts = searched_prompt
    text_feat_ori = text_feat.clone()
    best_acc = acc
    for kk in range(0, 2):
        for ii in range(len(all_classes)):  
            for jj in range(len(gpt_sents[all_classes[ii]])):
                
                text_feat = random_textual_replace(cfg, prompt_feat=text_feat_ori, c_i=ii, sent_feat=text_feat_lib[all_classes[ii]][jj,:])
                text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
                
                logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0
                acc, _ = accuracy(logits, labels, topk=(1, 5))
                acc = (acc / image_feat.shape[0]) * 100

                if acc > best_acc:
                    prompts[ii] = gpt_sents[all_classes[ii]][jj]
                    text_feat_ori[ii, :] = torch.Tensor(text_feat_lib[all_classes[ii]][jj]).cuda().repeat(1, cfg.MODEL.PROJECT.NUM_VIEWS)
                    print('New best accuracy: {:.2f}, i-th class: {}, j-th sentence: {}'.format(acc, ii, jj))
                    best_acc = acc
    print('\nThe best prompt is: ')
    print(prompts)
    
    print('\nAfter prompt search, zero-shot accuracy: {}'.format(best_acc))
    return prompts, image_feat


@torch.no_grad()
def search_weights_zs(cfg, prompt, vweights, image_feature=None, ):
    print("\n***** Searching for view weights *****")
    if image_feature is None:
        print("\n***** Loading saved features *****")
        image_feat = torch.load(osp.join(cfg.OUTPUT_DIR, "features.pt"))
    else: 
        image_feat = image_feature
    labels = torch.load(osp.join(cfg.OUTPUT_DIR, "labels.pt"))

    clip_model, _ = clip.load(cfg.MODEL.BACKBONE.NAME)
    clip_model.eval()
    text_feat = textual_encoder(cfg, clip_model, searched_prompt=prompt)
    text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
    
    view_weights = torch.tensor(vweights).cuda()

    # # Before search
    logits = clip_model.logit_scale.exp() * image_feat @ text_feat.t() * 1.0
    acc, _ = accuracy(logits, labels, topk=(1, 5))
    acc = (acc / image_feat.shape[0]) * 100
    print(f"=> Before search, zero-shot accuracy: {acc:.2f}")

    # Search
    print("During search:")
    best_acc = acc
    vw = vweights
    # Search_time can be modulated in the config for faster search
    search_time, search_range = cfg.SEARCH.TIME, cfg.SEARCH.RANGE
    search_list = [(i + 1) * search_range / search_time  for i in range(search_time)]
    for a in search_list:
        for b in search_list:
            for c in search_list:
                for d in search_list:
                    for e in search_list:
                        for f in search_list:
                            for g in search_list:
                                # Reweight different views
                                view_weights = torch.tensor([0.75, 0.75, 0.75, a, b, c, d, e, f, g]).cuda()
                                image_feat_w = image_feat.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS, cfg.MODEL.BACKBONE.CHANNEL) * view_weights.reshape(1, -1, 1)
                                image_feat_w = image_feat_w.reshape(-1, cfg.MODEL.PROJECT.NUM_VIEWS * cfg.MODEL.BACKBONE.CHANNEL).type(clip_model.dtype)
                                
                                logits = clip_model.logit_scale.exp() * image_feat_w @ text_feat.t() * 1.0                                       
                                acc, _ = accuracy(logits, labels, topk=(1, 5))
                                acc = (acc / image_feat.shape[0]) * 100

                                if acc > best_acc:
                                    vw = [0.75, 0.75, 0.75, a, b, c, d, e, f, g]
                                    print('New best accuracy: {:.2f}, view weights: {}'.format(acc, vw))
                                    best_acc = acc

    print(f"=> After view weight search, zero-shot accuracy: {best_acc:.2f}")
    return vw



