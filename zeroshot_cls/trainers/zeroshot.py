import torch
from clip import clip
import torch.nn as nn

from trainers.best_param import best_prompt_weight
from trainers.mv_utils_zs import Realistic_Projection
from dassl.engine import TRAINER_REGISTRY, TrainerX

class Textual_Encoder(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.cfg = cfg
        self.classnames = classnames
        self.clip_model = clip_model
        self.dtype = clip_model.dtype
    
    def forward(self):
        prompts = best_prompt_weight['{}_{}_test_prompts'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]
        prompts = torch.cat([clip.tokenize(p) for p in prompts]).cuda()
        text_feat = self.clip_model.encode_text(prompts).repeat(1, self.cfg.MODEL.PROJECT.NUM_VIEWS)
        return text_feat


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    
    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location='cpu').eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location='cpu')
    
    model = clip.build_model(state_dict or model.state_dict())
    return model


@TRAINER_REGISTRY.register()
class PointCLIPV2_ZS(TrainerX):

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f'Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})')
        clip_model = load_clip_to_cpu(cfg)
        clip_model.cuda()

        # Encoders from CLIP
        self.visual_encoder = clip_model.visual
        textual_encoder = Textual_Encoder(cfg, classnames, clip_model)
        
        text_feat = textual_encoder()
        self.text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)
        
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.channel = cfg.MODEL.BACKBONE.CHANNEL
    
        # Realistic projection
        self.num_views = cfg.MODEL.PROJECT.NUM_VIEWS
        pc_views = Realistic_Projection()
        self.get_img = pc_views.get_img

        # Store features for post-search
        self.feat_store = []
        self.label_store = []
        
        self.view_weights = torch.Tensor(best_prompt_weight['{}_{}_test_weights'.format(self.cfg.DATASET.NAME.lower(), self.cfg.MODEL.BACKBONE.NAME2)]).cuda()

    def real_proj(self, pc, imsize=224):
        img = self.get_img(pc).cuda()
        img = torch.nn.functional.interpolate(img, size=(imsize, imsize), mode='bilinear', align_corners=True)        
        return img
    
    def model_inference(self, pc, label=None):
        
        with torch.no_grad():
            # Realistic Projection
            images = self.real_proj(pc)            
            images = images.type(self.dtype)
            
            # Image features
            image_feat = self.visual_encoder(images)
            
            image_feat = image_feat / image_feat.norm(dim=-1, keepdim=True)
            
            image_feat_w = image_feat.reshape(-1, self.num_views, self.channel) * self.view_weights.reshape(1, -1, 1)
            image_feat_w = image_feat_w.reshape(-1, self.num_views * self.channel).type(self.dtype)
                        
            image_feat = image_feat.reshape(-1, self.num_views * self.channel)

            # Store for zero-shot
            self.feat_store.append(image_feat)
            self.label_store.append(label)
                        
            logits = 100. * image_feat_w @ self.text_feat.t()
        return logits
