# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
from xmlrpc.client import TRANSPORT_ERROR
from black import T
import numpy as np
import torch
from torch import nn
from torch.nn import init
from demo_visualizer import Have_a_Look
from detectron2.data.detection_utils import convert_image_to_rgb
from detectron2.structures import ImageList, Boxes, Instances
from detectron2.utils.events import get_event_storage
from detectron2.utils.logger import log_first_n

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.proposal_generator import build_proposal_generator
from .fsod_roi_heads import build_roi_heads
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from torch.autograd import Variable
from detectron2.modeling.poolers import ROIPooler
import torch.nn.functional as F 
from demo_visualizer import Have_a_Look
from .fsod_fast_rcnn import FsodFastRCNNOutputs

import os
import math
import matplotlib.pyplot as plt
import pandas as pd

from detectron2.data.catalog import MetadataCatalog
import detectron2.data.detection_utils as utils
import pickle
import sys



@META_ARCH_REGISTRY.register()
class CenterNet2Detector(nn.Module):
    """
    Generalized R-CNN. Any models that contains the following three components:
    1. Per-image feature extraction (aka backbone)
    2. Region proposal generation
    3. Per-region feature extraction and prediction
    """

    def __init__(self, cfg,pos_encoding=True):
        super().__init__()

        self.backbone = build_backbone(cfg)
        self.proposal_generator = build_proposal_generator(cfg, self.backbone.output_shape())
        self.roi_heads = build_roi_heads(cfg, self.backbone.output_shape())
        self.vis_period = cfg.VIS_PERIOD
        self.input_format = cfg.INPUT.FORMAT

        assert len(cfg.MODEL.PIXEL_MEAN) == len(cfg.MODEL.PIXEL_STD)
        self.register_buffer("pixel_mean", torch.Tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1))
        self.register_buffer("pixel_std", torch.Tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1))

        self.in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES

        self.support_way = cfg.INPUT.FS.SUPPORT_WAY
        self.support_shot = cfg.INPUT.FS.SUPPORT_SHOT
        self.logger = logging.getLogger(__name__)
        # self.conv_1 = nn.Conv2d(160, 144, 1, padding=0, bias=False)
        self.agp1=nn.AdaptiveAvgPool2d((32,32))
        self.agp2=nn.AdaptiveAvgPool2d((16,16))
        self.agp3=nn.AdaptiveAvgPool2d((8,8))
        self.vip_p3=SM_Block(128,32)
        self.vip_p4=SM_Block(128,16)
        self.vip_p5=SM_Block(128,8)
        self.support_pool_1x1=nn.AdaptiveAvgPool2d((1,1))
        # self.support_pool_3x1=nn.AdaptiveAvgPool2d((1,1))
        self.support_pool_1x3=nn.AdaptiveAvgPool2d((1,3))
        self.support_pool_3x1=nn.AdaptiveAvgPool2d((3,1))
        self.conv1 = nn.Conv2d(128, 64, 1)
        self.conv2 = nn.Conv2d(128, 64, 1)
        self.conv3 = nn.Conv2d(256, 128, 1)
        
        # self.cot_p3=CoTAttention(128,3)
        # self.cot_p4=CoTAttention(128,3)
        # self.cot_p5=CoTAttention(128,3)
        # self.Polarize_p3=ParallelPolarizedSelfAttention(128)
        # self.Polarize_p4=ParallelPolarizedSelfAttention(128)
        # self.Polarize_p5=ParallelPolarizedSelfAttention(128)
        # self.cbam_p3=CBAMBlock(128)
        # self.cbam_p4=CBAMBlock(128)
        # self.cbam_p5=CBAMBlock(128)

    @property
    def device(self):
        return self.pixel_mean.device

    def visualize_training(self, batched_inputs, proposals):
        """
        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = "Left: GT bounding boxes;  Right: Predicted proposals"
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if not self.training:
            self.init_model()
            return self.inference(batched_inputs)
        
        images, support_images = self.preprocess_image(batched_inputs)
        if "instances" in batched_inputs[0]:
            for x in batched_inputs:
                x['instances'].set('gt_classes', torch.full_like(x['instances'].get('gt_classes'), 0))
            
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None
        
        features = self.backbone(images.tensor)
        
        support_bboxes_ls = []
        for item in batched_inputs:
            bboxes = item['support_bboxes']
            for box in bboxes:
                box = Boxes(box[np.newaxis, :])
                support_bboxes_ls.append(box.to(self.device))
        
        B, N, C, H, W = support_images.tensor.shape
        assert N == self.support_way * self.support_shot

        support_images = support_images.tensor.reshape(B*N, C, H, W)
        support_features = self.backbone(support_images)
        support_features_pooler_rcnn = []
        support_features_pool_rcnn_8 = self.roi_heads.box_pooler([support_features[f] for f in self.in_features], support_bboxes_ls)
        # support_features_pool_rcnn_12 = self.roi_heads.box_pooler1([support_features[f] for f in self.in_features], support_bboxes_ls)
        support_features_pool_rcnn_4 = self.roi_heads.box_pooler2([support_features[f] for f in self.in_features], support_bboxes_ls)
        support_features_pooler_rcnn = [support_features_pool_rcnn_8,support_features_pool_rcnn_4]


        assert self.support_way == 1 # now only 2 way support
        
        loss_centernet_loc=[]
        loss_centernet_agn_pos=[]
        loss_centernet_agn_neg=[]
        loss_cls_stage0=[]
        loss_box_reg_stage0=[]
        loss_cls_stage1=[]
        loss_box_reg_stage1=[]
        loss_cls_stage2=[]
        loss_box_reg_stage2=[]
        for i in range(B): # batch
            # query
            query_gt_instances = [gt_instances[i]] # one query gt instances
            query_images = ImageList.from_tensors([images[i]]) # one query image

            query_feature_p3 = features['p3'][i].unsqueeze(0) # one query feature for attention [1,C,H,W]
            query_feature_p4 = features['p4'][i].unsqueeze(0) 
            query_feature_p5 = features['p5'][i].unsqueeze(0) 
            
            
            
            pos_begin = i * self.support_shot * self.support_way
            pos_end = pos_begin + self.support_shot
            support_features_p3 = support_features['p3'] #[9,c,h,w]
            support_features_p4 = support_features['p4']
            support_features_p5 = support_features['p5']
            
            
            ################vip
            support_features_p3 = self.agp1(support_features_p3).permute(0,2,3,1)
            support_features_p4 = self.agp2(support_features_p4).permute(0,2,3,1)
            support_features_p5 = self.agp3(support_features_p5).permute(0,2,3,1)

            support_features_p3 = self.vip_p3(support_features_p3).permute(0,3,2,1) #[9,16,16,160]
            support_features_p4 = self.vip_p4(support_features_p4).permute(0,3,2,1)
            support_features_p5 = self.vip_p5(support_features_p5).permute(0,3,2,1)
            ######################

            support_features_p3_pool = support_features_p3[pos_begin:pos_end].mean(0, True)
            support_features_p4_pool = support_features_p4[pos_begin:pos_end].mean(0, True)
            support_features_p5_pool = support_features_p5[pos_begin:pos_end].mean(0, True)
            
            #p3
            support_pool_p3_1x1 = self.support_pool_1x1(support_features_p3_pool)
            support_pool__p3_1x3 = self.support_pool_1x3(support_features_p3_pool)
            support_pool_p3_3x1 = self.support_pool_3x1(support_features_p3_pool)
            
            
            pos_correlation__p3_1_1 = F.relu(F.conv2d(query_feature_p3, support_pool_p3_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation__p3_1_2 = F.relu(F.conv2d(pos_correlation__p3_1_1, support_pool_p3_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p3_2_1 = F.relu(F.conv2d(query_feature_p3, support_pool__p3_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p3_2_2 = F.relu(F.conv2d(pos_correlation_p3_2_1, support_pool_p3_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            
            # pos_correlation_p3_3_1 = F.conv2d(query_feature_p3, support_pool_p3_1x7.permute(1,0,2,3),padding=(0, 3), groups=128)
            # pos_correlation_p3_3_2 = F.conv2d(pos_correlation_p3_3_1, support_pool_p3_7x1.permute(1,0,2,3),padding=(3, 0), groups=128)
            attn1 = pos_correlation__p3_1_2 + pos_correlation_p3_2_2 +query_feature_p3
            attn1 = F.relu(self.conv3(torch.cat((attn1,query_feature_p3),1)))#+torch.cat((self.conv1(attn1),self.conv2(query_feature_p3)),1)

            #p4
            support_pool_p4_1x1 = self.support_pool_1x1(support_features_p4_pool)
            # support_pool_p4_3x1 = self.support_pool_3x1(support_features_p4_pool)
            support_pool_p4_1x3 = self.support_pool_1x3(support_features_p4_pool)
            support_pool_p4_3x1 = self.support_pool_3x1(support_features_p4_pool)
            
            
            pos_correlation_p4_1_1 = F.relu(F.conv2d(query_feature_p4, support_pool_p4_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation_p4_1_2 = F.relu(F.conv2d(pos_correlation_p4_1_1, support_pool_p4_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p4_2_1 = F.relu(F.conv2d(query_feature_p4, support_pool_p4_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p4_2_2 = F.relu(F.conv2d(pos_correlation_p4_2_1, support_pool_p4_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            
            attn2 = pos_correlation_p4_1_2 + pos_correlation_p4_2_2  +query_feature_p4
            attn2 = F.relu(self.conv3(torch.cat((attn2,query_feature_p4),1)))#+torch.cat((self.conv1(attn2),self.conv2(query_feature_p4)),1)
            
            
            #p5
            support_pool_p5_1x1 = self.support_pool_1x1(support_features_p5_pool)
            support_pool_p5_1x3 = self.support_pool_1x3(support_features_p5_pool)
            support_pool_p5_3x1 = self.support_pool_3x1(support_features_p5_pool)
            
            pos_correlation_p5_1_1 = F.relu(F.conv2d(query_feature_p5, support_pool_p5_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation_p5_1_2 = F.relu(F.conv2d(pos_correlation_p5_1_1, support_pool_p5_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p5_2_1 = F.relu(F.conv2d(query_feature_p5, support_pool_p5_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p5_2_2 = F.relu(F.conv2d(pos_correlation_p5_2_1, support_pool_p5_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            
            attn3 = pos_correlation_p5_1_2 + pos_correlation_p5_2_2  + query_feature_p5
            attn3 = F.relu(self.conv3(torch.cat((attn3,query_feature_p5),1)))#+torch.cat((self.conv1(attn3),self.conv2(query_feature_p5)),1)
            
            pos_features = {'p3': attn1,'p4': attn2,'p5': attn3} # attention map for attention rpn
            
            proposals, proposal_losses  = self.proposal_generator(query_images, pos_features, query_gt_instances) # attention rpn
            _, detector_losses = self.roi_heads(query_images, features,support_features_pooler_rcnn, proposals, query_gt_instances)
        #     loss_centernet_loc.append(proposal_losses['loss_centernet_loc'])
        #     loss_centernet_agn_pos.append(proposal_losses['loss_centernet_agn_pos'])
        #     loss_centernet_agn_neg.append(proposal_losses['loss_centernet_agn_neg'])
            
        #     loss_cls_stage0.append(detector_losses['loss_cls_stage0'])
        #     loss_box_reg_stage0.append(detector_losses['loss_box_reg_stage0'])
        #     loss_cls_stage1.append(detector_losses['loss_cls_stage1'])
        #     loss_box_reg_stage1.append(detector_losses['loss_box_reg_stage1'])
        #     loss_cls_stage2.append(detector_losses['loss_cls_stage2'])
        #     loss_box_reg_stage2.append(detector_losses['loss_box_reg_stage2'])
        
        # proposal_losses = {}
        # detector_losses = {}
        # proposal_losses['loss_centernet_loc']= torch.stack(loss_centernet_loc).mean()
        # proposal_losses['loss_centernet_agn_pos']= torch.stack(loss_centernet_agn_pos).mean()
        # proposal_losses['loss_centernet_agn_neg']= torch.stack(loss_centernet_agn_neg).mean()
        
        # detector_losses['loss_cls_stage0']= torch.stack(loss_cls_stage0).mean()
        # detector_losses['loss_box_reg_stage0']= torch.stack(loss_box_reg_stage0).mean()
        # detector_losses['loss_cls_stage1']= torch.stack(loss_cls_stage1).mean()
        # detector_losses['loss_box_reg_stage1']= torch.stack(loss_box_reg_stage1).mean()
        # detector_losses['loss_cls_stage2']= torch.stack(loss_cls_stage2).mean()
        # detector_losses['loss_box_reg_stage2']= torch.stack(loss_box_reg_stage2).mean()
        
        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)
        return losses
        
        
        

    def init_model(self):
        self.support_on = True #False

        support_dir = './support_dir'
        if not os.path.exists(support_dir):
            os.makedirs(support_dir)

        support_file_name = os.path.join(support_dir, 'support_feature.pkl')
        if not os.path.exists(support_file_name):
            support_path = './datasets/coco/10_shot_support_df.pkl'
            support_df = pd.read_pickle(support_path)

            metadata = MetadataCatalog.get('coco_2017_val_stone')
            # # unmap the category mapping ids for COCO
            # reverse_id_mapper = lambda dataset_id: metadata.thing_dataset_id_to_contiguous_id[dataset_id]  # noqa
            # support_df['category_id'] = support_df['category_id'].map(reverse_id_mapper)
            support_dict = {'p3': {}, 'p4': {},'p5': {},'rcnn_8': {},'rcnn_4': {}}
            #support_dict = {'p3': {}, 'p4': {},'p5': {},'p6': {},'p7': {}}
            for cls in support_df['category_id'].unique():
                support_cls_df = support_df.loc[support_df['category_id'] == cls, :].reset_index()
                support_data_all = []
                support_box_all = []

                for index, support_img_df in support_cls_df.iterrows():
                    if index < self.support_shot :
                        img_path = os.path.join('./datasets/coco', support_img_df['file_path'])
                        support_data = utils.read_image(img_path, format='BGR')
                        support_data = torch.as_tensor(np.ascontiguousarray(support_data.transpose(2, 0, 1)))
                        support_data_all.append(support_data)

                        support_box = support_img_df['support_box']
                        support_box_all.append(Boxes([support_box]).to(self.device))
                    else: 
                        break
                # support images
                support_images = [x.to(self.device) for x in support_data_all]
                support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
                support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)
                support_features = self.backbone(support_images.tensor)
                
                
                
                support_features_pool_rcnn_8 = self.roi_heads.box_pooler([support_features[f] for f in self.in_features], support_box_all)
                # support_features_pool_rcnn_12 = self.roi_heads.box_pooler1([support_features[f] for f in self.in_features], support_box_all)
                support_features_pool_rcnn_4 = self.roi_heads.box_pooler2([support_features[f] for f in self.in_features], support_box_all)
                
                
                
                support_features_p3 = support_features['p3']
                support_features_p4 = support_features['p4']
                support_features_p5 = support_features['p5']
                #support_features_p6 = support_features['p6']
                #support_features_p7 = support_features['p7']
                #######################vip
                support_features_p3 = self.agp1(support_features_p3).permute(0,2,3,1)
                support_features_p4 = self.agp2(support_features_p4).permute(0,2,3,1)
                support_features_p5 = self.agp3(support_features_p5).permute(0,2,3,1)
                
                support_features_p3 = self.vip_p3(support_features_p3).permute(0,3,2,1) #[9,16,16,160]
                support_features_p4 = self.vip_p4(support_features_p4).permute(0,3,2,1)
                support_features_p5 = self.vip_p5(support_features_p5).permute(0,3,2,1)
               
                support_features_p3_pool = support_features_p3.mean(0, True)
                support_features_p4_pool = support_features_p4.mean(0, True)
                support_features_p5_pool = support_features_p5.mean(0, True)
                

                # Have_a_Look(support_features_p4_pool,4)
                
                # print(support_features_pool_rcnn.shape)
                # support_features_pool_rcnn = support_features_p3_pool + support_features_p4_pool +support_features_p5_pool #for rcnn 
                support_dict['p3'][cls] = support_features_p3_pool.detach().cpu().data
                support_dict['p4'][cls] = support_features_p4_pool.detach().cpu().data
                support_dict['p5'][cls] = support_features_p5_pool.detach().cpu().data
                # support_dict['rcnn_12'][cls]  = support_features_pool_rcnn_12.detach().cpu().data
                support_dict['rcnn_8'][cls]  = support_features_pool_rcnn_8.detach().cpu().data
                support_dict['rcnn_4'][cls]  = support_features_pool_rcnn_4.detach().cpu().data
                
                #support_dict['p6'][cls] = support_features_p6.detach().cpu().data
                #support_dict['p7'][cls] = support_features_p7.detach().cpu().data
                print(type(support_dict))
                print(len(support_dict))

                del support_features_p3
                del support_features_p4
                del support_features_p5
                #del support_features_p6
                #del support_features_p7
                del support_features
                

            with open(support_file_name, 'wb') as f:
               pickle.dump(support_dict, f)
            self.logger.info("=========== Offline support features are generated. ===========")
            self.logger.info("============ Few-shot object detetion will start. =============")
            sys.exit(0)
            
        else:
            with open(support_file_name, "rb") as hFile:
                self.support_dict  = pickle.load(hFile, encoding="latin1")
            for res_key, res_dict in self.support_dict.items():
                for cls_key, feature in res_dict.items():
                    self.support_dict[res_key][cls_key] = feature.cuda()

    def inference(self, batched_inputs, detected_instances=None, do_postprocess=True):
        """
        Run inference on the given inputs.
        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.
        Returns:
            same as in :meth:`forward`.
        """
        assert not self.training
        
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        B, _, _, _ = features['p3'].shape
        assert B == 1 # only support 1 query image in test
        assert len(images) == 1
        
        
        query_images = ImageList.from_tensors([images[0]]) # one query image    
        query_feature_p3 = features['p3'] # one query feature for attention rpn
        query_feature_p4 = features['p4'] # one query feature for attention rpn
        query_feature_p5 = features['p5'] # one query feature for attention rpn
        #query_feature_p6 = features['p6'] # one query feature for attention rpn
        #query_feature_p7 = features['p7'] # one query feature for attention rpn
        
        
        
        
        
        
        for cls_id, support_features_p3_pool in self.support_dict['p3'].items(): #以列表的形式返回可遍历的（键，值）元组
            
            
            # support_features_p3_pool = self.support_dict['p3'][cls_id]
            support_pool_p3_1x1 = self.support_pool_1x1(support_features_p3_pool)
            support_pool__p3_1x3 = self.support_pool_1x3(support_features_p3_pool)
            support_pool_p3_3x1 = self.support_pool_3x1(support_features_p3_pool)
            
            
            pos_correlation__p3_1_1 = F.relu(F.conv2d(query_feature_p3, support_pool_p3_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation__p3_1_2 = F.relu(F.conv2d(pos_correlation__p3_1_1, support_pool_p3_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p3_2_1 = F.relu(F.conv2d(query_feature_p3, support_pool__p3_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p3_2_2 = F.relu(F.conv2d(pos_correlation_p3_2_1, support_pool_p3_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            
            attn1 = pos_correlation__p3_1_2 + pos_correlation_p3_2_2 +query_feature_p3
            attn1 = F.relu(self.conv3(torch.cat((attn1,query_feature_p3),1)))#+torch.cat((self.conv1(attn1),self.conv2(query_feature_p3)),1)
            # attn1 = torch.cat((attn1,query_feature_p3),1)
            # attn1 =self.conv3(attn1)
            
        for cls_id, support_features_p4_pool in self.support_dict['p4'].items(): 
            
            support_pool_p4_1x1 = self.support_pool_1x1(support_features_p4_pool)
            # support_pool_p4_3x1 = self.support_pool_3x1(support_features_p4_pool)
            support_pool_p4_1x3 = self.support_pool_1x3(support_features_p4_pool)
            support_pool_p4_3x1 = self.support_pool_3x1(support_features_p4_pool)
            
            
            pos_correlation_p4_1_1 = F.relu(F.conv2d(query_feature_p4, support_pool_p4_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation_p4_1_2 = F.relu(F.conv2d(pos_correlation_p4_1_1, support_pool_p4_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p4_2_1 = F.relu(F.conv2d(query_feature_p4, support_pool_p4_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p4_2_2 = F.relu(F.conv2d(pos_correlation_p4_2_1, support_pool_p4_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            

            attn2 = pos_correlation_p4_1_2 + pos_correlation_p4_2_2  +query_feature_p4
            # attn2 = torch.cat((attn2,query_feature_p4),1)
            attn2 = F.relu(self.conv3(torch.cat((attn2,query_feature_p4),1)))#+torch.cat((self.conv1(attn2),self.conv2(query_feature_p4)),1)
            # attn2 =self.conv3(attn2)
            

        for cls_id, support_features_p5_pool in self.support_dict['p5'].items():     
            
            # support_features_p5_pool = self.support_dict['p5'][cls_id]
            support_pool_p5_1x1 = self.support_pool_1x1(support_features_p5_pool)
            support_pool_p5_1x3 = self.support_pool_1x3(support_features_p5_pool)
            support_pool_p5_3x1 = self.support_pool_3x1(support_features_p5_pool)
            
            pos_correlation_p5_1_1 = F.relu(F.conv2d(query_feature_p5, support_pool_p5_1x1.permute(1,0,2,3),padding=(0, 0), groups=128)) # attention map
            pos_correlation_p5_1_2 = F.relu(F.conv2d(pos_correlation_p5_1_1, support_pool_p5_1x1.permute(1,0,2,3),padding=(0, 0), groups=128))
            
            pos_correlation_p5_2_1 = F.relu(F.conv2d(query_feature_p5, support_pool_p5_1x3.permute(1,0,2,3),padding=(0, 1), groups=128))
            pos_correlation_p5_2_2 = F.relu(F.conv2d(pos_correlation_p5_2_1, support_pool_p5_3x1.permute(1,0,2,3),padding=(1, 0), groups=128))
            
            attn3 = pos_correlation_p5_1_2 + pos_correlation_p5_2_2  + query_feature_p5
            attn3 = F.relu(self.conv3(torch.cat((attn3,query_feature_p5),1)))#+torch.cat((self.conv1(attn3),self.conv2(query_feature_p5)),1)
            # Have_a_Look(query_feature_p4,4)
        support_features_pooler_rcnn = []    
        # support_features_rcnn_12 = self.support_dict['rcnn_12'][cls_id]
        support_features_rcnn_8 = self.support_dict['rcnn_8'][cls_id]
        support_features_rcnn_4 = self.support_dict['rcnn_4'][cls_id]
        support_features_pooler_rcnn = [support_features_rcnn_8,support_features_rcnn_4] 

        pos_features = {'p3': attn1,'p4': attn2,'p5': attn3} # attention map for attention rpn
        
        del attn1
        del attn2
        del attn3
        del query_feature_p3
        del query_feature_p4
        del query_feature_p5

        
        proposals, _ = self.proposal_generator(query_images, pos_features, None)
        
        results, _ = self.roi_heads(query_images, features,support_features_pooler_rcnn, proposals, None)
        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return CenterNet2Detector._postprocess(results, batched_inputs, images.image_sizes)
        else:
            return results
        
        
        
        

    def preprocess_image(self, batched_inputs):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)
        if self.training:
            # support images
            support_images = [x['support_images'].to(self.device) for x in batched_inputs]
            support_images = [(x - self.pixel_mean) / self.pixel_std for x in support_images]
            support_images = ImageList.from_tensors(support_images, self.backbone.size_divisibility)

            return images, support_images
        else:
            return images

    @staticmethod
    def _postprocess(instances, batched_inputs, image_sizes):
        """
        Rescale the output instances to the target size.
        """
        # note: private function; subject to changes
        processed_results = []
        for results_per_image, input_per_image, image_size in zip(
            instances, batched_inputs, image_sizes
        ):
            height = input_per_image.get("height", image_size[0])
            width = input_per_image.get("width", image_size[1])
            r = detector_postprocess(results_per_image, height, width)
            processed_results.append({"instances": r})
        return processed_results

class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class SM_Block(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim=seg_dim

        # self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)

        self.reweighting=MLP(dim,dim//2,dim*2)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,x) :
        B,H,W,C=x.shape
        

        # c_embed=self.mlp_c(x)

        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,self.seg_dim,S)
        # print(x.shape)
        h_embed=h_embed.permute(0,3,2,1,4)
        # print(h_embed.shape)
        h_embed=h_embed.reshape(B,self.seg_dim,W,H*S)
        # print(h_embed.shape)
        # print('###')
        h_embed=self.mlp_h(h_embed)
        h_embed=h_embed.reshape(B,self.seg_dim,W,H,S)
        # print(h_embed.shape)
        h_embed=h_embed.permute(0,3,2,1,4).reshape(B,H,W,C)
        # print(h_embed.shape)
        w_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,1,2,4).reshape(B,self.seg_dim,H,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,W,S).permute(0,2,3,1,4).reshape(B,H,W,C)
        # print((c_embed+h_embed+w_embed).shape)
        weight=(h_embed+w_embed).permute(0,3,1,2).flatten(2).mean(2)
        # print(weight.shape)
        weight=self.reweighting(weight).reshape(B,C,2)
        # print(weight.shape)
        weight=weight.permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2)
        
        x=w_embed*weight[0]+h_embed*weight[1]

        x=self.proj_drop(self.proj(x))

        return x







class CoTAttention(nn.Module):

    def __init__(self, dim=512,kernel_size=3):
        super().__init__()
        self.dim=dim
        self.kernel_size=kernel_size

        self.key_embed=nn.Sequential(
            nn.Conv2d(dim,dim,kernel_size=kernel_size,padding=kernel_size//2,groups=4,bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed=nn.Sequential(
            nn.Conv2d(dim,dim,1,bias=False),
            nn.BatchNorm2d(dim)
        )

        factor=4
        self.attention_embed=nn.Sequential(
            nn.Conv2d(2*dim,2*dim//factor,1,bias=False),
            nn.BatchNorm2d(2*dim//factor),
            nn.ReLU(),
            nn.Conv2d(2*dim//factor,kernel_size*kernel_size*dim,1)
        )


    def forward(self, x):
        bs,c,h,w=x.shape
        k1=self.key_embed(x) #bs,c,h,w
        v=self.value_embed(x).view(bs,c,-1) #bs,c,h,w

        y=torch.cat([k1,x],dim=1) #bs,2c,h,w
        att=self.attention_embed(y) #bs,c*k*k,h,w
        att=att.reshape(bs,c,self.kernel_size*self.kernel_size,h,w)
        att=att.mean(2,keepdim=False).view(bs,c,-1) #bs,c,h*w
        k2=F.softmax(att,dim=-1)*v
        k2=k2.view(bs,c,h,w)


        return k1+k2
    




class ParallelPolarizedSelfAttention(nn.Module):

    def __init__(self, channel=512):
        super().__init__()
        self.ch_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.ch_wq=nn.Conv2d(channel,1,kernel_size=(1,1))
        self.softmax_channel=nn.Softmax(1)
        self.softmax_spatial=nn.Softmax(-1)
        self.ch_wz=nn.Conv2d(channel//2,channel,kernel_size=(1,1))
        self.ln=nn.LayerNorm(channel)
        self.sigmoid=nn.Sigmoid()
        self.sp_wv=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.sp_wq=nn.Conv2d(channel,channel//2,kernel_size=(1,1))
        self.agp=nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        b, c, h, w = x.size()

        #Channel-only Self-Attention
        channel_wv=self.ch_wv(x) #bs,c//2,h,w
        channel_wq=self.ch_wq(x) #bs,1,h,w
        channel_wv=channel_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        channel_wq=channel_wq.reshape(b,-1,1) #bs,h*w,1
        channel_wq=self.softmax_channel(channel_wq)
        channel_wz=torch.matmul(channel_wv,channel_wq).unsqueeze(-1) #bs,c//2,1,1
        channel_weight=self.sigmoid(self.ln(self.ch_wz(channel_wz).reshape(b,c,1).permute(0,2,1))).permute(0,2,1).reshape(b,c,1,1) #bs,c,1,1
        channel_out=channel_weight*x

        #Spatial-only Self-Attention
        spatial_wv=self.sp_wv(x) #bs,c//2,h,w
        spatial_wq=self.sp_wq(x) #bs,c//2,h,w
        spatial_wq=self.agp(spatial_wq) #bs,c//2,1,1
        spatial_wv=spatial_wv.reshape(b,c//2,-1) #bs,c//2,h*w
        spatial_wq=spatial_wq.permute(0,2,3,1).reshape(b,1,c//2) #bs,1,c//2
        spatial_wq=self.softmax_spatial(spatial_wq)
        spatial_wz=torch.matmul(spatial_wq,spatial_wv) #bs,1,h*w
        spatial_weight=self.sigmoid(spatial_wz.reshape(b,1,h,w)) #bs,1,h,w
        spatial_out=spatial_weight*x
        out=spatial_out+channel_out
        return out
    


class ChannelAttention(nn.Module):
    def __init__(self,channel,reduction=16):
        super().__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.se=nn.Sequential(
            nn.Conv2d(channel,channel//reduction,1,bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction,channel,1,bias=False)
        )
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result=self.maxpool(x)
        avg_result=self.avgpool(x)
        max_out=self.se(max_result)
        avg_out=self.se(avg_result)
        output=self.sigmoid(max_out+avg_out)
        return output

class SpatialAttention(nn.Module):
    def __init__(self,kernel_size=7):
        super().__init__()
        self.conv=nn.Conv2d(2,1,kernel_size=kernel_size,padding=kernel_size//2)
        self.sigmoid=nn.Sigmoid()
    
    def forward(self, x) :
        max_result,_=torch.max(x,dim=1,keepdim=True)
        avg_result=torch.mean(x,dim=1,keepdim=True)
        result=torch.cat([max_result,avg_result],1)
        output=self.conv(result)
        output=self.sigmoid(output)
        return output



class CBAMBlock(nn.Module):

    def __init__(self, channel=512,reduction=16,kernel_size=49):
        super().__init__()
        self.ca=ChannelAttention(channel=channel,reduction=reduction)
        self.sa=SpatialAttention(kernel_size=kernel_size)


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        residual=x
        out=x*self.ca(x)
        out=out*self.sa(out)
        return out+residual
