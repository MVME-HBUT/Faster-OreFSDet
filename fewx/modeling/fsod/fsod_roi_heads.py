# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import inspect
import logging
from nis import cat
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, nonzero_tuple
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.backbone.resnet import BottleneckBlock, make_stage
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import add_ground_truth_to_proposals
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.roi_heads import ROIHeads
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads
from .fsod_fast_rcnn import FsodFastRCNNOutputLayers
from CenterNet2.centernet.modeling.roi_heads.custom_fast_rcnn import CustomFastRCNNOutputLayers
import time
from detectron2.structures import Boxes, Instances
from torch.autograd.function import Function
from torch.nn import functional as F

ROI_HEADS_REGISTRY = Registry("ROI_HEADS")
ROI_HEADS_REGISTRY.__doc__ = """
Registry for ROI heads in a generalized R-CNN model.
ROIHeads take feature maps and region proposals, and
perform per-region computation.

The registered object will be called with `obj(cfg, input_shape)`.
The call is expected to return an :class:`ROIHeads`.
"""

logger = logging.getLogger(__name__)

def build_roi_heads(cfg, input_shape):
    """
    Build ROIHeads defined by `cfg.MODEL.ROI_HEADS.NAME`.
    """
    name = cfg.MODEL.ROI_HEADS.NAME
    return ROI_HEADS_REGISTRY.get(name)(cfg, input_shape)
'''
在predictor中的信息流是,res4的feature和proposal一起输入ROIPool,得到每个box的feature,
然后经过res5得到每一个box的feature,这里有一点需要注意,在代码中这个feature
经过box_features.mean(dim=[2, 3])使得原来7*7的feature变成了1维的向量,
例如(987,2048,7,7)的feature变成了(987,2048)给到box_predictor中的两个线性层,一个输出类别的分数，
另一个输出box的delta,然后将最终的结果经过NMS输出出来。
'''

@ROI_HEADS_REGISTRY.register()
class FsodRes5ROIHeads(ROIHeads):
    """
    The ROIHeads in a typical "C4" R-CNN model, where
    the box and mask head share the cropping and
    the per-region feature computation by a Res5 block.
    """
    
    def __init__(self, cfg, input_shape):
        super().__init__(cfg)
        #参数res5通过cls._build_res5_block(cfg)构建res5层，用来作为ROI的输入层。
        # fmt: off
        self.in_features  = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        pooler_scales     = (1.0 / input_shape[self.in_features[0]].stride, )
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        self.mask_on      = cfg.MODEL.MASK_ON
        # fmt: on
        assert not cfg.MODEL.KEYPOINT_ON
        assert len(self.in_features) == 1
        # 参数ROIPooler是完成ROIpooling操作的类，它继承自torchvision的ops。
        self.pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        #参数res5通过cls._build_res5_block(cfg)构建res5层，用来作为ROI的输入层。
        self.res5, out_channels = self._build_res5_block(cfg)
        #参数box_predictor的输入是FastRCNNOutputLayers，其作用是输出分类score和box回归delta。
        self.box_predictor = FsodFastRCNNOutputLayers(
            cfg, ShapeSpec(channels=out_channels, height=1, width=1)
        )

    def _build_res5_block(self, cfg):
        # fmt: off
        stage_channel_factor = 2 ** 3  # res5 is 8x res2
        num_groups           = cfg.MODEL.RESNETS.NUM_GROUPS
        width_per_group      = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
        bottleneck_channels  = num_groups * width_per_group * stage_channel_factor
        out_channels         = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS * stage_channel_factor
        stride_in_1x1        = cfg.MODEL.RESNETS.STRIDE_IN_1X1
        norm                 = cfg.MODEL.RESNETS.NORM
        assert not cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE[-1], \
            "Deformable conv is not yet supported in res5 head."
        # fmt: on

        blocks = make_stage(
            BottleneckBlock,
            3,
            #first_stride=2,
            stride_per_block=[2, 1, 1],
            in_channels=out_channels // 2,
            bottleneck_channels=bottleneck_channels,
            out_channels=out_channels,
            num_groups=num_groups,
            norm=norm,
            stride_in_1x1=stride_in_1x1,
        )
        return nn.Sequential(*blocks), out_channels

    def _shared_roi_transform(self, features, boxes):
        x = self.pooler(features, boxes)
        return self.res5(x)

    def roi_pooling(self, features, boxes):
        box_features = self.pooler(
            [features[f] for f in self.in_features], boxes
        )
        #feature_pooled = box_features.mean(dim=[2, 3], keepdim=True)  # pooled to 1x1

        return box_features #feature_pooled

    def forward(self, images, features, support_box_features, proposals, targets=None):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images
        
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets
        proposal_boxes = [x.proposal_boxes for x in proposals]
        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )

        #support_features = self.res5(support_features)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features, support_box_features)

        return pred_class_logits, pred_proposal_deltas, proposals

    @torch.no_grad()
    def eval_with_support(self, images, features, support_proposals_dict, support_box_features_dict):
        """
        See :meth:`ROIHeads.forward`.
        """
        del images
        
        full_proposals_ls = []
        cls_ls = []
        for cls_id, proposals in support_proposals_dict.items():
            full_proposals_ls.append(proposals[0])
            cls_ls.append(cls_id)
        
        full_proposals_ls = [Instances.cat(full_proposals_ls)]

        proposal_boxes = [x.proposal_boxes for x in full_proposals_ls]
        #assert len(proposal_boxes[0]) == 2000

        box_features = self._shared_roi_transform(
            [features[f] for f in self.in_features], proposal_boxes
        )
        
        full_scores_ls = []
        full_bboxes_ls = []
        full_cls_ls = []
        cnt = 0
        #for cls_id, support_box_features in support_box_features_dict.items():
        for cls_id in cls_ls:
            support_box_features = support_box_features_dict[cls_id]
            query_features = box_features[cnt*100:(cnt+1)*100]
            pred_class_logits, pred_proposal_deltas = self.box_predictor(query_features, support_box_features)
            full_scores_ls.append(pred_class_logits)
            full_bboxes_ls.append(pred_proposal_deltas)
            full_cls_ls.append(torch.full_like(pred_class_logits[:, 0].unsqueeze(-1), cls_id).to(torch.int8))
            del query_features
            del support_box_features

            cnt += 1
        
        class_logits = torch.cat(full_scores_ls, dim=0)
        proposal_deltas = torch.cat(full_bboxes_ls, dim=0)
        pred_cls = torch.cat(full_cls_ls, dim=0) #.unsqueeze(-1)
        
        predictions = class_logits, proposal_deltas
        proposals = full_proposals_ls
        pred_instances, _ = self.box_predictor.inference(pred_cls, predictions, proposals)
        pred_instances = self.forward_with_given_boxes(features, pred_instances)

        return pred_instances, {}

    def forward_with_given_boxes(self, features, instances):
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            instances (Instances):
                the same `Instances` object, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        if self.mask_on:
            features = [features[f] for f in self.in_features]
            x = self._shared_roi_transform(features, [x.pred_boxes for x in instances])
            return self.mask_head(x, instances)
        else:
            return instances




class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None

@ROI_HEADS_REGISTRY.register()
class CustomROIHeads(StandardROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictor']
        ret['box_predictor'] = CustomFastRCNNOutputLayers(
            cfg, ret['box_head'].output_shape)
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret

    def forward(self, images, features,support_box_features, proposals, targets=None):
        """
        enable debug
        """
        if not self.debug:
            del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses = self._forward_box(features,support_box_features, proposals)
            losses.update(self._forward_mask(features,support_box_features, proposals))
            losses.update(self._forward_keypoint(features,support_box_features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features,support_box_features, proposals)
            pred_instances = self.forward_with_given_boxes(features,support_box_features, pred_instances)
            if self.debug:
                from CenterNet2.centernet.modeling.debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(images[0].clone())],
                    pred_instances, proposals=proposals,
                    debug_show_name=self.debug_show_name)
            return pred_instances, {}

    
        
    
        
        
@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CascadeROIHeads):
    @classmethod
    def _init_box_head(self, cfg, input_shape):
        self.mult_proposal_score = cfg.MODEL.ROI_BOX_HEAD.MULT_PROPOSAL_SCORE
        ret = super()._init_box_head(cfg, input_shape)
        del ret['box_predictors']
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        box_predictors = []
        for box_head, bbox_reg_weights in zip(ret['box_heads'], cascade_bbox_reg_weights):
            box_predictors.append(
                CustomFastRCNNOutputLayers(
                    cfg, box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights)
                ))
        ret['box_predictors'] = box_predictors
        self.debug = cfg.DEBUG
        if self.debug:
            self.debug_show_name = cfg.DEBUG_SHOW_NAME
            self.save_debug = cfg.SAVE_DEBUG
            self.vis_thresh = cfg.VIS_THRESH
            self.pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
            self.pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(
                torch.device(cfg.MODEL.DEVICE)).view(3, 1, 1)
        return ret
    
    
    
    def _shared_roi_transform(self, features, boxes):
        x = self.box_pooler(features, boxes)
        return x
    
    
    # def roi_pooling(self, features, boxes):
    #     box_features = self.box_pooler(
    #         [features[f] for f in self.in_features], boxes
    #     )
    #     return box_features #feature_pooled
    
    def _forward_box(self, features, proposals, targets=None):
        """
        Add mult proposal scores at testing
        """
        if (not self.training) and self.mult_proposal_score:
            if len(proposals) > 0 and proposals[0].has('scores'):
                proposal_scores = [
                    p.get('scores') for p in proposals]
            else:
                proposal_scores = [
                    p.get('objectness_logits') for p in proposals]
        
        features = [features[f] for f in self.box_in_features]
        head_outputs = []  # (predictor, predictions, proposals)
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]
        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope("stage{}".format(stage)):
                    stage_losses = predictor.losses(predictions, proposals)
                losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
            return losses
        else:
            # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            
            if self.mult_proposal_score:
                scores = [(s * ps[:, None]) ** 0.5 \
                    for s, ps in zip(scores, proposal_scores)]

            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            
            return pred_instances

    def forward(self, images, features,support_box_features, proposals, targets=None):
        '''
        enable debug
        '''
        if not self.debug:
            del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            losses = self._forward_box(features,support_box_features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            # import pdb; pdb.set_trace()
            pred_instances = self._forward_box(features,support_box_features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            if self.debug:
                from CenterNet2.centernet.modeling.debug import debug_second_stage
                denormalizer = lambda x: x * self.pixel_std + self.pixel_mean
                debug_second_stage(
                    [denormalizer(x.clone()) for x in images],
                    pred_instances, proposals=proposals,
                    save_debug=self.save_debug,
                    debug_show_name=self.debug_show_name,
                    vis_thresh=self.vis_thresh)
            return pred_instances, {}

    ################################################################
    def _forward_box(self, features,support_box_features, proposals, targets=None):
            """
            Args:
                features, targets: the same as in
                    Same as in :meth:`ROIHeads.forward`.
                proposals (list[Instances]): the per-image object proposals with
                    their matching ground truth.
                    Each has fields "proposal_boxes", and "objectness_logits",
                    "gt_classes", "gt_boxes".
            """
            features = [features[f] for f in self.box_in_features] 
            prev_pred_boxes = None
            image_sizes = [x.image_size for x in proposals]
            head_outputs = []
            for k in range(self.num_cascade_stages):
                if k > 0:
                    # The output boxes of the previous stage are used to create the input
                    # proposals of the next stage.
                    proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                    if self.training:
                        proposals = self._match_and_label_boxes(proposals, k, targets)
                predictions = self._run_stage(features,support_box_features, proposals, k)
                prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
                head_outputs.append((self.box_predictor[k], predictions, proposals))

            if self.training:
                losses = {}
                storage = get_event_storage()
                for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                    with storage.name_scope("stage{}".format(stage)):
                        stage_losses = predictor.losses(predictions, proposals)
                    losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})
                return losses
            else:
                # Each is a list[Tensor] of length #image. Each tensor is Ri x (K+1)
                scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]

                # Average the scores across heads
                scores = [
                    sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                    for scores_per_image in zip(*scores_per_stage)
                ]
                # Use the boxes of the last head
                predictor, predictions, proposals = head_outputs[-1]
                boxes = predictor.predict_boxes(predictions, proposals)
                pred_instances, _ = fast_rcnn_inference(
                    boxes,
                    scores,
                    image_sizes,
                    predictor.test_score_thresh,
                    predictor.test_nms_thresh,
                    predictor.test_topk_per_image,
                )
                return pred_instances
        
    def _run_stage(self, features,support_box_features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        #feature [1,128,h,w]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        #box_features_12 = self.box_pooler1(features, [x.proposal_boxes for x in proposals])
        box_features_4 = self.box_pooler2(features, [x.proposal_boxes for x in proposals])
        # print(box_features.shape)
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        #box_features_12 = _ScaleGradient.apply(box_features_12, 1.0 / self.num_cascade_stages)
        box_features_4 = _ScaleGradient.apply(box_features_4, 1.0 / self.num_cascade_stages)

        support_box_features_ = support_box_features[0].mean(0,True)
        support_box_features_4 = support_box_features[1].mean(0,True)
        
        if self.attention_rpn:
            #glabal
            x_query_fc = self.avgpool_fc(box_features).squeeze(3).squeeze(2) #[128, 128, 7, 7]
            
            support_fc = self.avgpool_fc(support_box_features_).squeeze(3).squeeze(2).expand_as(x_query_fc)
            cat_fc = torch.cat((x_query_fc, support_fc), 1)
            out_fc = F.relu(self.fc_1(cat_fc), inplace=True)
            out_fc = F.relu(self.fc_2(out_fc), inplace=True)
            global_relation=out_fc
            #local
            x_query_cor = self.conv_cor(box_features)
            support_cor = self.conv_cor(support_box_features_)
            local_relation = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=256), inplace=True).squeeze(3).squeeze(2)
            #patch
            support_box_features_ = support_box_features_.expand_as(box_features)
            x = torch.cat((box_features, support_box_features_), 1)
            x = F.relu(self.conv_1(x), inplace=True) # 5x5
            x = self.avgpool(x)
            x = F.relu(self.conv_2(x), inplace=True) # 3x3
            x = F.relu(self.conv_3(x), inplace=True) # 3x3
            x = self.avgpool(x) # 1x1
            patch_relation = x.squeeze(3).squeeze(2)
        
        if self.head_cnn:
            
            
            
            support_box_features_ = support_box_features_.expand_as(box_features)
            attn_8 = self.conv3(torch.cat((box_features,support_box_features_),1))+torch.cat((self.conv1(box_features),self.conv2(support_box_features_)),1) 
            attn_8 =self.box_head[stage](attn_8)
            # box_features =self.box_head[stage](box_features_8)
            #4x4
            
            support_box_features_4 = support_box_features_4.expand_as(box_features_4)
            
            attn_4 = self.conv3(torch.cat((box_features_4,support_box_features_4),1))+torch.cat((self.conv1(box_features_4),self.conv2(support_box_features_4)),1) 
            
            attn_4 =F.relu(self.fc2(attn_4.flatten(1)))

            # print(attn_4.shape)
            # print(attn_8.shape)
            cls_attn = F.relu(self.fc3(torch.cat((attn_4,attn_8),1)))
            
            # box_features = self.box_head[stage](box_features)
        else:
            box_features = self.box_head[stage](box_features)
        
        # if self.attentionrpn:
        #     #global
            
        #     x_query_fc = self.avgpool_fc(box_features) #[1, 2048, 7, 7]
        #     support_fc = self.avgpool_fc(support_box_features).expand_as(x_query_fc)
        #     cat_fc = torch.cat((x_query_fc, support_fc), 1)
        #     cat_fc =self.conv3(cat_fc)
            
        #     global_relation = self.box_head[stage](cat_fc)
        #     #local
        #     x_query_cor = self.conv_cor(box_features)
        #     support_cor = self.conv_cor(support_box_features)
        #     local_relation = F.relu(F.conv2d(x_query_cor, support_cor.permute(1,0,2,3), groups=128), inplace=True).squeeze(3).squeeze(2)
            
        #     #patch
        #     support_box_features = support_box_features.expand_as(box_features)
        #     x = torch.cat((box_features, support_box_features), 1)
        #     x = F.relu(self.conv_1(x), inplace=True) # 5x5
        #     x = self.avgpool(x)
        #     x = F.relu(self.conv_2(x), inplace=True) # 3x3
        #     x = F.relu(self.conv_3(x), inplace=True) # 3x3
        #     x = self.avgpool(x) # 1x1
        #     patch_relation = x.mean(dim=[2, 3], keepdim=True).squeeze(3).squeeze(2)
            
        
        
        return self.box_predictor[stage](attn_8)
