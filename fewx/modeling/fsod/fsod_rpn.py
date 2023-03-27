# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F
from fvcore.nn import smooth_l1_loss
from torch import nn

from detectron2.config import configurable
from detectron2.layers import ShapeSpec, cat
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.utils.events import get_event_storage
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.utils.registry import Registry
from detectron2.utils.comm import get_world_size
from detectron2.modeling.anchor_generator import build_anchor_generator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.sampling import subsample_labels
from detectron2.modeling.proposal_generator.build import PROPOSAL_GENERATOR_REGISTRY
from detectron2.modeling.proposal_generator.proposal_utils import find_top_rpn_proposals

from CenterNet2.centernet.modeling.layers.heatmap_focal_loss import heatmap_focal_loss_jit
from CenterNet2.centernet.modeling.layers.heatmap_focal_loss import binary_heatmap_focal_loss_jit
from CenterNet2.centernet.modeling.layers.iou_loss import IOULoss
from CenterNet2.centernet.modeling.layers.ml_nms import ml_nms
from CenterNet2.centernet.modeling.debug import debug_train, debug_test
from CenterNet2.centernet.modeling.dense_heads.utils import reduce_sum, _transpose
from CenterNet2.centernet.modeling.dense_heads.centernet_head import CenterNetHead
RPN_HEAD_REGISTRY = Registry("RPN_HEAD")
RPN_HEAD_REGISTRY.__doc__ = """
Registry for RPN heads, which take feature maps and perform
objectness classification and bounding box regression for anchors.

The registered object will be called with `obj(cfg, input_shape)`.
The call should return a `nn.Module` object.
"""


"""
Shape shorthand in this module:

    N: number of images in the minibatch
    L: number of feature maps per image on which RPN is run
    A: number of cell anchors (must be the same for all feature maps)
    Hi, Wi: height and width of the i-th feature map
    B: size of the box parameterization

Naming convention:

    objectness: refers to the binary classification of an anchor as object vs. not object.

    deltas: refers to the 4-d (dx, dy, dw, dh) deltas that parameterize the box2box
    transform (see :class:`box_regression.Box2BoxTransform`), or 5d for rotated boxes.

    pred_objectness_logits: predicted objectness scores in [-inf, +inf]; use
        sigmoid(pred_objectness_logits) to estimate P(object).

    gt_labels: ground-truth binary classification labels for objectness

    pred_anchor_deltas: predicted box2box transform deltas

    gt_anchor_deltas: ground-truth box2box transform deltas
"""


def build_rpn_head(cfg, input_shape):
    """
    Build an RPN head defined by `cfg.MODEL.RPN.HEAD_NAME`.
    """
    name = cfg.MODEL.RPN.HEAD_NAME
    return RPN_HEAD_REGISTRY.get(name)(cfg, input_shape)


@RPN_HEAD_REGISTRY.register()
class StandardRPNHead(nn.Module):
    """
    Standard RPN classification and regression heads described in :paper:`Faster R-CNN`.
    Uses a 3x3 conv to produce a shared hidden state from which one 1x1 conv predicts
    objectness logits for each anchor and a second 1x1 conv predicts bounding-box deltas
    specifying how to deform each anchor into an object proposal.
    """

    @configurable
    def __init__(self, *, in_channels: int, num_anchors: int, box_dim: int = 4):
        """
        NOTE: this interface is experimental.

        Args:
            in_channels (int): number of input feature channels. When using multiple
                input features, they must have the same number of channels.
            num_anchors (int): number of anchors to predict for *each spatial position*
                on the feature map. The total number of anchors for each
                feature map will be `num_anchors * H * W`.
            box_dim (int): dimension of a box, which is also the number of box regression
                predictions to make for each anchor. An axis aligned box has
                box_dim=4, while a rotated box has box_dim=5.
        """
        super().__init__()
        # 3x3 conv for the hidden representation
        self.conv = nn.Conv2d(int(in_channels), int(192), kernel_size=3, stride=1, padding=1)
        # 1x1 conv for predicting objectness logits
        self.objectness_logits = nn.Conv2d(int(192), num_anchors, kernel_size=1, stride=1)
        # 1x1 conv for predicting box2box transform deltas
        self.anchor_deltas = nn.Conv2d(int(192), num_anchors * box_dim, kernel_size=1, stride=1)

        for l in [self.conv, self.objectness_logits, self.anchor_deltas]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)

    @classmethod
    def from_config(cls, cfg, input_shape):
        # Standard RPN is shared across levels:
        in_channels = [s.channels for s in input_shape]
        assert len(set(in_channels)) == 1, "Each level must have the same channel!"
        in_channels = in_channels[0]

        # RPNHead should take the same input as anchor generator
        # NOTE: it assumes that creating an anchor generator does not have unwanted side effect.
        anchor_generator = build_anchor_generator(cfg, input_shape)
        num_anchors = anchor_generator.num_anchors
        box_dim = anchor_generator.box_dim
        assert (
            len(set(num_anchors)) == 1
        ), "Each level must have the same number of anchors per spatial position"
        return {"in_channels": in_channels, "num_anchors": num_anchors[0], "box_dim": box_dim}

    def forward(self, features: List[torch.Tensor]):
        """
        Args:
            features (list[Tensor]): list of feature maps

        Returns:
            list[Tensor]: A list of L elements.
                Element i is a tensor of shape (N, A, Hi, Wi) representing
                the predicted objectness logits for all anchors. A is the number of cell anchors.
            list[Tensor]: A list of L elements. Element i is a tensor of shape
                (N, A*box_dim, Hi, Wi) representing the predicted "deltas" used to transform anchors
                to proposals.
        """
        pred_objectness_logits = []
        pred_anchor_deltas = []
        for x in features:
            t = F.relu(self.conv(x))
            pred_objectness_logits.append(self.objectness_logits(t))
            pred_anchor_deltas.append(self.anchor_deltas(t))
        return pred_objectness_logits, pred_anchor_deltas


@PROPOSAL_GENERATOR_REGISTRY.register()
class FsodRPN(nn.Module):
    """
    Region Proposal Network, introduced by :paper:`Faster R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        in_features: List[str],
        head: nn.Module,
        anchor_generator: nn.Module,
        anchor_matcher: Matcher,
        box2box_transform: Box2BoxTransform,
        batch_size_per_image: int,
        positive_fraction: float,
        pre_nms_topk: Tuple[float, float],
        post_nms_topk: Tuple[float, float],
        nms_thresh: float = 0.7,
        min_box_size: float = 0.0,
        anchor_boundary_thresh: float = -1.0,
        loss_weight: float = 1.0,
        smooth_l1_beta: float = 0.0
    ):
        """
        NOTE: this interface is experimental.

        Args:
            in_features (list[str]): list of names of input features to use
            head (nn.Module): a module that predicts logits and regression deltas
                for each level from a list of per-level features
            anchor_generator (nn.Module): a module that creates anchors from a
                list of features. Usually an instance of :class:`AnchorGenerator`
            anchor_matcher (Matcher): label the anchors by matching them with ground truth.
            box2box_transform (Box2BoxTransform): defines the transform from anchors boxes to
                instance boxes
            batch_size_per_image (int): number of anchors per image to sample for training
            positive_fraction (float): fraction of foreground anchors to sample for training
            pre_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select before NMS, in
                training and testing.
            post_nms_topk (tuple[float]): (train, test) that represents the
                number of top k proposals to select after NMS, in
                training and testing.
            nms_thresh (float): NMS threshold used to de-duplicate the predicted proposals
            min_box_size (float): remove proposal boxes with any side smaller than this threshold,
                in the unit of input image pixels
            anchor_boundary_thresh (float): legacy option
            loss_weight (float): weight to be multiplied to the loss
            smooth_l1_beta (float): beta parameter for the smooth L1
                regression loss. Default to use L1 loss.
        """
        super().__init__()
        self.in_features = in_features
        self.rpn_head = head
        self.anchor_generator = anchor_generator
        self.anchor_matcher = anchor_matcher
        self.box2box_transform = box2box_transform
        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        # Map from self.training state to train/test settings
        self.pre_nms_topk = {True: pre_nms_topk[0], False: pre_nms_topk[1]}
        self.post_nms_topk = {True: post_nms_topk[0], False: post_nms_topk[1]}
        self.nms_thresh = nms_thresh
        self.min_box_size = min_box_size
        self.anchor_boundary_thresh = anchor_boundary_thresh
        self.loss_weight = loss_weight
        self.smooth_l1_beta = smooth_l1_beta

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        in_features = cfg.MODEL.RPN.IN_FEATURES
        ret = {
            "in_features": in_features,
            "min_box_size": cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE,
            "nms_thresh": cfg.MODEL.RPN.NMS_THRESH,
            "batch_size_per_image": cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE,
            "positive_fraction": cfg.MODEL.RPN.POSITIVE_FRACTION,
            "smooth_l1_beta": cfg.MODEL.RPN.SMOOTH_L1_BETA,
            "loss_weight": cfg.MODEL.RPN.LOSS_WEIGHT,
            "anchor_boundary_thresh": cfg.MODEL.RPN.BOUNDARY_THRESH,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.RPN.BBOX_REG_WEIGHTS),
        }

        ret["pre_nms_topk"] = (cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST)
        ret["post_nms_topk"] = (cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST)

        ret["anchor_generator"] = build_anchor_generator(cfg, [input_shape[f] for f in in_features])
        ret["anchor_matcher"] = Matcher(
            cfg.MODEL.RPN.IOU_THRESHOLDS, cfg.MODEL.RPN.IOU_LABELS, allow_low_quality_matches=True
        )
        ret["head"] = build_rpn_head(cfg, [input_shape[f] for f in in_features])
        return ret

    def _subsample_labels(self, label):
        """
        Randomly sample a subset of positive and negative examples, and overwrite
        the label vector to the ignore value (-1) for all elements that are not
        included in the sample.

        Args:
            labels (Tensor): a vector of -1, 0, 1. Will be modified in-place and returned.
        """
        pos_idx, neg_idx = subsample_labels(
            label, self.batch_size_per_image, self.positive_fraction, 0
        )
        # Fill with the ignore label (-1), then set positive and negative labels
        label.fill_(-1)
        label.scatter_(0, pos_idx, 1)
        label.scatter_(0, neg_idx, 0)
        return label

    @torch.no_grad()
    def label_and_sample_anchors(self, anchors: List[Boxes], gt_instances: List[Instances]):
        """
        Args:
            anchors (list[Boxes]): anchors for each feature map.
            gt_instances: the ground-truth instances for each image.

        Returns:
            list[Tensor]:
                List of #img tensors. i-th element is a vector of labels whose length is
                the total number of anchors across all feature maps R = sum(Hi * Wi * A).
                Label values are in {-1, 0, 1}, with meanings: -1 = ignore; 0 = negative
                class; 1 = positive class.
            list[Tensor]:
                i-th element is a Rx4 tensor. The values are the matched gt boxes for each
                anchor. Values are undefined for those anchors not labeled as 1.
        """
        anchors = Boxes.cat(anchors)

        gt_boxes = [x.gt_boxes for x in gt_instances]
        image_sizes = [x.image_size for x in gt_instances]
        del gt_instances

        gt_labels = []
        matched_gt_boxes = []
        for image_size_i, gt_boxes_i in zip(image_sizes, gt_boxes):
            """
            image_size_i: (h, w) for the i-th image
            gt_boxes_i: ground-truth boxes for i-th image
            """

            match_quality_matrix = retry_if_cuda_oom(pairwise_iou)(gt_boxes_i, anchors)
            matched_idxs, gt_labels_i = retry_if_cuda_oom(self.anchor_matcher)(match_quality_matrix)
            # Matching is memory-expensive and may result in CPU tensors. But the result is small
            gt_labels_i = gt_labels_i.to(device=gt_boxes_i.device)
            del match_quality_matrix

            if self.anchor_boundary_thresh >= 0:
                # Discard anchors that go out of the boundaries of the image
                # NOTE: This is legacy functionality that is turned off by default in Detectron2
                anchors_inside_image = anchors.inside_box(image_size_i, self.anchor_boundary_thresh)
                gt_labels_i[~anchors_inside_image] = -1

            # A vector of labels (-1, 0, 1) for each anchor
            gt_labels_i = self._subsample_labels(gt_labels_i)

            if len(gt_boxes_i) == 0:
                # These values won't be used anyway since the anchor is labeled as background
                matched_gt_boxes_i = torch.zeros_like(anchors.tensor)
            else:
                # TODO wasted indexing computation for ignored boxes
                matched_gt_boxes_i = gt_boxes_i[matched_idxs].tensor

            gt_labels.append(gt_labels_i)  # N,AHW
            matched_gt_boxes.append(matched_gt_boxes_i)
        return gt_labels, matched_gt_boxes

    def losses(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        gt_labels: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        gt_boxes,
    ):
        """
        Return the losses from a set of RPN predictions and their associated ground-truth.

        Args:
            anchors (list[Boxes or RotatedBoxes]): anchors for each feature map, each
                has shape (Hi*Wi*A, B), where B is box dimension (4 or 5).
            pred_objectness_logits (list[Tensor]): A list of L elements.
                Element i is a tensor of shape (N, Hi*Wi*A) representing
                the predicted objectness logits for all anchors.
            gt_labels (list[Tensor]): Output of :meth:`label_and_sample_anchors`.
            pred_anchor_deltas (list[Tensor]): A list of L elements. Element i is a tensor of shape
                (N, Hi*Wi*A, 4 or 5) representing the predicted "deltas" used to transform anchors
                to proposals.
            gt_boxes (list[Boxes or RotatedBoxes]): Output of :meth:`label_and_sample_anchors`.

        Returns:
            dict[loss name -> loss value]: A dict mapping from loss name to loss value.
                Loss names are: `loss_rpn_cls` for objectness classification and
                `loss_rpn_loc` for proposal localization.
        """
        num_images = len(gt_labels)
        gt_labels = torch.stack(gt_labels)  # (N, sum(Hi*Wi*Ai))
        anchors = type(anchors[0]).cat(anchors).tensor  # Ax(4 or 5)
        #print(anchors.shape, gt_boxes[0].shape, len(gt_boxes))
        gt_anchor_deltas = [self.box2box_transform.get_deltas(anchors, k) for k in gt_boxes]
        gt_anchor_deltas = torch.stack(gt_anchor_deltas)  # (N, sum(Hi*Wi*Ai), 4 or 5)

        # Log the number of positive/negative anchors per-image that's used in training
        pos_mask = gt_labels == 1
        num_pos_anchors = pos_mask.sum().item()
        num_neg_anchors = (gt_labels == 0).sum().item()
        storage = get_event_storage()
        storage.put_scalar("rpn/num_pos_anchors", num_pos_anchors / num_images)
        storage.put_scalar("rpn/num_neg_anchors", num_neg_anchors / num_images)

        localization_loss = smooth_l1_loss(
            cat(pred_anchor_deltas, dim=1)[pos_mask],
            gt_anchor_deltas[pos_mask],
            self.smooth_l1_beta,
            reduction="sum",
        )
        valid_mask = gt_labels >= 0
        objectness_loss = F.binary_cross_entropy_with_logits(
            cat(pred_objectness_logits, dim=1)[valid_mask],
            gt_labels[valid_mask].to(torch.float32),
            reduction="sum",
        )
        normalizer = self.batch_size_per_image * num_images
        return {
            "loss_rpn_cls": objectness_loss / normalizer,
            "loss_rpn_loc": localization_loss / normalizer,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        gt_instances: Optional[Instances] = None,
    ):
        """
        Args:
            images (ImageList): input images of length `N`
            features (dict[str, Tensor]): input data as a mapping from feature
                map name to tensor. Axis 0 represents the number of images `N` in
                the input data; axes 1-3 are channels, height, and width, which may
                vary between feature maps (e.g., if a feature pyramid is used).
            gt_instances (list[Instances], optional): a length `N` list of `Instances`s.
                Each `Instances` stores ground-truth instances for the corresponding image.

        Returns:
            proposals: list[Instances]: contains fields "proposal_boxes", "objectness_logits"
            loss: dict[Tensor] or None
        """
        features = [features[f] for f in self.in_features]
        anchors = self.anchor_generator(features)

        pred_objectness_logits, pred_anchor_deltas = self.rpn_head(features)
        # Transpose the Hi*Wi*A dimension to the middle:
        pred_objectness_logits = [
            # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
            score.permute(0, 2, 3, 1).flatten(1)
            for score in pred_objectness_logits
        ]
        pred_anchor_deltas = [
            # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
            x.view(x.shape[0], -1, self.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
            .permute(0, 3, 4, 1, 2)
            .flatten(1, -2)
            for x in pred_anchor_deltas
        ]

        if self.training:
            gt_labels, gt_boxes = self.label_and_sample_anchors(anchors, gt_instances)
            #losses = self.losses(
            #    anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes
            #)
            #losses = {k: v * self.loss_weight for k, v in losses.items()}
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )

            return proposals, anchors, pred_objectness_logits, gt_labels, pred_anchor_deltas, gt_boxes #, losses
        else:
            losses = {}
            proposals = self.predict_proposals(
                anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
            )
            return proposals, losses

    @torch.no_grad()
    def predict_proposals(
        self,
        anchors,
        pred_objectness_logits: List[torch.Tensor],
        pred_anchor_deltas: List[torch.Tensor],
        image_sizes: List[Tuple[int, int]],
    ):
        """
        Decode all the predicted box regression deltas to proposals. Find the top proposals
        by applying NMS and removing boxes that are too small.

        Returns:
            proposals (list[Instances]): list of N Instances. The i-th Instances
                stores post_nms_topk object proposals for image i, sorted by their
                objectness score in descending order.
        """
        # The proposals are treated as fixed for approximate joint training with roi heads.
        # This approach ignores the derivative w.r.t. the proposal boxes’ coordinates that
        # are also network responses, so is approximate.
        pred_proposals = self._decode_proposals(anchors, pred_anchor_deltas)
        return find_top_rpn_proposals(
            pred_proposals,
            pred_objectness_logits,
            image_sizes,
            self.nms_thresh,
            self.pre_nms_topk[self.training],
            self.post_nms_topk[self.training],
            self.min_box_size,
            self.training,
        )

    def _decode_proposals(self, anchors, pred_anchor_deltas: List[torch.Tensor]):
        """
        Transform anchors into proposals by applying the predicted anchor deltas.

        Returns:
            proposals (list[Tensor]): A list of L tensors. Tensor i has shape
                (N, Hi*Wi*A, B)
        """
        N = pred_anchor_deltas[0].shape[0]
        proposals = []
        # For each feature map
        for anchors_i, pred_anchor_deltas_i in zip(anchors, pred_anchor_deltas):
            B = anchors_i.tensor.size(1)
            pred_anchor_deltas_i = pred_anchor_deltas_i.reshape(-1, B)
            # Expand anchors to shape (N*Hi*Wi*A, B)
            anchors_i = anchors_i.tensor.unsqueeze(0).expand(N, -1, -1).reshape(-1, B)
            proposals_i = self.box2box_transform.apply_deltas(pred_anchor_deltas_i, anchors_i)
            # Append feature map proposals with shape (N, Hi*Wi*A, B)
            proposals.append(proposals_i.view(N, -1, B))
        return proposals


INF = 100000000
@PROPOSAL_GENERATOR_REGISTRY.register()
class CenterNet(nn.Module):
    @configurable
    def __init__(self, 
        # input_shape: Dict[str, ShapeSpec],
        in_channels=256,
        *,
        num_classes=80,
        in_features=("p3", "p4", "p5", "p6", "p7"),
        strides=(8, 16, 32, 64, 128),
        score_thresh=0.05,
        hm_min_overlap=0.8,
        loc_loss_type='giou',
        min_radius=4,
        hm_focal_alpha=0.25,
        hm_focal_beta=4,
        loss_gamma=2.0,
        reg_weight=2.0,
        not_norm_reg=True,
        with_agn_hm=False,
        only_proposal=False,
        as_proposal=False,
        not_nms=False,
        pos_weight=1.,
        neg_weight=1.,
        sigmoid_clamp=1e-4,
        ignore_high_fp=-1.,
        center_nms=False,
        sizes_of_interest=[[0,80],[64,160],[128,320],[256,640],[512,10000000]],
        more_pos=False,
        more_pos_thresh=0.2,
        more_pos_topk=9,
        pre_nms_topk_train=1000,
        pre_nms_topk_test=1000,
        post_nms_topk_train=100,
        post_nms_topk_test=100,
        nms_thresh_train=0.6,
        nms_thresh_test=0.6,
        no_reduce=False,
        not_clamp_box=False,
        debug=False,
        vis_thresh=0.5,
        pixel_mean=[103.530,116.280,123.675],
        pixel_std=[1.0,1.0,1.0],
        device='cuda',
        centernet_head=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.in_features = in_features
        self.strides = strides
        self.score_thresh = score_thresh
        self.min_radius = min_radius
        self.hm_focal_alpha = hm_focal_alpha
        self.hm_focal_beta = hm_focal_beta
        self.loss_gamma = loss_gamma
        self.reg_weight = reg_weight
        self.not_norm_reg = not_norm_reg
        self.with_agn_hm = with_agn_hm
        self.only_proposal = only_proposal
        self.as_proposal = as_proposal
        self.not_nms = not_nms
        self.pos_weight = pos_weight
        self.neg_weight = neg_weight
        self.sigmoid_clamp = sigmoid_clamp
        self.ignore_high_fp = ignore_high_fp
        self.center_nms = center_nms
        self.sizes_of_interest = sizes_of_interest
        self.more_pos = more_pos
        self.more_pos_thresh = more_pos_thresh
        self.more_pos_topk = more_pos_topk
        self.pre_nms_topk_train = pre_nms_topk_train
        self.pre_nms_topk_test = pre_nms_topk_test
        self.post_nms_topk_train = post_nms_topk_train
        self.post_nms_topk_test = post_nms_topk_test
        self.nms_thresh_train = nms_thresh_train
        self.nms_thresh_test = nms_thresh_test
        self.no_reduce = no_reduce
        self.not_clamp_box = not_clamp_box
        
        self.debug = debug
        self.vis_thresh = vis_thresh
        if self.center_nms:
            self.not_nms = True
        self.iou_loss = IOULoss(loc_loss_type)
        assert (not self.only_proposal) or self.with_agn_hm
        # delta for rendering heatmap
        self.delta = (1 - hm_min_overlap) / (1 + hm_min_overlap)
        if centernet_head is None:
            self.centernet_head = CenterNetHead(
                in_channels=in_channels,
                num_levels=len(in_features),
                with_agn_hm=with_agn_hm,
                only_proposal=only_proposal)
        else:
            self.centernet_head = centernet_head
        if self.debug:
            pixel_mean = torch.Tensor(pixel_mean).to(
                torch.device(device)).view(3, 1, 1)
            pixel_std = torch.Tensor(pixel_std).to(
                torch.device(device)).view(3, 1, 1)
            self.denormalizer = lambda x: x * pixel_std + pixel_mean

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {
            # 'input_shape': input_shape,
            'in_channels': input_shape[
                cfg.MODEL.CENTERNET.IN_FEATURES[0]].channels,
            'num_classes': cfg.MODEL.CENTERNET.NUM_CLASSES,
            'in_features': cfg.MODEL.CENTERNET.IN_FEATURES,
            'strides': cfg.MODEL.CENTERNET.FPN_STRIDES,
            'score_thresh': cfg.MODEL.CENTERNET.INFERENCE_TH,
            'loc_loss_type': cfg.MODEL.CENTERNET.LOC_LOSS_TYPE,
            'hm_min_overlap': cfg.MODEL.CENTERNET.HM_MIN_OVERLAP,
            'min_radius': cfg.MODEL.CENTERNET.MIN_RADIUS,
            'hm_focal_alpha': cfg.MODEL.CENTERNET.HM_FOCAL_ALPHA,
            'hm_focal_beta': cfg.MODEL.CENTERNET.HM_FOCAL_BETA,
            'loss_gamma': cfg.MODEL.CENTERNET.LOSS_GAMMA,
            'reg_weight': cfg.MODEL.CENTERNET.REG_WEIGHT,
            'not_norm_reg': cfg.MODEL.CENTERNET.NOT_NORM_REG,
            'with_agn_hm': cfg.MODEL.CENTERNET.WITH_AGN_HM,
            'only_proposal': cfg.MODEL.CENTERNET.ONLY_PROPOSAL,
            'as_proposal': cfg.MODEL.CENTERNET.AS_PROPOSAL,
            'not_nms': cfg.MODEL.CENTERNET.NOT_NMS,
            'pos_weight': cfg.MODEL.CENTERNET.POS_WEIGHT,
            'neg_weight': cfg.MODEL.CENTERNET.NEG_WEIGHT,
            'sigmoid_clamp': cfg.MODEL.CENTERNET.SIGMOID_CLAMP,
            'ignore_high_fp': cfg.MODEL.CENTERNET.IGNORE_HIGH_FP,
            'center_nms': cfg.MODEL.CENTERNET.CENTER_NMS,
            'sizes_of_interest': cfg.MODEL.CENTERNET.SOI,
            'more_pos': cfg.MODEL.CENTERNET.MORE_POS,
            'more_pos_thresh': cfg.MODEL.CENTERNET.MORE_POS_THRESH,
            'more_pos_topk': cfg.MODEL.CENTERNET.MORE_POS_TOPK,
            'pre_nms_topk_train': cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TRAIN,
            'pre_nms_topk_test': cfg.MODEL.CENTERNET.PRE_NMS_TOPK_TEST,
            'post_nms_topk_train': cfg.MODEL.CENTERNET.POST_NMS_TOPK_TRAIN,
            'post_nms_topk_test': cfg.MODEL.CENTERNET.POST_NMS_TOPK_TEST,
            'nms_thresh_train': cfg.MODEL.CENTERNET.NMS_TH_TRAIN,
            'nms_thresh_test': cfg.MODEL.CENTERNET.NMS_TH_TEST,
            'no_reduce': cfg.MODEL.CENTERNET.NO_REDUCE,
            'not_clamp_box': cfg.INPUT.NOT_CLAMP_BOX,
            'debug': cfg.DEBUG,
            'vis_thresh': cfg.VIS_THRESH,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'device': cfg.MODEL.DEVICE,
            'centernet_head': CenterNetHead(
                cfg, [input_shape[f] for f in cfg.MODEL.CENTERNET.IN_FEATURES]),
        }
        return ret


    def forward(self, images, features_dict, gt_instances):
        features = [features_dict[f] for f in self.in_features]
        clss_per_level, reg_pred_per_level, agn_hm_pred_per_level = \
            self.centernet_head(features)
        grids = self.compute_grids(features)
        shapes_per_level = grids[0].new_tensor(
                    [(x.shape[2], x.shape[3]) for x in reg_pred_per_level])
        
        if not self.training:
            return self.inference(
                images, clss_per_level, reg_pred_per_level, 
                agn_hm_pred_per_level, grids)
        else:
            pos_inds, labels, reg_targets, flattened_hms = \
                self._get_ground_truth(
                    grids, shapes_per_level, gt_instances)
            # logits_pred: M x F, reg_pred: M x 4, agn_hm_pred: M
            logits_pred, reg_pred, agn_hm_pred = self._flatten_outputs(
                clss_per_level, reg_pred_per_level, agn_hm_pred_per_level)

            if self.more_pos:
                # add more pixels as positive if \
                #   1. they are within the center3x3 region of an object
                #   2. their regression losses are small (<self.more_pos_thresh)
                pos_inds, labels = self._add_more_pos(
                    reg_pred, gt_instances, shapes_per_level)
            
            losses = self.losses(
                pos_inds, labels, reg_targets, flattened_hms,
                logits_pred, reg_pred, agn_hm_pred)
            
            proposals = None
            if self.only_proposal:
                agn_hm_pred_per_level = [x.sigmoid() for x in agn_hm_pred_per_level]
                proposals = self.predict_instances(
                    grids, agn_hm_pred_per_level, reg_pred_per_level, 
                    images.image_sizes, [None for _ in agn_hm_pred_per_level])
            elif self.as_proposal: # category specific bbox as agnostic proposals
                clss_per_level = [x.sigmoid() for x in clss_per_level]
                proposals = self.predict_instances(
                    grids, clss_per_level, reg_pred_per_level, 
                    images.image_sizes, agn_hm_pred_per_level)
            if self.only_proposal or self.as_proposal:
                for p in range(len(proposals)):
                    proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                    proposals[p].objectness_logits = proposals[p].get('scores')
                    proposals[p].remove('pred_boxes')
                    proposals[p].remove('scores')
                    proposals[p].remove('pred_classes')

            if self.debug:
                debug_train(
                    [self.denormalizer(x) for x in images], 
                    gt_instances, flattened_hms, reg_targets, 
                    labels, pos_inds, shapes_per_level, grids, self.strides)
            return proposals, losses


    def losses(
        self, pos_inds, labels, reg_targets, flattened_hms,
        logits_pred, reg_pred, agn_hm_pred):
        '''
        Inputs:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C
            logits_pred: M x C
            reg_pred: M x 4
            agn_hm_pred: M x 1 or None
            N: number of positive locations in all images
            M: number of pixels from all FPN levels
            C: number of classes
        '''
        # assert (torch.isfinite(reg_pred).all().item())
        num_pos_local = pos_inds.numel()
        num_gpus = get_world_size()
        if self.no_reduce:
            total_num_pos = num_pos_local * num_gpus
        else:
            total_num_pos = reduce_sum(
                pos_inds.new_tensor([num_pos_local])).item()
        num_pos_avg = max(total_num_pos / num_gpus, 1.0)

        losses = {}
        if not self.only_proposal:
            pos_loss, neg_loss = heatmap_focal_loss_jit(
                logits_pred.float(), flattened_hms.float(), pos_inds, labels,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma, 
                reduction='sum',
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            pos_loss = self.pos_weight * pos_loss / num_pos_avg
            neg_loss = self.neg_weight * neg_loss / num_pos_avg
            losses['loss_centernet_pos'] = pos_loss
            losses['loss_centernet_neg'] = neg_loss
        
        reg_inds = torch.nonzero(reg_targets.max(dim=1)[0] >= 0).squeeze(1)
        reg_pred = reg_pred[reg_inds]
        reg_targets_pos = reg_targets[reg_inds]
        reg_weight_map = flattened_hms.max(dim=1)[0]
        reg_weight_map = reg_weight_map[reg_inds]
        reg_weight_map = reg_weight_map * 0 + 1 \
            if self.not_norm_reg else reg_weight_map
        if self.no_reduce:
            reg_norm = max(reg_weight_map.sum(), 1)
        else:
            reg_norm = max(reduce_sum(reg_weight_map.sum()).item() / num_gpus, 1)
        
        reg_loss = self.reg_weight * self.iou_loss(
            reg_pred, reg_targets_pos, reg_weight_map,
            reduction='sum') / reg_norm
        losses['loss_centernet_loc'] = reg_loss

        if self.with_agn_hm:
            cat_agn_heatmap = flattened_hms.max(dim=1)[0] # M
            agn_pos_loss, agn_neg_loss = binary_heatmap_focal_loss_jit(
                agn_hm_pred.float(), cat_agn_heatmap.float(), pos_inds,
                alpha=self.hm_focal_alpha, 
                beta=self.hm_focal_beta, 
                gamma=self.loss_gamma,
                sigmoid_clamp=self.sigmoid_clamp,
                ignore_high_fp=self.ignore_high_fp,
            )
            agn_pos_loss = self.pos_weight * agn_pos_loss / num_pos_avg
            agn_neg_loss = self.neg_weight * agn_neg_loss / num_pos_avg
            losses['loss_centernet_agn_pos'] = agn_pos_loss
            losses['loss_centernet_agn_neg'] = agn_neg_loss
    
        if self.debug:
            print('losses', losses)
            print('total_num_pos', total_num_pos)
        return losses


    def compute_grids(self, features):
        grids = []
        for level, feature in enumerate(features):
            h, w = feature.size()[-2:]
            shifts_x = torch.arange(
                0, w * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shifts_y = torch.arange(
                0, h * self.strides[level], 
                step=self.strides[level],
                dtype=torch.float32, device=feature.device)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)
            grids_per_level = torch.stack((shift_x, shift_y), dim=1) + \
                self.strides[level] // 2
            grids.append(grids_per_level)
        return grids


    def _get_ground_truth(self, grids, shapes_per_level, gt_instances):
        '''
        Input:
            grids: list of tensors [(hl x wl, 2)]_l
            shapes_per_level: list of tuples L x 2:
            gt_instances: gt instances
        Retuen:
            pos_inds: N
            labels: N
            reg_targets: M x 4
            flattened_hms: M x C or M x 1
            N: number of objects in all images
            M: number of pixels from all FPN levels
        '''

        # get positive pixel index
        if not self.more_pos:
            pos_inds, labels = self._get_label_inds(
                gt_instances, shapes_per_level) 
        else:
            pos_inds, labels = None, None
        heatmap_channels = self.num_classes
        L = len(grids)
        num_loc_list = [len(loc) for loc in grids]
        strides = torch.cat([
            shapes_per_level.new_ones(num_loc_list[l]) * self.strides[l] \
            for l in range(L)]).float() # M
        reg_size_ranges = torch.cat([
            shapes_per_level.new_tensor(self.sizes_of_interest[l]).float().view(
            1, 2).expand(num_loc_list[l], 2) for l in range(L)]) # M x 2
        grids = torch.cat(grids, dim=0) # M x 2
        M = grids.shape[0]

        reg_targets = []
        flattened_hms = []
        for i in range(len(gt_instances)): # images
            boxes = gt_instances[i].gt_boxes.tensor # N x 4
            area = gt_instances[i].gt_boxes.area() # N
            gt_classes = gt_instances[i].gt_classes # N in [0, self.num_classes]

            N = boxes.shape[0]
            if N == 0:
                reg_targets.append(grids.new_zeros((M, 4)) - INF)
                flattened_hms.append(
                    grids.new_zeros((
                        M, 1 if self.only_proposal else heatmap_channels)))
                continue
            
            l = grids[:, 0].view(M, 1) - boxes[:, 0].view(1, N) # M x N
            t = grids[:, 1].view(M, 1) - boxes[:, 1].view(1, N) # M x N
            r = boxes[:, 2].view(1, N) - grids[:, 0].view(M, 1) # M x N
            b = boxes[:, 3].view(1, N) - grids[:, 1].view(M, 1) # M x N
            reg_target = torch.stack([l, t, r, b], dim=2) # M x N x 4

            centers = ((boxes[:, [0, 1]] + boxes[:, [2, 3]]) / 2) # N x 2
            centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
            strides_expanded = strides.view(M, 1, 1).expand(M, N, 2)
            centers_discret = ((centers_expanded / strides_expanded).int() * \
                strides_expanded).float() + strides_expanded / 2 # M x N x 2
            
            is_peak = (((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_discret) ** 2).sum(dim=2) == 0) # M x N
            is_in_boxes = reg_target.min(dim=2)[0] > 0 # M x N
            is_center3x3 = self.get_center3x3(
                grids, centers, strides) & is_in_boxes # M x N
            is_cared_in_the_level = self.assign_reg_fpn(
                reg_target, reg_size_ranges) # M x N
            reg_mask = is_center3x3 & is_cared_in_the_level # M x N

            dist2 = ((grids.view(M, 1, 2).expand(M, N, 2) - \
                centers_expanded) ** 2).sum(dim=2) # M x N
            dist2[is_peak] = 0
            radius2 = self.delta ** 2 * 2 * area # N
            radius2 = torch.clamp(
                radius2, min=self.min_radius ** 2)
            weighted_dist2 = dist2 / radius2.view(1, N).expand(M, N) # M x N            
            reg_target = self._get_reg_targets(
                reg_target, weighted_dist2.clone(), reg_mask, area) # M x 4

            if self.only_proposal:
                flattened_hm = self._create_agn_heatmaps_from_dist(
                    weighted_dist2.clone()) # M x 1
            else:
                flattened_hm = self._create_heatmaps_from_dist(
                    weighted_dist2.clone(), gt_classes, 
                    channels=heatmap_channels) # M x C

            reg_targets.append(reg_target)
            flattened_hms.append(flattened_hm)
        
        # transpose im first training_targets to level first ones
        reg_targets = _transpose(reg_targets, num_loc_list)
        flattened_hms = _transpose(flattened_hms, num_loc_list)
        for l in range(len(reg_targets)):
            reg_targets[l] = reg_targets[l] / float(self.strides[l])
        reg_targets = cat([x for x in reg_targets], dim=0) # MB x 4
        flattened_hms = cat([x for x in flattened_hms], dim=0) # MB x C
        
        return pos_inds, labels, reg_targets, flattened_hms


    def _get_label_inds(self, gt_instances, shapes_per_level):
        '''
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        Returns:
            pos_inds: N'
            labels: N'
        '''
        pos_inds = []
        labels = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2).contiguous()
            if self.not_clamp_box:
                h, w = gt_instances[im_i]._image_size
                centers[:, :, 0].clamp_(min=0).clamp_(max=w-1)
                centers[:, :, 1].clamp_(min=0).clamp_(max=h-1)
            strides = strides_default.view(1, L, 1).expand(n, L, 2)
            centers_inds = (centers / strides).long() # n x L x 2
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            pos_ind = level_bases.view(1, L).expand(n, L) + \
                       im_i * loc_per_level.view(1, L).expand(n, L) + \
                       centers_inds[:, :, 1] * Ws + \
                       centers_inds[:, :, 0] # n x L
            is_cared_in_the_level = self.assign_fpn_level(bboxes)
            # print(L)
            # print(is_cared_in_the_level.size)
            # print(type(pos_ind))
            # print(pos_ind.shape)
            pos_ind = pos_ind[is_cared_in_the_level].view(-1)
            label = targets_per_im.gt_classes.view(
                n, 1).expand(n, L)[is_cared_in_the_level].view(-1)

            pos_inds.append(pos_ind) # n'
            labels.append(label) # n'
        pos_inds = torch.cat(pos_inds, dim=0).long()
        labels = torch.cat(labels, dim=0)
        return pos_inds, labels # N, N


    def assign_fpn_level(self, boxes):
        '''
        Inputs:
            boxes: n x 4
            size_ranges: L x 2
        Return:
            is_cared_in_the_level: n x L
        '''
        size_ranges = boxes.new_tensor(
            self.sizes_of_interest).view(len(self.sizes_of_interest), 2) # L x 2
        crit = ((boxes[:, 2:] - boxes[:, :2]) **2).sum(dim=1) ** 0.5 / 2 # n
        n, L = crit.shape[0], size_ranges.shape[0]
        crit = crit.view(n, 1).expand(n, L)
        size_ranges_expand = size_ranges.view(1, L, 2).expand(n, L, 2)
        is_cared_in_the_level = (crit >= size_ranges_expand[:, :, 0]) & \
            (crit <= size_ranges_expand[:, :, 1])
        return is_cared_in_the_level
    

    def assign_reg_fpn(self, reg_targets_per_im, size_ranges):
        '''
        TODO (Xingyi): merge it with assign_fpn_level
        Inputs:
            reg_targets_per_im: M x N x 4
            size_ranges: M x 2
        '''
        crit = ((reg_targets_per_im[:, :, :2] + \
            reg_targets_per_im[:, :, 2:])**2).sum(dim=2) ** 0.5 / 2 # M x N
        is_cared_in_the_level = (crit >= size_ranges[:, [0]]) & \
            (crit <= size_ranges[:, [1]])
        return is_cared_in_the_level


    def _get_reg_targets(self, reg_targets, dist, mask, area):
        '''
          reg_targets (M x N x 4): long tensor
          dist (M x N)
          is_*: M x N
        '''
        dist[mask == 0] = INF * 1.0
        min_dist, min_inds = dist.min(dim=1) # M
        reg_targets_per_im = reg_targets[
            range(len(reg_targets)), min_inds] # M x N x 4 --> M x 4
        reg_targets_per_im[min_dist == INF] = - INF
        return reg_targets_per_im


    def _create_heatmaps_from_dist(self, dist, labels, channels):
        '''
        dist: M x N
        labels: N
        return:
          heatmaps: M x C
        '''
        heatmaps = dist.new_zeros((dist.shape[0], channels))
        for c in range(channels):
            inds = (labels == c) # N
            if inds.int().sum() == 0:
                continue
            heatmaps[:, c] = torch.exp(-dist[:, inds].min(dim=1)[0])
            zeros = heatmaps[:, c] < 1e-4
            heatmaps[zeros, c] = 0
        return heatmaps


    def _create_agn_heatmaps_from_dist(self, dist):
        '''
        TODO (Xingyi): merge it with _create_heatmaps_from_dist
        dist: M x N
        return:
          heatmaps: M x 1
        '''
        heatmaps = dist.new_zeros((dist.shape[0], 1))
        heatmaps[:, 0] = torch.exp(-dist.min(dim=1)[0])
        zeros = heatmaps < 1e-4
        heatmaps[zeros] = 0
        return heatmaps


    def _flatten_outputs(self, clss, reg_pred, agn_hm_pred):
        # Reshape: (N, F, Hl, Wl) -> (N, Hl, Wl, F) -> (sum_l N*Hl*Wl, F)
        clss = cat([x.permute(0, 2, 3, 1).reshape(-1, x.shape[1]) \
            for x in clss], dim=0) if clss[0] is not None else None
        reg_pred = cat(
            [x.permute(0, 2, 3, 1).reshape(-1, 4) for x in reg_pred], dim=0)            
        agn_hm_pred = cat([x.permute(0, 2, 3, 1).reshape(-1) \
            for x in agn_hm_pred], dim=0) if self.with_agn_hm else None
        return clss, reg_pred, agn_hm_pred


    def get_center3x3(self, locations, centers, strides):
        '''
        Inputs:
            locations: M x 2
            centers: N x 2
            strides: M
        '''
        M, N = locations.shape[0], centers.shape[0]
        locations_expanded = locations.view(M, 1, 2).expand(M, N, 2) # M x N x 2
        centers_expanded = centers.view(1, N, 2).expand(M, N, 2) # M x N x 2
        strides_expanded = strides.view(M, 1, 1).expand(M, N, 2) # M x N
        centers_discret = ((centers_expanded / strides_expanded).int() * \
            strides_expanded).float() + strides_expanded / 2 # M x N x 2
        dist_x = (locations_expanded[:, :, 0] - centers_discret[:, :, 0]).abs()
        dist_y = (locations_expanded[:, :, 1] - centers_discret[:, :, 1]).abs()
        return (dist_x <= strides_expanded[:, :, 0]) & \
            (dist_y <= strides_expanded[:, :, 0])


    @torch.no_grad()
    def inference(self, images, clss_per_level, reg_pred_per_level, 
        agn_hm_pred_per_level, grids):
        logits_pred = [x.sigmoid() if x is not None else None \
            for x in clss_per_level]
        agn_hm_pred_per_level = [x.sigmoid() if x is not None else None \
            for x in agn_hm_pred_per_level]

        if self.only_proposal:
            proposals = self.predict_instances(
                grids, agn_hm_pred_per_level, reg_pred_per_level, 
                images.image_sizes, [None for _ in agn_hm_pred_per_level])
        else:
            proposals = self.predict_instances(
                grids, logits_pred, reg_pred_per_level, 
                images.image_sizes, agn_hm_pred_per_level)
        if self.as_proposal or self.only_proposal:
            for p in range(len(proposals)):
                proposals[p].proposal_boxes = proposals[p].get('pred_boxes')
                proposals[p].objectness_logits = proposals[p].get('scores')
                proposals[p].remove('pred_boxes')

        if self.debug:
            debug_test(
                [self.denormalizer(x) for x in images], 
                logits_pred, reg_pred_per_level, 
                agn_hm_pred_per_level, preds=proposals,
                vis_thresh=self.vis_thresh, 
                debug_show_name=False)
        return proposals, {}


    @torch.no_grad()
    def predict_instances(
        self, grids, logits_pred, reg_pred, image_sizes, agn_hm_pred, 
        is_proposal=False):
        sampled_boxes = []
        for l in range(len(grids)):
            sampled_boxes.append(self.predict_single_level(
                grids[l], logits_pred[l], reg_pred[l] * self.strides[l],
                image_sizes, agn_hm_pred[l], l, is_proposal=is_proposal))
        boxlists = list(zip(*sampled_boxes))
        boxlists = [Instances.cat(boxlist) for boxlist in boxlists]
        boxlists = self.nms_and_topK(
            boxlists, nms=not self.not_nms)
        return boxlists

    
    @torch.no_grad()
    def predict_single_level(
        self, grids, heatmap, reg_pred, image_sizes, agn_hm, level, 
        is_proposal=False):
        N, C, H, W = heatmap.shape
        # put in the same format as grids
        if self.center_nms:
            heatmap_nms = nn.functional.max_pool2d(
                heatmap, (3, 3), stride=1, padding=1)
            heatmap = heatmap * (heatmap_nms == heatmap).float()
        heatmap = heatmap.permute(0, 2, 3, 1) # N x H x W x C
        heatmap = heatmap.reshape(N, -1, C) # N x HW x C
        box_regression = reg_pred.view(N, 4, H, W).permute(0, 2, 3, 1) # N x H x W x 4 
        box_regression = box_regression.reshape(N, -1, 4)

        candidate_inds = heatmap > self.score_thresh # 0.05
        pre_nms_top_n = candidate_inds.view(N, -1).sum(1) # N
        pre_nms_topk = self.pre_nms_topk_train if self.training else self.pre_nms_topk_test
        pre_nms_top_n = pre_nms_top_n.clamp(max=pre_nms_topk) # N

        if agn_hm is not None:
            agn_hm = agn_hm.view(N, 1, H, W).permute(0, 2, 3, 1)
            agn_hm = agn_hm.reshape(N, -1)
            heatmap = heatmap * agn_hm[:, :, None]

        results = []
        for i in range(N):
            per_box_cls = heatmap[i] # HW x C
            per_candidate_inds = candidate_inds[i] # n
            per_box_cls = per_box_cls[per_candidate_inds] # n

            per_candidate_nonzeros = per_candidate_inds.nonzero() # n
            per_box_loc = per_candidate_nonzeros[:, 0] # n
            per_class = per_candidate_nonzeros[:, 1] # n

            per_box_regression = box_regression[i] # HW x 4
            per_box_regression = per_box_regression[per_box_loc] # n x 4
            per_grids = grids[per_box_loc] # n x 2

            per_pre_nms_top_n = pre_nms_top_n[i] # 1

            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_box_cls, top_k_indices = \
                    per_box_cls.topk(per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_box_regression = per_box_regression[top_k_indices]
                per_grids = per_grids[top_k_indices]
            
            detections = torch.stack([
                per_grids[:, 0] - per_box_regression[:, 0],
                per_grids[:, 1] - per_box_regression[:, 1],
                per_grids[:, 0] + per_box_regression[:, 2],
                per_grids[:, 1] + per_box_regression[:, 3],
            ], dim=1) # n x 4

            # avoid invalid boxes in RoI heads
            detections[:, 2] = torch.max(detections[:, 2], detections[:, 0] + 0.01)
            detections[:, 3] = torch.max(detections[:, 3], detections[:, 1] + 0.01)
            boxlist = Instances(image_sizes[i])
            boxlist.scores = torch.sqrt(per_box_cls) \
                if self.with_agn_hm else per_box_cls # n
            # import pdb; pdb.set_trace()
            boxlist.pred_boxes = Boxes(detections)
            boxlist.pred_classes = per_class
            results.append(boxlist)
        return results

    
    @torch.no_grad()
    def nms_and_topK(self, boxlists, nms=True):
        num_images = len(boxlists)
        results = []
        for i in range(num_images):
            nms_thresh = self.nms_thresh_train if self.training else \
                self.nms_thresh_test
            result = ml_nms(boxlists[i], nms_thresh) if nms else boxlists[i]
            if self.debug:
                print('#proposals before nms', len(boxlists[i]))
                print('#proposals after nms', len(result))
            num_dets = len(result)
            post_nms_topk = self.post_nms_topk_train if self.training else \
                self.post_nms_topk_test
            if num_dets > post_nms_topk:
                cls_scores = result.scores
                image_thresh, _ = torch.kthvalue(
                    cls_scores.float().cpu(),
                    num_dets - post_nms_topk + 1
                )
                keep = cls_scores >= image_thresh.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]
            if self.debug:
                print('#proposals after filter', len(result))
            results.append(result)
        return results

    
    @torch.no_grad()
    def _add_more_pos(self, reg_pred, gt_instances, shapes_per_level):
        labels, level_masks, c33_inds, c33_masks, c33_regs = \
            self._get_c33_inds(gt_instances, shapes_per_level)
        N, L, K = labels.shape[0], len(self.strides), 9
        c33_inds[c33_masks == 0] = 0
        reg_pred_c33 = reg_pred[c33_inds].detach() # N x L x K
        invalid_reg = c33_masks == 0
        c33_regs_expand = c33_regs.view(N * L * K, 4).clamp(min=0)
        if N > 0:
            with torch.no_grad():
                c33_reg_loss = self.iou_loss(
                    reg_pred_c33.view(N * L * K, 4), 
                    c33_regs_expand, None,
                    reduction='none').view(N, L, K).detach() # N x L x K
        else:
            c33_reg_loss = reg_pred_c33.new_zeros((N, L, K)).detach()
        c33_reg_loss[invalid_reg] = INF # N x L x K
        c33_reg_loss.view(N * L, K)[level_masks.view(N * L), 4] = 0 # real center
        c33_reg_loss = c33_reg_loss.view(N, L * K)
        if N == 0:
            loss_thresh = c33_reg_loss.new_ones((N)).float()
        else:
            loss_thresh = torch.kthvalue(
                c33_reg_loss, self.more_pos_topk, dim=1)[0] # N
        loss_thresh[loss_thresh > self.more_pos_thresh] = self.more_pos_thresh # N
        new_pos = c33_reg_loss.view(N, L, K) < \
            loss_thresh.view(N, 1, 1).expand(N, L, K)
        pos_inds = c33_inds[new_pos].view(-1) # P
        labels = labels.view(N, 1, 1).expand(N, L, K)[new_pos].view(-1)
        return pos_inds, labels
        
    
    @torch.no_grad()
    def _get_c33_inds(self, gt_instances, shapes_per_level):
        '''
        TODO (Xingyi): The current implementation is ugly. Refactor.
        Get the center (and the 3x3 region near center) locations of each objects
        Inputs:
            gt_instances: [n_i], sum n_i = N
            shapes_per_level: L x 2 [(h_l, w_l)]_L
        '''
        labels = []
        level_masks = []
        c33_inds = []
        c33_masks = []
        c33_regs = []
        L = len(self.strides)
        B = len(gt_instances)
        shapes_per_level = shapes_per_level.long()
        loc_per_level = (shapes_per_level[:, 0] * shapes_per_level[:, 1]).long() # L
        level_bases = []
        s = 0
        for l in range(L):
            level_bases.append(s)
            s = s + B * loc_per_level[l]
        level_bases = shapes_per_level.new_tensor(level_bases).long() # L
        strides_default = shapes_per_level.new_tensor(self.strides).float() # L
        K = 9
        dx = shapes_per_level.new_tensor([-1, 0, 1, -1, 0, 1, -1, 0, 1]).long()
        dy = shapes_per_level.new_tensor([-1, -1, -1, 0, 0, 0, 1, 1, 1]).long()
        for im_i in range(B):
            targets_per_im = gt_instances[im_i]
            bboxes = targets_per_im.gt_boxes.tensor # n x 4
            n = bboxes.shape[0]
            if n == 0:
                continue
            centers = ((bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2) # n x 2
            centers = centers.view(n, 1, 2).expand(n, L, 2)

            strides = strides_default.view(1, L, 1).expand(n, L, 2) # 
            centers_inds = (centers / strides).long() # n x L x 2
            center_grids = centers_inds * strides + strides // 2# n x L x 2
            l = center_grids[:, :, 0] - bboxes[:, 0].view(n, 1).expand(n, L)
            t = center_grids[:, :, 1] - bboxes[:, 1].view(n, 1).expand(n, L)
            r = bboxes[:, 2].view(n, 1).expand(n, L) - center_grids[:, :, 0]
            b = bboxes[:, 3].view(n, 1).expand(n, L) - center_grids[:, :, 1] # n x L
            reg = torch.stack([l, t, r, b], dim=2) # n x L x 4
            reg = reg / strides_default.view(1, L, 1).expand(n, L, 4).float()
            
            Ws = shapes_per_level[:, 1].view(1, L).expand(n, L)
            Hs = shapes_per_level[:, 0].view(1, L).expand(n, L)
            expand_Ws = Ws.view(n, L, 1).expand(n, L, K)
            expand_Hs = Hs.view(n, L, 1).expand(n, L, K)
            label = targets_per_im.gt_classes.view(n).clone()
            mask = reg.min(dim=2)[0] >= 0 # n x L
            mask = mask & self.assign_fpn_level(bboxes)
            labels.append(label) # n
            level_masks.append(mask) # n x L

            Dy = dy.view(1, 1, K).expand(n, L, K)
            Dx = dx.view(1, 1, K).expand(n, L, K)
            c33_ind = level_bases.view(1, L, 1).expand(n, L, K) + \
                       im_i * loc_per_level.view(1, L, 1).expand(n, L, K) + \
                       (centers_inds[:, :, 1:2].expand(n, L, K) + Dy) * expand_Ws + \
                       (centers_inds[:, :, 0:1].expand(n, L, K) + Dx) # n x L x K
            
            c33_mask = \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) < expand_Hs) & \
                ((centers_inds[:, :, 1:2].expand(n, L, K) + dy) >= 0) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) < expand_Ws) & \
                ((centers_inds[:, :, 0:1].expand(n, L, K) + dx) >= 0)
            # TODO (Xingyi): think about better way to implement this
            # Currently it hard codes the 3x3 region
            c33_reg = reg.view(n, L, 1, 4).expand(n, L, K, 4).clone()
            c33_reg[:, :, [0, 3, 6], 0] -= 1
            c33_reg[:, :, [0, 3, 6], 2] += 1
            c33_reg[:, :, [2, 5, 8], 0] += 1
            c33_reg[:, :, [2, 5, 8], 2] -= 1
            c33_reg[:, :, [0, 1, 2], 1] -= 1
            c33_reg[:, :, [0, 1, 2], 3] += 1
            c33_reg[:, :, [6, 7, 8], 1] += 1
            c33_reg[:, :, [6, 7, 8], 3] -= 1
            c33_mask = c33_mask & (c33_reg.min(dim=3)[0] >= 0) # n x L x K
            c33_inds.append(c33_ind)
            c33_masks.append(c33_mask)
            c33_regs.append(c33_reg)
        
        if len(level_masks) > 0:
            labels = torch.cat(labels, dim=0)
            level_masks = torch.cat(level_masks, dim=0)
            c33_inds = torch.cat(c33_inds, dim=0).long()
            c33_regs = torch.cat(c33_regs, dim=0)
            c33_masks = torch.cat(c33_masks, dim=0)
        else:
            labels = shapes_per_level.new_zeros((0)).long()
            level_masks = shapes_per_level.new_zeros((0, L)).bool()
            c33_inds = shapes_per_level.new_zeros((0, L, K)).long()
            c33_regs = shapes_per_level.new_zeros((0, L, K, 4)).float()
            c33_masks = shapes_per_level.new_zeros((0, L, K)).bool()
        return labels, level_masks, c33_inds, c33_masks, c33_regs # N x L, N x L x K