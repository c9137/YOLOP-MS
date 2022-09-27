# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""YOLOP based on DarkNet."""

from cgi import MiniFieldStorage
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
import mindspore.numpy as np
from mindspore import ms_function
from mindspore.ops import constexpr

from src.general import bbox_iou
from src.backbone import YOLOPBackbone, Conv, BottleneckCSP

from src.model_utils.config import config as default_config


class YOLOP(nn.Cell):  #                   0   1   2   3    4    5    6
    def __init__(self, backbone):  # shape[3, 32, 64, 128, 256, 512, 1]
        super(YOLOP, self).__init__()
        self.backbone = backbone
        self.config = default_config
        self.config.out_channel = (self.config.num_classes + 5) * 3

        # Encoder
        self.conv10 = Conv(512, 256, k=1, s=1)  # 10
        # Upsample 11
        # Concat 12   11 + 6
        self.csp13 = BottleneckCSP(512, 256, n=1, shortcut=False)  # 13
        self.conv14 = Conv(256, 128, k=1, s=1)  # 14
        # Upsample 15
        # Concat 16   15 + 4

        # detect
        self.csp17 = BottleneckCSP(256, 128, n=1, shortcut=False)  # 17
        self.conv18 = Conv(128, 128, k=3, s=2)  # 18
        # Concat 19  18 + 14
        self.csp20 = BottleneckCSP(256, 256, n=1, shortcut=False)  # 20
        self.conv21 = Conv(256, 256, k=3, s=2)  # 21
        # Concat 22  21 + 10
        self.csp23 = BottleneckCSP(512, 512, n=1, shortcut=False)  # 23

        # Detect 24
        self.back_block1 = YoloBlock(128, self.config.out_channel)
        self.back_block2 = YoloBlock(256, self.config.out_channel)
        self.back_block3 = YoloBlock(512, self.config.out_channel)

        # Driving area segmentation head
        self.conv25 = Conv(256, 128, k=3, s=1)  # 25
        # Upsample 26
        self.csp27 = BottleneckCSP(128, 64, 1, False)  # 27
        self.conv28 = Conv(64, 32, 3, 1)  # 28
        # Upsample 29
        self.conv30 = Conv(32, 16, 3, 1)  # 30
        self.csp31 = BottleneckCSP(16, 8, 1, False)  # 31
        # Upsample 32
        self.conv33 = Conv(8, 2, 3, 1)  # 33

        # Lane line segmentation head
        self.conv34 = Conv(256, 128, k=3, s=1)  # 34
        # Upsample 35
        self.csp36 = BottleneckCSP(128, 64, 1, False)  # 36
        self.conv37 = Conv(64, 32, 3, 1)  # 37
        # Upsample 38
        self.conv39 = Conv(32, 16, 3, 1)  # 39
        self.csp40 = BottleneckCSP(16, 8, 1, False)  # 40
        # Upsample 41
        self.conv42 = Conv(8, 2, 3, 1)  # 42

        self.concat = ops.Concat(axis=1)

    def construct(self, x):
        """
        input_shape of x is (batch_size, 3, h, w)
        e4 is (batch_size, backbone_shape[2], h/8, w/8) 80
        e6 is (batch_size, backbone_shape[3], h/16, w/16) 40
        e9 is (batch_size, backbone_shape[4], h/32, w/32) 20
        """
        img_height = x.shape[2]
        img_width = x.shape[3]

        e4, e6, e9 = self.backbone(x)  # 4, 6, 9

        # Encoder
        e10 = self.conv10(e9)  # 10
        e_ups11 = ops.ResizeNearestNeighbor((img_height // 16, img_width // 16))(e10)  # 11
        e12 = self.concat((e_ups11, e6))  # 12
        e13 = self.csp13(e12)  # 13
        e14 = self.conv14(e13)  # 14
        e_ups15 = ops.ResizeNearestNeighbor((img_height // 8, img_width // 8))(e14)  # 15
        e16 = self.concat((e_ups15, e4))  # 16

        # det small_object_output
        d17 = self.csp17(e16)  # 17
        d18 = self.conv18(d17)  # 18
        d19 = self.concat((d18, e14))  # 19
        # det medium_object_output
        d20 = self.csp20(d19)  # 20
        d21 = self.conv21(d20)   # 21
        d22 = self.concat((d21, e10))  # 22
        # det big_object_output
        d23 = self.csp23(d22)  # 23
        # 24
        small_object_output = self.back_block1(d17)
        medium_object_output = self.back_block2(d20)
        big_object_output = self.back_block3(d23)

        # Driving area segmentation
        s25 = self.conv25(e16)  # 25
        s_ups26 = ops.ResizeNearestNeighbor((img_height // 4, img_width // 4))(s25)
        s27 = self.csp27(s_ups26)  # 27
        s28 = self.conv28(s27)  # 28
        s_ups29 = ops.ResizeNearestNeighbor((img_height // 2, img_width // 2))(s28)
        s30 = self.conv30(s_ups29)  # 30
        s31 = self.csp31(s30)  # 31
        s_ups32 = ops.ResizeNearestNeighbor((img_height, img_width))(s31)
        da_seg_output = self.conv33(s_ups32)  # 33

        # Lane line segmentation head
        s34 = self.conv34(e16)  # 34
        s_ups35 = ops.ResizeNearestNeighbor((img_height // 4, img_width // 4))(s34)
        s36 = self.csp36(s_ups35)  # 36
        s37 = self.conv37(s36)  # 37
        s_ups38 = ops.ResizeNearestNeighbor((img_height // 2, img_width // 2))(s37)
        s39 = self.conv39(s_ups38)  # 39
        s40 = self.csp40(s39)  # 40
        s_ups41 = ops.ResizeNearestNeighbor((img_height, img_width))(s40)
        ll_seg_output = self.conv42(s_ups41)  # 42

        # （1, 18, 80, 80）  （1, 18, 40, 40）  （1, 18, 20, 20）  （1, 2, 640, 640）  （1, 2, 640, 640）
        return small_object_output, medium_object_output, big_object_output, da_seg_output, ll_seg_output


class YoloBlock(nn.Cell):
    """
    YoloBlock for YOLOP.

    Args:
        in_channels: Integer. Input channel.
        out_channels: Integer. Output channel.

    Returns:
        Tuple, tuple of output tensor,(f1,f2,f3).

    Examples:
        YoloBlock(12, 255)
        YoloPBlock(128, 3*(1+5))
                  (256, 3*(1+5))
                  (512, 3*(1+5))

    """
    def __init__(self, in_channels, out_channels):
        super(YoloBlock, self).__init__()
        self.conv24 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, has_bias=True)

    def construct(self, x):
        """construct method"""
        out = self.conv24(x)
        return out


    
class DetectionBlock(nn.Cell):
    """
     YOLOP detection Network. It will finally output the detection result.

     Args:
         scale: Character.
         config: config, Configuration instance.
         is_training: Bool, Whether train or not, default True.

     Returns:
         Tuple, tuple of output tensor,(f1,f2,f3).

     Examples:
         DetectionBlock(scale='l',stride=32)
     """

    def __init__(self, scale, config=default_config, is_training=True):
        super(DetectionBlock, self).__init__()
        self.config = config
        if scale == 's':
            idx = (0, 1, 2)
            self.stride = 8.0
        elif scale == 'm':
            idx = (3, 4, 5)
            self.stride = 16.0
        elif scale == 'l':
            idx = (6, 7, 8)
            self.stride = 32.0
        else:
            raise KeyError("Invalid scale value for DetectionBlock")
        # self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32) / self.stride
        
        self.num_anchors_per_scale = 3
        self.num_attrib = 4 + 1 + self.config.num_classes
        self.lambda_coord = 1

        self.sigmoid = nn.Sigmoid()
        self.reshape = ops.Reshape()
        self.tile = ops.Tile()
        self.concat = ops.Concat(axis=-1)
        # self.pow = ops.Pow()
        self.transpose = ops.Transpose()
        # self.exp = ops.Exp()
        self.conf_training = is_training

        self.anchors = ms.Tensor([self.config.anchor_scales[i] for i in idx], ms.float32)
        self.anchor_grid = self.anchors.view(1, -1, 1, 1, 2)

    def construct(self, x):
        """construct method"""
        # x:(num_batch, 3 * 6, grid_size, grid_size)
        num_batch = x.shape[0]
        grid_size = x.shape[2:4]  # big (20, 20)  medium (40, 40)  small (80, 80)

        # Reshape and transpose the feature to [n, grid_size[0], grid_size[1], 3, num_attrib]
        prediction = self.reshape(x, (num_batch,
                                      self.num_anchors_per_scale,  # 3
                                      self.num_attrib,  # 6
                                      grid_size[0],
                                      grid_size[1]))
        # prediction = self.transpose(prediction, (0, 3, 4, 1, 2))  # (1, 80, 80, 3, 6)
        prediction = self.transpose(prediction, (0, 1, 3, 4, 2))  # (1, 3, 80, 80, 6)
        grid = self._make_grid(grid_size[1], grid_size[0])

        box_xy = prediction[:, :, :, :, :2]  # (1, 3, 80, 80, 2)
        box_wh = prediction[:, :, :, :, 2:4]  # (1, 3, 80, 80, 2)
        box_confidence = prediction[:, :, :, :, 4:5]  # (1, 3, 80, 80, 1)
        box_probs = prediction[:, :, :, :, 5:]  # (1, 3, 80, 80, 1)

        # gridsize1 is x
        # gridsize0 is y
        box_xy = (2. * self.sigmoid(box_xy) - 0.5 + grid) * self.stride
        # box_wh is w->h
        box_wh = (self.sigmoid(box_wh) * 2.0) ** 2 * self.anchor_grid

        box_confidence = self.sigmoid(box_confidence)
        box_probs = self.sigmoid(box_probs)

        if self.conf_training:
            return prediction, box_xy, box_wh

        # return self.concat((box_xy, box_wh, box_confidence, box_probs))
        #
        y = self.concat((box_xy, box_wh, box_confidence, box_probs))
        return self.reshape(y, (num_batch, -1, self.num_attrib)), prediction
    
    @constexpr
    def _make_grid(nx=20, ny=20):
        xv, yv = np.meshgrid(np.arange(nx), np.arange(ny))
        return ops.Cast()((ops.Stack(2)((xv, yv)).view((1, 1, ny, nx, 2))), ms.float32)


class MCnet(nn.Cell):
    """
    Args:
        is_training: Bool. Whether train or not.

    Returns:
        Cell, cell instance of YOLOP neural network.

   """

    def __init__(self, is_training):
        super(MCnet, self).__init__()
        self.config = default_config

        # self.shape = self.config.input_shape  # [3, 32, 64, 128, 256, 512, 1]
        self.feature_map = YOLOP(backbone=YOLOPBackbone())

        # prediction on the default anchor boxes
        self.detect_1 = DetectionBlock('s', is_training=is_training)
        self.detect_2 = DetectionBlock('m', is_training=is_training)
        self.detect_3 = DetectionBlock('l', is_training=is_training)

        self.is_training = is_training

        self.sigmoid = nn.Sigmoid()
        self.concat = ops.Concat(axis=1)

        self.nc = 1
        self.names = [str(i) for i in range(self.nc)]

    def construct(self, x):  # input_shape  640
        # （1, 18, 80, 80）  （1, 18, 40, 40）  （1, 18, 20, 20）  （1, 2, 640, 640）  （1, 2, 640, 640）
        small_object_output, medium_object_output, big_object_output, da_seg_output, ll_seg_output = self.feature_map(x)
        da_seg_output = self.sigmoid(da_seg_output)
        ll_seg_output = self.sigmoid(ll_seg_output)

        if not self.is_training:  # infer
            output_small, pre_small = self.detect_1(small_object_output)  # （1, 19200, 6）（1, 80, 80, 3, 6）
            output_me, pre_me = self.detect_2(medium_object_output)  # （1, 4800, 6）（1, 40, 40, 3, 6）
            output_big, pre_big = self.detect_3(big_object_output)  # （1, 1200, 6）（1, 20, 20, 3, 6）
            #        （1, 25200, 6）
            return [self.concat([output_small, output_me, output_big]), [pre_small, pre_me, pre_big]], da_seg_output, ll_seg_output
        else:  # train
            output_small, _, _ = self.detect_1(small_object_output)  # （1, 80, 80, 3, 6）
            output_me, _, _ = self.detect_2(medium_object_output)  # （1, 40, 40, 3, 6）
            output_big, _, _ = self.detect_3(big_object_output)  # （1, 20, 20, 3, 6）
        return [output_small, output_me, output_big], da_seg_output, ll_seg_output


class MultiHeadLoss(nn.Cell):
    """
    collect all the loss we need
    """
    def __init__(self, losses, cfg, lambdas=None):
        """
        Inputs:
        - losses: (list)[nn.Cell, nn.Cell, ...]
        - cfg: config object
        - lambdas: (list) + IoU loss, weight for each loss
        """
        super().__init__()
        # lambdas: [cls, obj, iou, la_seg, ll_seg, ll_iou]
        if not lambdas:
            lambdas = [1.0 for _ in range(len(losses) + 3)]
        assert all(lam >= 0.0 for lam in lambdas)

        self.losses = nn.CellList(losses)
        self.lambdas = ops.tuple_to_array(tuple(lambdas))
        self.cfg = cfg

        # assign model params
        self.gr = 1.0
        self.nc = 1
        self.sigmoid = ops.Sigmoid()
        self.cast = ops.Cast()

        self.SINGLE_CLS = cfg.SINGLE_CLS
        self.LOSS_CLS_GAIN = cfg.LOSS_CLS_GAIN
        self.LOSS_OBJ_GAIN = cfg.LOSS_OBJ_GAIN
        self.LOSS_BOX_GAIN = cfg.LOSS_BOX_GAIN

        self.LOSS_DA_SEG_GAIN = cfg.LOSS_DA_SEG_GAIN
        self.LOSS_LL_SEG_GAIN = cfg.LOSS_LL_SEG_GAIN
        self.LOSS_LL_IOU_GAIN = cfg.LOSS_LL_IOU_GAIN

        self.cp, self.cn = smooth_BCE(0.0)  # 1   0

    def construct(self, predictions, targets, shapes, tcls, tbox, indices, anchors):
        """
        Inputs:
        - head_fields: (list) output from each task head
        - head_targets: (list) ground-truth for each task head

        Returns:
        - total_loss: sum of all the loss
        - head_losses: (tuple) contain all loss[loss1, loss2, ...]
        """
        lcls, lbox, lobj = ops.Zeros()(1, ms.float32), ops.Zeros()(1, ms.float32), ops.Zeros()(1, ms.float32)

        # Class label smoothing https://arxiv.org/pdf/1902.04103.pdf eqn 3


        BCEcls, BCEobj, BCEseg = self.losses

        # Calculate Losses
        nt = 0  # number of targets
        no = len(predictions[0])  # number of outputs
        balance = ops.tuple_to_array((4.0, 1.0, 0.4)) if no == 3 else ops.tuple_to_array((4.0, 1.0, 0.4, 0.1))  # P3-5 or P3-6

        # calculate detection loss
        for i, pi in enumerate(predictions[0]):  # layer index, layer predictions
            b, a, gj, gi = indices[i]  # image, anchor, gridy, gridx
            tobj = ops.ZerosLike()(pi[..., 0])  # target obj

            n = b.shape[0]  # number of targets
            if n:
                nt += n  # cumulative targets
                ps = pi[b, a, gj, gi]  # prediction subset corresponding to targets

                # Regression
                pxy = self.sigmoid(ps[:, :2]) * 2. - 0.5
                pwh = (self.sigmoid(ps[:, 2:4]) * 2) ** 2 * anchors[i]
                pbox = ops.Concat(axis=1)((pxy, pwh))  # predicted box
                iou = bbox_iou(pbox.T, tbox[i], x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=True, eps=1e-9)  # iou(prediction, target)
                lbox += (1.0 - iou).mean()  # iou loss

                # assign model params
                model_gr = 1.0
                # Objectness

                iou = ops.stop_gradient(iou)
                tobj[b, a, gj, gi] = (1.0 - model_gr) + ops.scalar_to_array(model_gr) * ops.clip_by_value(iou, ops.scalar_to_array(0.0), ops.scalar_to_array(100.0))  # iou ratio
                if not self.SINGLE_CLS:  # cls loss (only if multiple classes)
                    t = ms.numpy.full_like(ps[:, 5:], self.cn)  # targets
                    t[range(n), tcls[i]] = self.cp
                    lcls += BCEcls(ps[:, 5:], t)  # BCE
            lobj += BCEobj(pi[..., 4], tobj) * balance[i]  # obj loss

        drive_area_seg_predicts = predictions[1].view(-1)
        drive_area_seg_targets = targets[1].view(-1)
        lseg_da = BCEseg(drive_area_seg_predicts, drive_area_seg_targets)

        lane_line_seg_predicts = predictions[2].view(-1)
        lane_line_seg_targets = targets[2].view(-1)
        lseg_ll = BCEseg(lane_line_seg_predicts, lane_line_seg_targets)

        # nb, _, height, width = targets[1].shape
        # shapes = self.cast(shapes, ms.int64)
        # pad_w, pad_h = shapes[0][4], shapes[0][5]
        # # pad_w = int(pad_w)
        # # pad_h = int(pad_h)
        # lane_line_pred, _ = ops.ArgMaxWithValue(axis=1)(predictions[2])
        # lane_line_gt, _ = ops.ArgMaxWithValue(axis=1)(targets[2])
        # lane_line_pred = lane_line_pred[:, pad_h:height - pad_h, pad_w:width - pad_w]
        # lane_line_gt = lane_line_gt[:, pad_h:height - pad_h, pad_w:width - pad_w]

        # self.metric.reset()
        # self.metric.addBatch(lane_line_pred, lane_line_gt)
        # IoU = self.metric.IntersectionOverUnion()

        # IoU = seg_iou(lane_line_pred, lane_line_gt, nc=2)
        # # liou_ll = 1 - IoU

        s = ops.scalar_to_tensor(3 / no, ms.float32)  # output count scaling
        lcls *= self.LOSS_CLS_GAIN * s * self.lambdas[0]
        lobj *= self.LOSS_OBJ_GAIN * s * (ops.scalar_to_array(1.4) if no == 4 else ops.scalar_to_array(1.)) * self.lambdas[1]
        lbox *= self.LOSS_BOX_GAIN * s * self.lambdas[2]

        lseg_da *= self.LOSS_DA_SEG_GAIN * self.lambdas[3]
        lseg_ll *= self.LOSS_LL_SEG_GAIN * self.lambdas[4]
        # liou_ll *= self.LOSS_LL_IOU_GAIN * self.lambdas[5]

        # loss = lbox + lobj + lcls + lseg_da + lseg_ll + liou_ll
        loss = lbox + lobj + lcls + lseg_da + lseg_ll
        # loss = lseg
        # return loss * bs, torch.cat((lbox, lobj, lcls, loss)).detach()
        # return loss, (lbox.item(), lobj.item(), lcls.item(), lseg_da.item(), lseg_ll.item(), liou_ll.item(), loss.item())
        return loss


class FocalLoss(nn.Cell):
    # Wraps focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn, gamma=1.5, alpha=0.25):
        # alpha  balance positive & negative samples
        # gamma  focus on difficult samples
        super(FocalLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element
        self.sigmoid = ops.Sigmoid()

    def construct(self, pred, true):
        loss = self.loss_fcn(pred, true)
        # p_t = torch.exp(-loss)
        # loss *= self.alpha * (1.000001 - p_t) ** self.gamma  # non-zero power for gradient stability

        # TF implementation https://github.com/tensorflow/addons/blob/v0.7.1/tensorflow_addons/losses/focal_loss.py
        pred_prob = self.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        alpha_factor = true * self.alpha + (1 - true) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        loss *= alpha_factor * modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


class YoloPWithLossCell(nn.Cell):
    """YOLOP loss."""
    def __init__(self, network, cfg):
        super(YoloPWithLossCell, self).__init__()
        self.yolop_network = network
        self.config = default_config

        # class loss criteria
        BCEcls = nn.BCEWithLogitsLoss(pos_weight=ms.Tensor([cfg.LOSS_CLS_POS_WEIGHT]))
        # object loss criteria
        BCEobj = nn.BCEWithLogitsLoss(pos_weight=ms.Tensor([cfg.LOSS_OBJ_POS_WEIGHT]))
        # segmentation loss criteria
        BCEseg = nn.BCEWithLogitsLoss(pos_weight=ms.Tensor([cfg.LOSS_SEG_POS_WEIGHT]))
        # Focal loss
        gamma = cfg.LOSS_FL_GAMMA  # focal loss gamma
        if gamma > 0:
            BCEcls, BCEobj = FocalLoss(BCEcls, gamma), FocalLoss(BCEobj, gamma)

        loss_list = [BCEcls, BCEobj, BCEseg]
        self.loss_model = MultiHeadLoss(loss_list, cfg=cfg)

        self.gr = 1.0
        self.nc = 1

    def construct(self, x, labels_det, seg_label, lane_label, shape, tcls, tbox, indices, anch):
        yolop_out = self.yolop_network(x)
        targets = [labels_det, seg_label, lane_label]
        total_loss = self.loss_model(yolop_out, targets, shape, tcls, tbox, indices, anch)
        return total_loss
        # return yolop_out


def smooth_BCE(eps=0.1):  # https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    # return positive, negative label smoothing BCE targets
    return 1.0 - 0.5 * eps, 0.5 * eps





if __name__ == "__main__":
    from yolo_dataset import create_bdd_dataset, build_targets
    from model_utils.config import config

    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='Ascend')

    # ds = create_bdd_dataset(config, 1, 0)
    # data_loader = ds.create_tuple_iterator(do_copy=False)
    #
    # network = MCnet(is_training=True)
    # net = YoloPWithLossCell(network, config)
    #
    # for step_idx, data in enumerate(data_loader):
    #     if step_idx > 1:
    #         break
    #     input, labels_det, ture_labels_num, seg_label, lane_label, paths, shapes = data
    #     # print(ture_labels_num)
    #     tcls, tbox, indices, anch = build_targets(config, labels_det, ture_labels_num)
    #
    #     loss = net(input, labels_det, seg_label, lane_label, shapes, tcls, tbox, indices, anch)
    #     print(step_idx, loss)
    #     # print()

    network = MCnet(is_training=False)
    # # default is kaiming-normal
    #
    # # load_yolov5_params(config, network)
    # network = YoloPWithLossCell(network, config)
    #
    image = ops.Zeros()((8, 3, 640, 640), ms.float32)
    out = network(image)

    print(out)
