import time

import mindspore.ops as ops
import mindspore as ms
import mindspore.numpy as np

import math

def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    box2 = box2.T
    eps = ops.scalar_to_array(eps)
    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / ops.scalar_to_array(2.0), box1[0] + box1[2] / ops.scalar_to_array(2.0)
        b1_y1, b1_y2 = box1[1] - box1[3] / ops.scalar_to_array(2.0), box1[1] + box1[3] / ops.scalar_to_array(2.0)
        b2_x1, b2_x2 = box2[0] - box2[2] / ops.scalar_to_array(2.0), box2[0] + box2[2] / ops.scalar_to_array(2.0)
        b2_y1, b2_y2 = box2[1] - box2[3] / ops.scalar_to_array(2.0), box2[1] + box2[3] / ops.scalar_to_array(2.0)

    minimum = ops.Minimum()
    maximum = ops.Maximum()
    # Intersection area
    inter = ops.clip_by_value(minimum(b1_x2, b2_x2) - maximum(b1_x1, b2_x1), ops.scalar_to_array(0.0), ops.scalar_to_array(1.0)) * \
            ops.clip_by_value(minimum(b1_y2, b2_y2) - maximum(b1_y1, b2_y1), ops.scalar_to_array(0.0), ops.scalar_to_array(1.0))

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union
    cw = maximum(b1_x2, b2_x2) - minimum(b1_x1, b2_x1)  # convex (smallest enclosing box) width
    ch = maximum(b1_y2, b2_y2) - minimum(b1_y1, b2_y1)  # convex height
    c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
    rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
            (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / ops.scalar_to_array(4.0)  # center distance squared
    v = ops.scalar_to_tensor(4 / math.pi ** 2, ms.float32) * ops.Pow()(ops.Atan()(w2 / h2) - ops.Atan()(w1 / h1), 2)
    alpha = v / ((1 + eps) - iou + v)
    alpha = ops.stop_gradient(alpha)
    return iou - (rho2 / c2 + v * alpha)  # CIoU


def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    tensor_0 = ms.Tensor(0, ms.int32)
    tensor_img_shape = [ms.Tensor(i, ms.int32) for i in img_shape]
    boxes[:, 0] = ops.clip_by_value(boxes[:, 0], tensor_0, tensor_img_shape[1])  #boxes[:, 0].clamp_(0, img_shape[1])  # x1
    boxes[:, 1] = ops.clip_by_value(boxes[:, 1], tensor_0, tensor_img_shape[0])  #boxes[:, 1].clamp_(0, img_shape[0])  # y1
    boxes[:, 2] = ops.clip_by_value(boxes[:, 2], tensor_0, tensor_img_shape[1])  #boxes[:, 2].clamp_(0, img_shape[1])  # x2
    boxes[:, 3] = ops.clip_by_value(boxes[:, 3], tensor_0, tensor_img_shape[0])  #boxes[:, 3].clamp_(0, img_shape[0])  # y2


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])  # (x2-x1)*(y2-y1)

    area1 = box_area(ops.Transpose()(box1, (1, 0)))  # area1 = box_area(box1.T)
    area2 = box_area(ops.Transpose()(box2, (1, 0)))  # area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    # inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    inter = (np.minimum(box1[:, None, 2:], box2[:, 2:]) - np.maximum(box1[:, None, :2], box2[:, :2]))
    clip_value_min = ms.Tensor(0, ms.float32)
    clip_value_max = ms.Tensor(np.amax(inter), ms.float32)
    inter = ops.clip_by_value(inter, clip_value_min, clip_value_max)  # inter.clamp(0)
    ops.ReduceProd(False)(inter, 2)  # inter.prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    #y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y = ops.ZerosLike()(x) if isinstance(x, ms.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, labels=()):
    """Performs Non-Maximum Suppression (NMS) on inference results
    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # print("xc.max :")
    # print(type(xc))

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output = [ops.Zeros()((0, 6), ms.float32)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x_mask_list = xc[xi].asnumpy().tolist()
        if any(x_mask_list) == False:
            print("no result!")
            continue
        # for x_mask in x_mask_list:  if x_mask : print("have true!")
        x = x[x_mask_list]  # confidence

        # print("x {0}:".format(xi))
        # print(x)

        # Cat apriori labels if autolabelling
        if labels and len(labels[xi]):
            l = labels[xi]
            # v = torch.zeros((len(l), nc + 5), device=x.device)
            v = ops.Zeros()((len(l), nc + 5), ms.float32)
            v[:, :4] = l[:, 1:5]  # box
            v[:, 4] = 1.0  # conf
            v[range(len(l)), ops.Cast()(l[:, 0], ms.int64) + 5] = 1.0  # cls
            x = ops.Concat(0)((x, v))  # x = torch.cat((x, v), 0)

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        # print(x[0:4])
        box = xywh2xyxy(x[:, :4])
        # print("box out:")
        # print(box)

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = ops.Concat(1)((box[i], x[i, j + 5, None], j[:, None].float()))
            # x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            # conf, j = x[:, 5:].max(1, keepdims=True)
            j, conf = ops.ArgMaxWithValue(axis=1, keep_dims=True)(x[:, 5:])
            conf_bool = (conf.view(-1) > conf_thres).asnumpy().tolist()
            x = ops.Concat(1)((box, conf, ms.Tensor(j, ms.float32)))[conf_bool]
            # x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes is not None:
            x = x[(x[:, 5:6] == ms.tensor(classes, ms.float32)).any(1)]
            # x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # Check shape
        n = x.shape[0]  # number of boxes
        if not n:  # no boxes
            continue
        elif n > max_nms:  # excess boxes
            x = x[(ops.Sort(descending=True)(x[:, 4]))[:max_nms]]  # sort by confidence
            # x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # print("xxx:")
        # print(x)

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        # i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
        boxes_socres = ops.Concat(1)((boxes, scores.view(-1, 1)))
        output_boxes_scores, indices, mask = ops.NMSWithMask(iou_thres)(boxes_socres)  # NMS
        output_boxes = output_boxes_scores[:, :4]  # 取前四列坐标
        output_scores = output_boxes_scores[:, 4]  # 取最后一列分数
        i = (indices.asnumpy())[mask.asnumpy()]
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
            iou = box_iou(output_boxes[i, :4], output_boxes[:, :4]) > iou_thres
            # iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
            weights = iou * output_scores[None]  # weights = iou * scores[None]  # box weights
            # x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
            x_MatMul = ops.MatMul()(weights, x[:, :4])
            x[i, :4] = ops.Cast()(x_MatMul, ms.float32) / weights.sum(axis=1, keepdim=True)
            # print("x[i,:4]:")
            # print(x[i,:4])
            if redundant:
                i = i[iou.sum(1) > 1]  # require redundancy

        # print(output_boxes.shape)
        # print(output_scores.shape)
        # print(ops.ZerosLike()(output_scores).shape)
        out_x_sort = ops.Concat(1)(
            [output_boxes, output_scores.view(-1, 1), ops.ZerosLike()(output_scores.view(-1, 1))])
        output[xi] = out_x_sort[i.tolist()]  # output[xi] = x[i.tolist()]
        # print("output[xi]:")
        # print(output[xi])
        if (time.time() - t) > time_limit:
            print(f'WARNING: NMS time limit {time_limit}s exceeded')
            break  # time limit exceeded

    return output

