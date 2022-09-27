# Copyright 2021-2022 Huawei Technologies Co., Ltd
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
"""YOLOP dataset."""

from pickle import TRUE
import random
import time

from tqdm import tqdm
from pathlib import Path
import json
import math
import multiprocessing

import mindspore as ms
import mindspore.ops as ops
import numpy as np
import cv2
import mindspore.dataset as ds
from src.distributed_sampler import DistributedSampler
from src.dataset.convert import convert, id_dict, id_dict_single
from src.utils import letterbox, augment_hsv, random_perspective, xyxy2xywh

# from distributed_sampler import DistributedSampler
# from dataset.convert import convert, id_dict, id_dict_single
# from utils import letterbox, augment_hsv, random_perspective, xyxy2xywh

# single_cls = True


class AutoDriveDataset:
    """
    A general Dataset for some common function
    """
    def __init__(self, cfg, is_train, inputsize=640, transform=None):
        """
        initial all the characteristic

        Inputs:
        -cfg: configurations
        -is_train(bool): whether train set or not
        -transform: ToTensor and Normalize

        Returns:
        None
        """
        self.is_train = is_train
        self.cfg = cfg
        self.transform = transform
        self.inputsize = inputsize
        self.Tensor = ds.vision.py_transforms.ToTensor()
        img_root = Path(cfg.DATASET_DATAROOT)
        label_root = Path(cfg.DATASET_LABELROOT)
        mask_root = Path(cfg.DATASET_MASKROOT)
        lane_root = Path(cfg.DATASET_LANEROOT)
        if is_train:
            indicator = cfg.DATASET_TRAIN_SET
        else:
            indicator = cfg.DATASET_TEST_SET
        self.img_root = img_root / indicator
        self.label_root = label_root / indicator
        self.mask_root = mask_root / indicator
        self.lane_root = lane_root / indicator
        # self.label_list = self.label_root.iterdir()
        self.mask_list = self.mask_root.iterdir()

        self.db = []

        self.data_format = cfg.DATASET_DATA_FORMAT

        self.scale_factor = cfg.DATASET_SCALE_FACTOR
        self.rotation_factor = cfg.DATASET_ROT_FACTOR
        self.flip = cfg.DATASET_FLIP
        self.color_rgb = cfg.DATASET_COLOR_RGB

        # self.target_type = cfg.MODEL.TARGET_TYPE
        self.shapes = np.array(cfg.DATASET_ORG_IMG_SIZE)  # [720, 1280]

    def _get_db(self):
        """
        finished on children Dataset(for dataset which is not in Bdd100k format, rewrite children Dataset)
        """
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir):
        """
        finished on children dataset
        """
        raise NotImplementedError

    def __len__(self,):
        """
        number of objects in the dataset
        """
        return len(self.db)

    def __getitem__(self, idx):
        """
        Get input and groud-truth from database & add data augmentation on input

        Inputs:
        -idx: the index of image in self.db(database)(list)
        self.db(list) [a,b,c,...]
        a: (dictionary){'image':, 'information':}

        Returns:
        -image: transformed image, first passed the data augmentation in __getitem__ function(type:numpy), then apply self.transform
        -target: ground truth(det_gt,seg_gt)

        function maybe useful
        cv2.imread
        cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        cv2.warpAffine
        """
        data = self.db[idx]
        img = cv2.imread(data["image"], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # (720, 1280, 3)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # seg_label = cv2.imread(data["mask"], 0)
        # if self.cfg.num_seg_class == 3:
        #     seg_label = cv2.imread(data["mask"])
        # else:
        seg_label = cv2.imread(data["mask"], 0)  # (720, 1280)  灰度图片，仅是二维矩阵
        lane_label = cv2.imread(data["lane"], 0)  # (720, 1280)

        resized_shape = self.inputsize  # 640
        if isinstance(resized_shape, list):
            resized_shape = max(resized_shape)
        h0, w0 = img.shape[:2]  # orig hw (720, 1280)
        r = resized_shape / max(h0, w0)  # resize image to img_size  0.5
        if r != 1:  # always resize down, only resize up if training with augmentation
            interp = cv2.INTER_AREA if r < 1 else cv2.INTER_LINEAR
            img = cv2.resize(img, (int(w0 * r), int(h0 * r)), interpolation=interp)
            seg_label = cv2.resize(seg_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
            lane_label = cv2.resize(lane_label, (int(w0 * r), int(h0 * r)), interpolation=interp)
        h, w = img.shape[:2]  # (360, 640)

        #  auto=True  (384, 640, 3)   (384, 640)   (384, 640)
        #  auto=False  (640, 640, 3)   (640, 640)   (640, 640)
        (img, seg_label, lane_label), ratio, pad = letterbox((img, seg_label, lane_label), resized_shape, auto=TRUE, scaleup=self.is_train)

        # (720, 1280)  ，  （2, 2）
        # shapes = (h0, w0), ((h / h0, w / w0), pad)  # for COCO mAP rescaling
        # ratio = (w / w0, h / h0)
        # print(resized_shape)

        det_label = data["label"]
        labels = []

        if det_label.size > 0:
            # Normalized xywh to pixel xyxy format
            labels = det_label.copy()
            labels[:, 1] = ratio[0] * w * (det_label[:, 1] - det_label[:, 3] / 2) + pad[0]  # pad width
            labels[:, 2] = ratio[1] * h * (det_label[:, 2] - det_label[:, 4] / 2) + pad[1]  # pad height
            labels[:, 3] = ratio[0] * w * (det_label[:, 1] + det_label[:, 3] / 2) + pad[0]
            labels[:, 4] = ratio[1] * h * (det_label[:, 2] + det_label[:, 4] / 2) + pad[1]

        if self.is_train:
            combination = (img, seg_label, lane_label)
            (img, seg_label, lane_label), labels = random_perspective(
                combination=combination,
                targets=labels,
                degrees=self.cfg.DATASET_ROT_FACTOR,
                translate=self.cfg.DATASET_TRANSLATE,
                scale=self.cfg.DATASET_SCALE_FACTOR,
                shear=self.cfg.DATASET_SHEAR
            )
            augment_hsv(img, hgain=self.cfg.DATASET_HSV_H, sgain=self.cfg.DATASET_HSV_S, vgain=self.cfg.DATASET_HSV_V)
            # img, seg_label, labels = cutout(combination=combination, labels=labels)

            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

            # if self.is_train:
            # random left-right flip
            lr_flip = True
            if lr_flip and random.random() < 0.5:
                img = np.fliplr(img)
                seg_label = np.fliplr(seg_label)
                lane_label = np.fliplr(lane_label)
                if len(labels):
                    labels[:, 1] = 1 - labels[:, 1]

            # random up-down flip
            ud_flip = False
            if ud_flip and random.random() < 0.5:
                img = np.flipud(img)
                seg_label = np.filpud(seg_label)
                lane_label = np.filpud(lane_label)
                if len(labels):
                    labels[:, 2] = 1 - labels[:, 2]

        else:
            if len(labels):
                # convert xyxy to xywh
                labels[:, 1:5] = xyxy2xywh(labels[:, 1:5])

                # Normalize coordinates 0 - 1
                labels[:, [2, 4]] /= img.shape[0]  # height
                labels[:, [1, 3]] /= img.shape[1]  # width

        # labels_out = ops.Zeros()((self.cfg.MAX_BOXES, 6), ms.float32)

        labels_out = np.zeros((self.cfg.MAX_BOXES, 6), np.float32)
        if len(labels):
            if labels.shape[0] < self.cfg.MAX_BOXES:
                labels_out[:labels.shape[0], 1:] = labels  # #######
                ture_labels_num = labels.shape[0]
            else:
                labels_out[:, 1:] = labels[:self.cfg.MAX_BOXES]
                ture_labels_num = self.cfg.MAX_BOXES
        else:
            ture_labels_num = 0

        labels_out = ms.Tensor.from_numpy(labels_out)

        # Convert
        # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        # img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        # seg_label = np.ascontiguousarray(seg_label)
        # if idx == 0:
        #     print(seg_label[:,:,0])

        # if self.cfg.num_seg_class == 3:
        #     _, seg0 = cv2.threshold(seg_label[:,:,0], 128, 255, cv2.THRESH_BINARY)
        #     _, seg1 = cv2.threshold(seg_label[:,:,1], 1, 255, cv2.THRESH_BINARY)
        #     _, seg2 = cv2.threshold(seg_label[:,:,2], 1, 255, cv2.THRESH_BINARY)
        # else:

        _, seg1 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY)
        _, seg2 = cv2.threshold(seg_label, 1, 255, cv2.THRESH_BINARY_INV)
        _, lane1 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY)
        _, lane2 = cv2.threshold(lane_label, 1, 255, cv2.THRESH_BINARY_INV)


        # if self.cfg.num_seg_class == 3:
        #     seg0 = self.Tensor(seg0)
        seg1 = self.Tensor(seg1)
        seg2 = self.Tensor(seg2)
        seg_label = np.stack([seg2[0], seg1[0]])  #######

        lane1 = self.Tensor(lane1)
        lane2 = self.Tensor(lane2)
        lane_label = np.stack([lane2[0], lane1[0]])

        seg_label = ms.Tensor(seg_label)
        lane_label = ms.Tensor(lane_label)

        # if self.cfg.num_seg_class == 3:
        #     seg_label = stack((seg0[0], seg1[0], seg2[0]))
        # else:


        # target = [labels_out, seg_label, lane_label]
        # img = self.transform(img)
        shapes = h0, w0, h / h0, w / w0, pad[0], pad[1]
        # shapes_tensor = mindspore.Tensor(shapes)

        # return img, labels_out, seg_label, lane_label, data["image"], h0, w0, h / h0, w / w0, pad[0], pad[1]
        return img, labels_out, ture_labels_num, seg_label, lane_label, data["image"], shapes


    def select_data(self, db):
        """
        You can use this function to filter useless images in the dataset

        Inputs:
        -db: (list)database

        Returns:
        -db_selected: (list)filtered dataset
        """
        db_selected = ...
        return db_selected

    @staticmethod
    def collate_fn(batch):
        img, label, paths, shapes = zip(*batch)
        label_det, label_seg, label_lane = [], [], []
        stack = ops.Stack(axis=0)
        for i, l in enumerate(label):
            l_det, l_seg, l_lane = l
            l_det[:, 0] = i  # add target image index for build_targets()
            label_det.append(l_det)
            label_seg.append(l_seg)
            label_lane.append(l_lane)
        return stack(img), [ops.Concat(axis=0)(label_det), stack(label_seg), stack(label_lane)], paths, shapes


class BddDataset(AutoDriveDataset):
    def __init__(self, cfg, is_train, inputsize, transform=None):
        super().__init__(cfg, is_train, inputsize, transform)
        self.db = self._get_db()
        self.cfg = cfg

    def _get_db(self):
        """
        get database from the annotation file

        Inputs:

        Returns:
        gt_db: (list)database   [a,b,c,...]
                a: (dictionary){'image':, 'information':, ......}
        image: image path
        label: [cls_id, center_x//256, center_y//256, w//256, h//256] 256=IMAGE_SIZE
        mask: path of the drivable area segmetation label  (img path)
        lane: path of the lane line segmetation label  (img path)
        """
        print('building database...')
        gt_db = []
        height, width = self.shapes  # [720, 1280]
        for mask in tqdm(list(self.mask_list)):
            mask_path = str(mask)
            label_path = mask_path.replace(str(self.mask_root), str(self.label_root)).replace(".png", ".json")
            image_path = mask_path.replace(str(self.mask_root), str(self.img_root)).replace(".png", ".jpg")
            lane_path = mask_path.replace(str(self.mask_root), str(self.lane_root))
            with open(label_path, 'r') as f:
                label = json.load(f)
            data = label['frames'][0]['objects']
            data = self.filter_data(data)
            gt = np.zeros((len(data), 5))
            for idx, obj in enumerate(data):
                category = obj['category']
                if category == "traffic light":
                    color = obj['attributes']['trafficLightColor']
                    category = "tl_" + color
                if category in id_dict.keys():
                    x1 = float(obj['box2d']['x1'])
                    y1 = float(obj['box2d']['y1'])
                    x2 = float(obj['box2d']['x2'])
                    y2 = float(obj['box2d']['y2'])
                    cls_id = id_dict[category]
                    if self.cfg.SINGLE_CLS:
                        cls_id = 0
                    gt[idx][0] = cls_id
                    box = convert((width, height), (x1, x2, y1, y2))
                    gt[idx][1:] = list(box)

            rec = [{
                'image': image_path,
                'label': gt,
                'mask': mask_path,
                'lane': lane_path
            }]

            gt_db += rec
        print('database build finish')
        return gt_db

    def filter_data(self, data):
        remain = []
        for obj in data:
            if 'box2d' in obj.keys():  # obj.has_key('box2d'):
                if self.cfg.SINGLE_CLS:
                    if obj['category'] in id_dict_single.keys():
                        remain.append(obj)
                else:
                    remain.append(obj)
        return remain

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        """
        """
        pass


def build_targets(cfg, ture_targets, ture_labels_num):
    ture_targets = ture_targets.asnumpy()
    ture_labels_num = ture_labels_num.asnumpy()
    g = 0.5
    off = np.array([[0, 0],
                         [1, 0], [0, 1], [-1, 0], [0, -1],  # j,k,l,m
                         # [1, 1], [1, -1], [-1, 1], [-1, -1],  # jk,jm,lk,lm
                         ], np.float32) * g  # offsets
    anchor_scales = np.array(cfg.anchor_scales, np.float32)
    stride = np.array([8.0, 16.0, 32.0])
    predictions_shape = np.array([[80, 80], [40, 40], [20, 20]])
    # build_targets_max = cfg['BUILD_TARGETS_MAX']

    labels_d = []
    for i in range(len(ture_targets)):
        ture_targets[i, :, 0] = i
        # box_num = ture_labels_num[i]
        labels_d.append(ture_targets[i, :ture_labels_num[i]])  ########
    targets = np.concatenate(labels_d, axis=0)
    # print(targets.shape)

    # Build targets for compute_loss(), input targets(image, class, x, y, w, h)
    na, nt = 3, targets.shape[0]  # number of anchors, number of targets
    tcls, tbox, indices, anch = [], [], [], []
    gain = np.ones(7)  # normalized to gridspace gain
    ai = np.arange(na).astype(float).reshape(na, 1).repeat(nt, 1)  # same as .repeat_interleave(nt)
    temp = np.array([targets, targets, targets])
    ttargets = np.concatenate((temp, ai[:, :, None]), 2)  # append anchor indices

    for i in range(3):
        anchors = anchor_scales[i * 3:i * 3 + 3, :] / stride[i]  # [3, 2]
        gain[2:6] = predictions_shape[i][[1, 0, 1, 0]]  # xyxy gain
        # Match targets to anchors
        t = ttargets * gain

        if nt:
            # Matches
            r = t[:, :, 4:6] / anchors[:, None]  # wh ratio
            j = np.maximum(r, 1. / r).max(axis=2) < 4.0  # compare
            # j = wh_iou(anchors, t[:, 4:6]) > model.hyp['iou_t']  # iou(3,n)=wh_iou(anchors(3,2), gwh(n,2))
            t = t[j]  # filter

            # print(t.shape)

            # Offsets
            gxy = t[:, 2:4]  # grid xy
            gxi = gain[[2, 3]] - gxy  # inverse
            j, k = ((gxy % 1. < g) & (gxy > 1.)).T
            l, m = ((gxi % 1. < g) & (gxi > 1.)).T
            j = np.stack((np.ones_like(j), j, k, l, m))
            t = np.array([t, t, t, t, t])[j]
            # t = t.repeat((5, 1, 1))[j]
            offsets = (np.zeros_like(gxy)[None] + off[:, None])[j]
        else:
            t = ttargets[0]
            offsets = 0

        # Define
        b, c = t[:, :2].astype(np.int32).T  # image, class
        gxy = t[:, 2:4]  # grid xy
        gwh = t[:, 4:6]  # grid wh
        gij = (gxy - offsets).astype(np.int32)
        gi, gj = gij.T  # grid xy indices

        # Append
        a = t[:, 6].astype(np.int32)  # anchor indices

        indices.append((ms.Tensor(b), ms.Tensor(a), ms.Tensor(gj.clip(0, gain[3] - 1), ms.int32), ms.Tensor(gi.clip(0, gain[2] - 1), ms.int32)))  # image, anchor, grid indices
        if b.shape[0] != 0:
            tbox.append(ms.Tensor(np.concatenate((gxy - gij, gwh), 1), ms.float32))  # box
            anch.append(ms.Tensor(anchors[a]))  # anchors
            tcls.append(ms.Tensor(c))
        else:
            tbox.append(ms.Tensor(0, ms.float32))
            anch.append(ms.Tensor(0, ms.float32))
            tcls.append(ms.Tensor(0, ms.float32))

    # return ms.Tensor(targets), tcls, tbox, indices, anch
    return tcls, tbox, indices, anch


def create_bdd_dataset(cfg, device_num, rank, is_training=True, shuffle=True):
    cv2.setNumThreads(0)
    ds.config.set_enable_shared_mem(True)

    bdd = BddDataset(cfg, is_train=is_training, inputsize=640)
    distributed_sampler = DistributedSampler(len(bdd), device_num, rank, shuffle=shuffle)

    cores = multiprocessing.cpu_count()  # 192
    num_parallel_workers = int(cores / device_num)

    # h0, w0, h / h0, w / w0, pad[0], pad[1]
    # dataset_column_names = ["image", "labels_det", "seg_label", "lane_label", "path", "h0", "w0", "ratio_h", "ratio_w", "pad_0", "pad_1"]
    dataset_column_names = ["image", "labels_det", "ture_labels_num", "seg_label", "lane_label", "path", "shape"]

    dataset = ds.GeneratorDataset(bdd, column_names=dataset_column_names, sampler=distributed_sampler, python_multiprocessing=False, num_parallel_workers=min(8, num_parallel_workers))
    mean = [m * 255 for m in [0.485, 0.456, 0.406]]
    std = [s * 255 for s in [0.229, 0.224, 0.225]]
    hwc_to_chw = ds.vision.c_transforms.HWC2CHW()
    dataset = dataset.map([ds.vision.c_transforms.Normalize(mean, std), hwc_to_chw],
                          input_columns=['image'])
    dataset = dataset.batch(cfg.per_batch_size, drop_remainder=True)
    return dataset


if __name__ == '__main__':
    from model_utils.config import config

    ds = create_bdd_dataset(config, 1, 0)
    print(ds.get_dataset_size())
    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU')

    data_loader = ds.create_tuple_iterator(do_copy=False)
    t4 = 0
    t0 = time.time()
    for step_idx, data in enumerate(data_loader):
        t1 = time.time()
        # input, labels_det, ture_labels_num, tcls, tbox, indices, anch, build_shape, seg_label, lane_label, paths, shapes = data
        input, labels_det, ture_labels_num, seg_label, lane_label, paths, shapes = data
        t2 = time.time()
        tcls, tbox, indices, anch = build_targets(config, labels_det, ture_labels_num)
        t3 = time.time()
        if step_idx == 0:
            print(f'step:{step_idx}\tgetitem:{t1-t0}\tunzip:{t2-t1}\tbuild_target:{t3-t2}')
        else:
            print(f'step:{step_idx}\tgetitem:{t1-t4}\tunzip:{t2-t1}\tbuild_target:{t3-t2}')
        t4 = time.time()
