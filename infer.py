import os
import shutil
import time
from tqdm import tqdm
from pathlib import Path

from mindspore import ops
from mindspore.numpy import randint
import mindspore as ms
import cv2
import numpy as np
import mindspore.dataset.vision.py_transforms as transforms
from src.dataset.DemoDataset import LoadStreams, LoadImages
from src.general import non_max_suppression, scale_coords
from src.util import AverageMeter
from src.utils import show_seg_result, plot_one_box

from src.yolop import MCnet, YoloPWithLossCell
from src.model_utils.config import config


transform = [
    transforms.ToTensor(),  
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
]


def inference(opt):
    # load
    net = MCnet(is_training=False)
    checkpoint = ms.load_checkpoint(opt.weights)
    rest_checkpoint = ms.load_param_into_net(net, checkpoint, strict_load=True)
    # print(f"rest checkpoint: {rest_checkpoint}")
    net.set_train(False)

    if os.path.exists(opt.save_dir):  # output dir
        shutil.rmtree(opt.save_dir)  # delete dir
    os.makedirs(opt.save_dir)  # make new dir

    if opt.source.isnumeric():
        # cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(opt.source, img_size=opt.img_size)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(opt.source, img_size=opt.img_size)
        bs = 1  # batch_size
    # Get names and colors
    names = net.Cell.names if hasattr(net, 'cell') else net.names
    # names =  model.Module.names if hasattr(model, 'module') else model.names
    colors = [[int(randint(0, 255)) for _ in range(3)] for _ in range(len(names))]

    # Run inference
    t0 = time.time()
    vid_path, vid_writer = None, None
    img = ops.Zeros()((1, 3, opt.img_size, opt.img_size), ms.float32)
    _ = net(img)  # run once
    net.set_train(False)  # model.eval()

    inf_time = AverageMeter('inf_time')
    nms_time = AverageMeter('nms_time')

    for i, (path, img, img_det, vid_cap, shapes) in tqdm(enumerate(dataset), total=len(dataset)):
        for trans in transform:  # img = transform(img)
            img = trans(img)
        img = ms.Tensor(img, ms.float32)
        if img.ndim == 3:
            img = ops.ExpandDims()(img, 0)
        # Inference
        t1 = time.time()
        det_out, da_seg_out, ll_seg_out = net(img)
        t2 = time.time()

        inf_out, _ = det_out
        inf_time.update(t2 - t1, img.shape[0])  # img.size(0)

        # Apply NMS
        t3 = time.time()
        det_pred = non_max_suppression(inf_out, conf_thres=opt.conf_thres, iou_thres=opt.iou_thres, classes=None,
                                       agnostic=False)
        t4 = time.time()
        # if i == 0 :
        #     print("here is det_pred")
        #     print(det_pred)

        nms_time.update(t4 - t3, img.shape[0])  # img.size(0)
        det = det_pred[0]

        save_path = str(opt.save_dir + '/' + Path(path).name) if dataset.mode != 'stream' else str(
            opt.save_dir + '/' + "web.mp4")

        _, _, height, width = img.shape
        h, w, _ = img_det.shape
        pad_w, pad_h = shapes[1][1]
        pad_w = int(pad_w)
        pad_h = int(pad_h)
        ratio = shapes[1][0][1]

        # from cvt_output.commonCVT import nearest
        scale_factor = int(1 / ratio)
        da_predict = da_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        da_seg_mask_shape = (scale_factor * da_predict.shape[2], scale_factor * da_predict.shape[3])
        da_seg_mask = ops.ResizeBilinear(da_seg_mask_shape)(da_predict)
        # da_seg_mask = torch.nn.functional.interpolate(da_predict, scale_factor=int(1/ratio), mode='bilinear')
        da_seg_mask, _ = ops.ArgMaxWithValue(1)(da_seg_mask)  # _, da_seg_mask = torch.max(da_seg_mask, 1)
        # da_seg_mask = da_seg_mask.int().squeeze().cpu().numpy()
        da_seg_mask = ops.Cast()(da_seg_mask, ms.int32)  # to int
        da_seg_mask = ops.Squeeze()(da_seg_mask)  # Tensor.squeeze()
        da_seg_mask = da_seg_mask.asnumpy()  # Tensor.numpy()

        ll_predict = ll_seg_out[:, :, pad_h:(height - pad_h), pad_w:(width - pad_w)]
        ll_seg_mask_shape = (scale_factor * ll_predict.shape[2], scale_factor * ll_predict.shape[3])
        ll_seg_mask = ops.ResizeBilinear(ll_seg_mask_shape)(ll_predict)
        # ll_seg_mask = torch.nn.functional.interpolate(ll_predict, scale_factor=int(1/ratio), mode='bilinear')
        ll_seg_mask, _ = ops.ArgMaxWithValue(1)(ll_seg_mask)  # ll_seg_mask = torch.max(ll_seg_mask, 1)
        # ll_seg_mask = ll_seg_mask.int().squeeze().cpu().numpy()
        ll_seg_mask = ops.Cast()(ll_seg_mask, ms.int32)  # to int
        ll_seg_mask = ops.Squeeze()(ll_seg_mask)  # Tensor.squeeze()
        ll_seg_mask = ll_seg_mask.asnumpy()  # Tensor.numpy()

        img_det = show_seg_result(img_det, (da_seg_mask, ll_seg_mask), _, _, is_demo=True)

        if len(det):
            det[:, :4] = ops.Round()(scale_coords(img.shape[2:], det[:, :4], img_det.shape))  # tensor.round()
            for *xyxy, conf, cls in reversed(det):
                label_det_pred = f'{names[int(cls)]} {conf.asnumpy().item():.2f}'
                plot_one_box(xyxy, img_det, label=label_det_pred, color=colors[int(cls)], line_thickness=2)

        if dataset.mode == 'images':
            cv2.imwrite(save_path, img_det)

        elif dataset.mode == 'video':
            if vid_path != save_path:  # new video
                vid_path = save_path
                if isinstance(vid_writer, cv2.VideoWriter):
                    vid_writer.release()  # release previous video writer

                fourcc = 'mp4v'  # output video codec
                fps = vid_cap.get(cv2.CAP_PROP_FPS)
                h, w, _ = img_det.shape
                vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
            vid_writer.write(img_det)

        else:
            cv2.imshow('image', img_det)
            cv2.waitKey(1)  # 1 millisecond

    print('Results saved to %s' % Path(opt.save_dir))
    print('Done. (%.3fs)' % (time.time() - t0))
    print('inf : (%.4fs/frame)   nms : (%.4fs/frame)' % (inf_time.avg, nms_time.avg))

def infer_time():
    from src.yolop import YoloPWithLossCell

    ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target='CPU')
    # load
    network = MCnet(is_training=True)
    checkpoint = ms.load_checkpoint('./weights/my_End-to-end.ckpt')
    rest_checkpoint = ms.load_param_into_net(network, checkpoint, strict_load=True)
    network = YoloPWithLossCell(network, config)
    # print(f"rest checkpoint: {rest_checkpoint}")
    network.set_train()

    img = ms.Tensor(cv2.imread('./test.jpg'), ms.float32)
    img = ops.Transpose()(img, (2, 0, 1))
    if img.ndim == 3:
        img = ops.ExpandDims()(img, 0)

    for i in range(100):
        t1 = time.time()
        out = network(img)  # Ascend:77ms  CPU:2251ms
        t2 = time.time()
        print(f'fps:{(t2 - t1) * 1000} ms')



if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='./weights/my_End-to-end.ckpt', help='model.ckpt path(s)')
    parser.add_argument('--source', type=str, default='./inference/images',
                        help='source')  # file/folder   ex:inference/images
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')  # 0.25
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--save-dir', type=str, default='./runs/inference/output', help='directory to save results')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=config.divice_id)
    inference(opt)









