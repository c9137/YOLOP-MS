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
"""YoloP train."""
import argparse
import os
import time
from datetime import datetime
import mindspore as ms
import mindspore.nn as nn
import mindspore.communication as comm

from src.yolop import YoloPWithLossCell, MCnet
from src.logger import get_logger
from src.util import AverageMeter, get_param_groups, cpu_affinity
from src.lr_scheduler import get_lr
from src.yolo_dataset import create_bdd_dataset, build_targets
from src.initializer import default_recurisive_init

from src.model_utils.config import config
from src.model_utils.device_adapter import get_device_id

# only useful for huawei cloud modelarts.
# from model_utils.moxing_adapter import moxing_wrapper, modelarts_pre_process, modelarts_post_process


ms.set_seed(1)


def init_distribute():
    comm.init()
    config.rank = comm.get_rank()
    config.group_size = comm.get_group_size()
    ms.set_auto_parallel_context(parallel_mode=ms.ParallelMode.DATA_PARALLEL, gradients_mean=True,
                                 device_num=config.group_size)


def train_preprocess():
    if config.lr_scheduler == 'cosine_annealing' and config.max_epoch > config.T_max:
        config.T_max = config.max_epoch

    # device_id = get_device_id()
    ms.set_context(mode=ms.GRAPH_MODE, device_target=config.device_target, device_id=config.device_id)

    if config.is_distributed:
        # init distributed
        init_distribute()

    # for promoting performance in GPU device
    if config.device_target == "GPU" and config.bind_cpu:
        cpu_affinity(config.rank, min(config.group_size, config.device_num))

    # logger module is managed by config, it is used in other function. e.x. config.logger.info("xxx")
    config.logger = get_logger(config.output_dir, config.rank)
    config.logger.save_args(config)


def run_train():
    train_preprocess()

    loss_meter = AverageMeter('loss')
    network = MCnet(is_training=True)
    # default is kaiming-normal

    default_recurisive_init(network)
    # load_yolov5_params(config, network)
    network = YoloPWithLossCell(network, config)

    ds = create_bdd_dataset(config, config.device_num, config.rank)
    config.logger.info('Finish loading dataset')
    
    data_loader = ds.create_tuple_iterator(do_copy=False)
    steps_per_epoch = ds.get_dataset_size()
    lr = get_lr(config, steps_per_epoch)
    opt = nn.Adam(params=get_param_groups(network))
    network = nn.TrainOneStepCell(network, opt)
    network.set_train()
    
    first_step = True
    t_end = time.time()
    
    for epoch_idx in range(config.max_epoch):
        for step_idx, data in enumerate(data_loader):
            # input, labels_det, ture_labels_num, seg_label, lane_label, paths, shapes = data
            tcls, tbox, indices, anch = build_targets(config, data[1], data[2])
            loss = network(data[0], data[1], data[3], data[4], data[6], tcls, tbox, indices, anch)
            loss_meter.update(loss.asnumpy())

            # it is used for loss, performance output per config.log_interval steps.
            if (epoch_idx * steps_per_epoch + step_idx) % config.log_interval == 0:
                time_used = time.time() - t_end
                if first_step:
                    fps = config.per_batch_size * config.group_size / time_used
                    per_step_time = time_used * 1000
                    first_step = False
                else:
                    fps = config.per_batch_size * config.log_interval * config.group_size / time_used
                    per_step_time = time_used / config.log_interval * 1000
                    
                config.logger.info('epoch[{}], iter[{}/{}], loss:{}, fps:{:.2f} imgs/sec, '
                                   'lr:{}, per step time: {}ms'.format(epoch_idx + 1, step_idx + 1, steps_per_epoch, 
                                                                       loss_meter.avg, fps, lr[step_idx], per_step_time))
                t_end = time.time()
                loss_meter.reset()
        if config.rank == 0 and step_idx % 10 == 0:
            ckpt_name = os.path.join(config.weights_output_dir, "yolop_{}_{}.ckpt".format(epoch_idx + 1, step_idx))
            ms.save_checkpoint(network, ckpt_name)

    config.logger.info('==========end training===============')

if __name__ == "__main__":
    # run_train()
    run_train()
