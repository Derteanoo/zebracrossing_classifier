from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch as t
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F
from torchnet import meter
import numpy as np


class EvalerParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_800_res101.pth'     # path to the ckpt file

    testdata_dir = './data/image/'

class Evaler(object):

    EvalerParams = EvalerParams

    def __init__(self, model, eval_params, eval_data, epoch):
        assert isinstance(eval_params, EvalerParams)

        self.params = eval_params
        self.epoch = epoch
        self.acc = []

        # Data loaders
        self.eval_data = eval_data

        # load model
        self.model = model
        
        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # set CUDA_VISIBLE_DEVICES, 1 GPU is enough
        if len(self.params.gpus) > 0:
            gpu_test = str(self.params.gpus[0])
            os.environ['CUDA_VISIBLE_DEVICES'] = gpu_test
            logger.info('Set CUDA_VISIBLE_DEVICES to {}...'.format(gpu_test))
            self.model = self.model.cuda()

        self.model.eval()

    def eval(self):

        for epoch in range(self.epoch):

            eval_cm, eval_accuracy = self._eval_one_epoch()

            self.acc.append(eval_accuracy)

        acc_ = np.mean(self.acc)

        print("acc = ", acc_)

    def _load_ckpt(self, ckpt):
        #备注：训练时用多gpu进行分布式训练，保存模型时会使用nn.DataParallel,
        # 加载模型的Key值中会多module.这7个字符。所以加载时，去掉key值前7个字符
        state_dict = t.load(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)

    def _eval_one_epoch(self):

        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(2)
        logger.info('Eval on validation set...')

        for step, (data, label) in enumerate(self.eval_data):

            # val model
            inputs = Variable(data, volatile=True)
            target = Variable(label.type(t.LongTensor), volatile=True)
            if len(self.params.gpus) > 0:
                inputs = inputs.cuda()
                target = target.cuda()

            score = self.model(inputs)
            confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

        self.model.train()
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
        return confusion_matrix, accuracy
