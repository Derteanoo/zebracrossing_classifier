from __future__ import print_function

import os
from PIL import Image
from .log import logger

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms.functional as tv_F


class TestParams(object):
    # params based on your local env
    gpus = []  # default to use CPU mode

    # loading existing checkpoint
    ckpt = './models/ckpt_epoch_800_res101.pth'     # path to the ckpt file

    testdata_dir = './testimg/'

class Tester(object):

    TestParams = TestParams

    def __init__(self, model, test_params):
        assert isinstance(test_params, TestParams)
        self.params = test_params

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

    def test(self):

        img_list = os.listdir(self.params.testdata_dir)

        for img_name in img_list:
            print('Processing image: ' + img_name)

            img = Image.open(os.path.join(self.params.testdata_dir, img_name)).convert("RGB")
            img.save('050_1.png')
            img = tv_F.to_tensor(tv_F.resize(img, (32, 32)))
            img = tv_F.normalize(img, [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            img_input = Variable(torch.unsqueeze(img, 0))
            if len(self.params.gpus) > 0:
                img_input = img_input.cuda()

            output = self.model(img_input)
            score = F.softmax(output, dim=1)
            _, prediction = torch.max(score.data, dim=1)
            
            print('Prediction number: ' + str(prediction[0]))

    def _load_ckpt(self, ckpt):
        #备注：训练时用多gpu进行分布式训练，保存模型时会使用nn.DataParallel,
        # 加载模型的Key值中会多module.这7个字符。所以加载时，去掉key值前7个字符
        state_dict = torch.load(ckpt)
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k,v in state_dict.items():
            name = k[7:]
            new_state_dict[name] = v
        self.model.load_state_dict(new_state_dict)
