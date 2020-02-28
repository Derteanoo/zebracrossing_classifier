from torch import nn
from utils import Tester
from network import mobilenet_v2, ghost_net

# Set Test parameters
params = Tester.TestParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_30.pth'  #'./models/ckpt_epoch_400_res34.pth'
params.testdata_dir = './testimg/'

# models
# model = resnet34(pretrained=False, num_classes=1000)  # batch_size=120, 1GPU Memory < 7000M
# model.fc = nn.Linear(512, 6)
model = mobilenet_v2(pretrained=False, width_mult=0.25, num_classes=2)

# Test
tester = Tester(model, params)
tester.test()
