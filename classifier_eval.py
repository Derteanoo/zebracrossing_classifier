from torch import nn
from torch.utils.data import DataLoader
from utils import Evaler
from network import mobilenet_v2, ghost_net, mobilenet_v3
from data import Zebra

# Hyper-params
data_root = './data/'
batch_size = 32  # batch_size per GPU, if use GPU mode; resnet34: batch_size=120
num_workers = 2

# Set Eval parameters
params = Evaler.EvalerParams()
params.gpus = [0]  # set 'params.gpus=[]' to use CPU model. if len(params.gpus)>1, default to use params.gpus[0] to test
params.ckpt = './models/ckpt_epoch_30.pth'  #'./models/ckpt_epoch_400_res34.pth'

#load data
print("Loading dataset....")
eval_data = Zebra(data_root, train=False)

batch_size = batch_size if len(params.gpus) == 0 else batch_size*len(params.gpus)
epoch = len(eval_data) // batch_size

eval_dataloader = DataLoader(eval_data, batch_size=batch_size, shuffle=False, num_workers=num_workers)
print('val dataset len: {}'.format(len(eval_dataloader.dataset)))

# models
#model = mobilenet_v2(pretrained=False, width_mult=0.25, num_classes=2)
#model = ghost_net(num_classes=2, width_mult=0.25)
model = mobilenet_v3(num_classes=2)

# Test
evaler = Evaler(model, params, eval_dataloader, epoch)
evaler.eval()
