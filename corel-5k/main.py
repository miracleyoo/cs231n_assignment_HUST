# coding: utf-8
import torch.nn
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch import autograd
from torch.autograd import Variable
from utils import *
from data_loader import *
from train import *
from config import Config

opt = Config()

transform_train = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize((0.3854, 0.4005, 0.3472), (0.2524, 0.2410, 0.2504)),
])

transform_test = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor(),
    transforms.Normalize((0.3854, 0.4005, 0.3472), (0.2524, 0.2410, 0.2504)),
])

folder_init()
files, train_pairs, val_pairs, test_pairs = load_data()
class_names  = load_class()
weights      = get_weight(train_pairs, val_pairs, test_pairs)
if opt.USE_CUDA:
    weights  = torch.FloatTensor(weights).cuda()
else:
    weights  = torch.FloatTensor(weights)

NUM_TRAIN    = len(train_pairs)
NUM_TEST     = len(test_pairs)

trainDataset = COREL_5K(train_pairs, transform_train)
train_loader = DataLoader(dataset=trainDataset, batch_size=opt.BATCH_SIZE, shuffle=True, num_workers=opt.NUM_WORKERS, drop_last=True)

valDataset   = COREL_5K(val_pairs, transform_test)
val_loader   = DataLoader(dataset=valDataset,   batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)

testDataset  = COREL_5K(test_pairs, transform_test)
test_loader  = DataLoader(dataset=testDataset,  batch_size=opt.BATCH_SIZE, shuffle=False, num_workers=opt.NUM_WORKERS, drop_last=False)

net_list     = [models.inception_v3()]
                # models.resnet152(),
                # models.densenet121(),
                # models.resnet18()]

for net in net_list:
    net.fc = nn.Linear(2048, 374)
    net.AuxLogits.fc = nn.Linear(768, 374)
    net = training(train_loader, test_loader, class_names, net, opt.TOP_NUM)
    # net          = opt.MODEL
    # net          = torch.load(opt.NET_SAVE_PATH+'%s_model.pkl'%(net.__class__.__name__))
    # print('==> Now testing model:','%s_model.pkl'%(net.__class__.__name__))
    validating(test_loader, net, class_names)
