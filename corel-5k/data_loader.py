# coding: utf-8
from torch.utils.data import Dataset
from skimage import io,transform
from PIL import Image
import numpy as np
from config import Config

opt = Config()

class COREL_5K(Dataset):
    def __init__(self, data, transform=None):
        super(COREL_5K, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data_path, label = self.data[index]
        label = np.array(label) - 1 # the minus 1 here turn 1~374 index into 0~373
        label = np.sum(np.eye(opt.NUM_CLASSES)[label], axis=0) # 374 is the number of labels
        img = Image.open(data_path)
        if self.transform:
            sample = self.transform(np.array(img))
        return sample, label.astype(np.float32), np.array(img.resize((128,128))), (img.size)