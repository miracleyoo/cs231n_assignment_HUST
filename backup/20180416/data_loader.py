# coding: utf-8
from torch.utils.data import Dataset
from skimage import io,transform
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
        label = np.array(label) - 1
        label = np.sum(np.eye(opt.NUM_CLASSES)[label], axis=0) # 374 is the number of labels
        img = io.imread(data_path)
        if self.transform:
            img = self.transform(img)
        return img, label.astype(np.float32)