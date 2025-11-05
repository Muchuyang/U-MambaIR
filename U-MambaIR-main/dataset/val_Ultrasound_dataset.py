##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####
import os

import h5py
import numpy as np
import torch
import torch.utils.data as data
from util.options import Options


def getFiles(filePath, fileName):
    data = h5py.File(filePath, 'r')
    return np.array(data[fileName]).astype(np.float32)


class UltrasoundDataset(data.Dataset):
    def __init__(self, is_train = True):
        self.opt = Options()

        if is_train:
            self.SR_root = self.opt.SR_root
            self.powerDoppler_root = self.opt.powerDoppler_root
        else:
            self.SR_root = self.opt.SR1_root
            self.powerDoppler_root = self.opt.powerDoppler1_root

    def __getitem__(self, index):
        file_name = os.listdir(self.SR_root)[index]

        SR_data = getFiles(os.path.join(self.SR_root, file_name), 'super_resolution1')

        powerDppler_data = getFiles(os.path.join(self.powerDoppler_root, file_name), 'power_Doppler')

        return {'SR_data': SR_data,  'powerDppler_data': powerDppler_data}

    def __len__(self):
        return len(os.listdir(self.SR_root))
