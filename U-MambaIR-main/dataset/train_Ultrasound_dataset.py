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
        self.SR_datas = []
        self.powerDppler_datas = []

        if is_train:
            self.SR_root = self.opt.SR_root
            self.powerDoppler_root = self.opt.powerDoppler_root
            file_names = os.listdir(self.SR_root)
            for file_name in file_names:

                powerDppler_data = getFiles(os.path.join(self.powerDoppler_root, file_name), 'power_Doppler')
                self.powerDppler_datas.append(powerDppler_data)
                SR_data = getFiles(os.path.join(self.SR_root, file_name),
                                   'super_resolution1')
                self.SR_datas.append(SR_data)
        else:
            self.SR_root = self.opt.SR1_root
            self.powerDoppler_root = self.opt.powerDoppler1_root
            file_names = os.listdir(self.SR_root)
            for file_name in file_names:

                powerDppler_data = getFiles(os.path.join(self.powerDoppler_root, file_name), 'power_Doppler')
                self.powerDppler_datas.append(powerDppler_data)
                SR_data = getFiles(os.path.join(self.SR_root, file_name), 'super_resolution1')
                self.SR_datas.append(SR_data)

    def __getitem__(self, index):
        return {'SR_data': self.SR_datas[index],'powerDppler_data': self.powerDppler_datas[index]}

    def __len__(self):
        return len(os.listdir(self.SR_root))
