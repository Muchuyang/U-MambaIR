##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####

from model import new_network_single
from util.options import Options
import h5py
import torch
import numpy as np

opt = Options()

model = new_network_single.Network(opt)

pth = r'D:\结果\models\2025-02-19-20-37-25 2scan\model\440_model_weight.pth'
model.loadModel(pth)

mat_path = r'E:\DL_Dataset\single_new_new\test\pd'

mat_name = 'power_Doppler'
mat_save_path = r'D:\结果\results\psnr_ssim\2chan\440_sim'
model.save_img(powerDpplerPath= mat_path,
                       matName= mat_name,
                       filePath= mat_save_path,
                       epoch=440)