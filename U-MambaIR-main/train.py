##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####
import random
import numpy as np
from collections import defaultdict

from model import new_network, new_network_single
from util.options import Options
from dataset import train_Ultrasound_dataset, val_Ultrasound_dataset, Ultrasound_dataloader
from torch.utils.tensorboard import SummaryWriter
from util import util
# from tqdm import tqdm
import torch
import time
import os

opt = Options()


current_time = time.localtime()
exp_folder = opt.exp_folder if opt.exp_folder else time.strftime("%Y-%m-%d-%H-%M-%S", current_time)   # 文件夹按时间命名
exp_folder = os.path.join( opt.exp_path, exp_folder)
opt.exp_folder = exp_folder

model_save_path = os.path.join(exp_folder, 'model')
opt.model_save_path = model_save_path
log_save_path = os.path.join(exp_folder, 'log')
opt.log_save_path = log_save_path
mat_save_path = os.path.join(exp_folder, 'mat')

if not os.path.exists(exp_folder):
    os.makedirs(exp_folder)
    print('初始化保存路径')
    os.makedirs(model_save_path)
    print('初始化model save路径')
    os.makedirs(log_save_path)
    print('初始化log save路径')
    """专门用来保存生成的mat"""
    os.makedirs(mat_save_path)
    print('初始化mat save路径')

logger = SummaryWriter( log_dir=os.path.join(log_save_path, 'exp'))

# torch.manual_seed(seed)
seed = opt.seed
if seed is None:
    seed = random.randint(1, 10000)
    # seed = 42
util.set_random_seed(seed)  # seed生效
print('初始化随机seed:' , seed)

logger.add_text('seed', str(seed))

torch.backends.cudnn.benckmark = True

dataset = train_Ultrasound_dataset.UltrasoundDataset()
dataLoader = Ultrasound_dataloader.getDataloader(dataset, opt)
print('完成训练集加载')

val_dataset = train_Ultrasound_dataset.UltrasoundDataset(is_train=False)
val_dataLoader = Ultrasound_dataloader.getDataloader(val_dataset, opt, is_Train=False)
print('完成测试集加载')

model = new_network_single.Network(opt)

if opt.start_epoch > 11:
    model.loadModel(opt.model_weight_path)
elif opt.if_init_model:
    model.initModel(init_type='normal',gain=0.02)


print('模型开始训练')
for epoch in (range(opt.start_epoch, opt.end_epoch + 1)):
    losses = defaultdict(int)
    step = 0
    start, end,  runtime = 0, 0, 0
    start = time.perf_counter()
    for idx, train_data in enumerate(dataLoader):
        SR_data = train_data['SR_data']
        powerDppler_data = train_data['powerDppler_data']

        loss_dict = model.feedData(SR_data, powerDppler_data, epoch, is_train = True)
        for k, v in loss_dict.items():
            losses[k] += v.item()
        step += 1
        print('epoch:{}, iter:{}, loss:{}'.format(epoch, idx, loss_dict["total_loss"]))
    for k, v in losses.items():
        logger.add_scalar(k, v / step, epoch)
    for k, v in model.getLR().items():
        logger.add_scalar(k, v, epoch)

    end = time.perf_counter()
    runtime = end - start
    print('epoch:{}, runtime:{}'.format(epoch, runtime))

    if epoch >= 180 and epoch % opt.save_model_step == 0:
        model.saveModel(os.path.join(model_save_path, str(epoch) + '_model_weight.pth'))
        print("保存pth完毕")

    if epoch >= 140 and epoch % opt.save_mat_step == 0:
        model.save_img(powerDpplerPath=opt.mat_path,
                       matName=opt.mat_name,
                       filePath=mat_save_path,
                       epoch=epoch)
        print("保存.mat完毕")

    if epoch % opt.val_step == 0:
        print('开始测试')
        losses1 = defaultdict(int)
        step = 0
        with torch.no_grad():
            for idx, data in enumerate(val_dataLoader):
                SR_data = data['SR_data']
                powerDppler_data = data['powerDppler_data']
                # val
                loss_dict1 = model.feedData(SR_data,powerDppler_data, epoch, is_train=False)
                for k, v in loss_dict1.items():
                    losses1[k] += v.item()
                step += 1
                print('val_epoch:{}, val_iter:{}, ValLoss:{}'.format(epoch, idx, loss_dict1["val_loss"].item()))

                torch.cuda.empty_cache()
        for k, v in losses1.items():
            logger.add_scalar(k, v / step, epoch)

    logger.flush()
logger.close()