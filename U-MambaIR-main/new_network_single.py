##### This paper is currently under submission. If it gets accepted, we will release the complete code as soon as possible. #####
import os
import cv2
import torch
import torch.nn as nn
from collections import defaultdict
import model.lr_scheduler as lr_scheduler

from analysis.model_zoo.mambaIR_8scans import MambaIR


from model.Losses.perceptual_loss import PerceptualLoss
from model.Losses.gan_loss import GANLoss,MultiScaleGANLoss
from model.Losses.discriminator import UNetDiscriminatorSN, MultiScaleDiscriminator

import h5py
import numpy as np

import torch.nn.init as init
import h5py
import numpy as np
import time

class Network:
    def __init__(self, opt):
        super(Network, self).__init__()

        self.opt = opt

        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

        # MambaIR
        self.model = MambaIR(img_size=,
                   patch_size=,
                   in_chans=,
                   embed_dim=,
                   depths=(),
                   mlp_ratio=,
                   drop_rate=,
                   norm_layer=nn.LayerNorm,
                   patch_norm=True,
                   use_checkpoint=False,
                   upscale=,    #
                   img_range=,
                   upsampler=,
                   resi_connection=)


        # self.model = UNet(1)

        self.model.to(self.device)

        # self.discriminator = UNetDiscriminatorSN(1).cuda()
        self.discriminator = MultiScaleDiscriminator(1).cuda()


        def get_loss(loss_type):
            if loss_type == 'l1':   # 1
                return nn.L1Loss(reduction='sum')
            elif loss_type == 'l2':
                return nn.MSELoss(reduction='sum')
            elif loss_type == 'cb':
                return CharbonnierLoss()
            elif loss_type == 'lp':
                return LapLoss(max_levels=5)
            elif loss_type == 'VGG':  # 2
                return PerceptualLoss()
            elif loss_type == 'resnet':
                return Anime_PerceptualLoss()
            elif loss_type == 'gan':  # 3
                # return GANLoss(gan_type="vanilla", loss_weight=0.2)
                return MultiScaleGANLoss(gan_type="lsgan",loss_weight=0.2)
            elif loss_type == 'style':
                return Perceptual_Style_Loss()
            elif loss_type == 'corr':
                return Loss_corr()
            elif loss_type == 'KL':
                return KL_loss()
            else:
                raise NotImplementedError('Loss type [{:s}] is not recognized.'.format(loss_type))

        self.loss = [get_loss(opt.loss1_type), get_loss(opt.loss2_type), get_loss(opt.loss3_type)]

        # optimizers
        optimizers_weight = opt.optimizers_weight if opt.optimizers_weight else 0
        optim_params = []
        for k, v in self.model.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer = torch.optim.Adam(optim_params, lr=opt.lr_optimizer,
                                          weight_decay=optimizers_weight,
                                          betas=(opt.beta1, opt.beta2))
        # dis optimizers
        optimizers_weight_d = opt.optimizers_weight_d if opt.optimizers_weight_d else 0
        optim_params_discriminator = []
        for k, v in self.discriminator.named_parameters():
            if v.requires_grad:
                optim_params_discriminator.append(v)
        self.optimizer_discriminator = torch.optim.Adam(optim_params_discriminator, lr=opt.lr_optimizer_d,
                                                        weight_decay=optimizers_weight_d,
                                                        betas=(opt.beta1_d, opt.beta2_d))
        # schedulers
        self.lr_decay = lr_scheduler.WarmupMultiStepLR(
            self.optimizer, milestones = opt.milestones, gamma = opt.gamma, warmup_factor = opt.warmup_factor,
            warmup_iters=opt.warmup_iters, warmup_method=opt.warmup_method)


        self.lr_decay_discriminator = lr_scheduler.WarmupMultiStepLR(
            self.optimizer_discriminator, milestones = opt.milestones_d, gamma = opt.gamma_d, warmup_factor = opt.warmup_factor_d,
            warmup_iters=opt.warmup_iters_d, warmup_method=opt.warmup_method_d)

    def saveModel(self, model_name):
        torch.save(self.model.state_dict(), model_name)

    def loadModel(self, model_name):
        self.model.load_state_dict(torch.load(model_name))

    def initModel(self, init_type='normal', gain=0.02):
        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1:
                init.normal_(m.weight.data, 1.0, gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        self.model.apply(init_func)
        print("模型完成初始化")
        self.discriminator.apply(init_func)
        print("判别器完成初始化")


    def feedData(self, SR_data, powerDppler_data, epoch, is_train = True):
        if is_train:
            self.model.train()
            self.discriminator.train()

            self.optimizer.zero_grad()
            for p in self.discriminator.parameters():
                p.requires_grad = False

            powerDppler_data = powerDppler_data.to(self.device)
            powerDppler_data = torch.unsqueeze(powerDppler_data, dim=1)

            out = self.model(powerDppler_data)

            SR_data = SR_data.to(self.device)
            SR_data = torch.unsqueeze(SR_data, dim=1)

            losses = self.calculate_loss(SR_data, out, epoch)

            losses["total_loss"].backward()
            self.optimizer.step()
            self.lr_decay.step()

            if epoch >=100:
                self.optimizer_discriminator.zero_grad()
                for p in self.discriminator.parameters():
                    p.requires_grad = True

                loss_dis = self.dis_loss(SR_data, out)

                loss_dis["loss_d_sum"].backward()

                self.optimizer_discriminator.step()
                self.lr_decay_discriminator.step()
                # #
                losses = dict(losses)
                losses.update(loss_dis)

        else:
            self.model.eval()
            self.discriminator.eval()

            powerDppler_data = powerDppler_data.to(self.device)
            powerDppler_data = torch.unsqueeze(powerDppler_data, dim=1)
            out = self.model(powerDppler_data)

            SR_data = SR_data.to(self.device)
            SR_data = torch.unsqueeze(SR_data, dim=1)

            losses = self.calculate_valloss(SR_data, out, epoch)

        return losses

    def expand_dims(self, x):
        x = x.repeat(1, 3, 1, 1)
        return x

    def calculate_loss(self, SR_data,  out, epoch):

        loss_pixel = self.loss[0](out, SR_data)
        loss_adv = self.loss[2](self.discriminator(out), True, is_disc=False)  # loss_weight (self.gan_loss_weight) is included
        out = self.expand_dims(out)
        SR_data = self.expand_dims(SR_data)
        loss_perceptual = self.loss[1](out, SR_data)


        return {"total_loss": losses, "loss_pixel": loss_pixel, "loss_perceptual": loss_perceptual,"loss_adv": loss_adv }

    def calculate_valloss(self, SR_data,  out, epoch):
        loss_pixel = self.loss[0](out, SR_data)
        loss_adv = self.loss[2](self.discriminator(out), True,  is_disc=False)  # loss_weight (self.gan_loss_weight) is included
        out1 = self.expand_dims(out)
        SR_data1 = self.expand_dims(SR_data)
        loss_perceptual = self.loss[1](out1, SR_data1)

        return {"val_loss": losses}

    def dis_loss(self, SR_data, out):

        # Discriminarter
        real_d = self.discriminator(SR_data)
        loss_dis_t = self.loss[2](real_d, True, is_disc=True)
        fake_d = self.discriminator(out.detach())
        loss_dis_f = self.loss[2](fake_d, False, is_disc=True)
        loss_diss_sum = 0.5 * (loss_dis_t + loss_dis_f)

        return {"loss_d_sum": loss_diss_sum, "loss_dis_t": loss_dis_t, "loss_dis_f": loss_dis_f}

    def getFiles(self, filePath, matName):
        data = h5py.File(filePath, 'r')
        return torch.tensor(np.array(data[matName]).astype(np.float32)).to(self.device)

    def save_img(self, powerDpplerPath, matName, filePath, epoch=100):
        self.model.eval()
        dopplerNamse = os.listdir(powerDpplerPath)
        print('开始保存mat')
        for name in dopplerNamse:
            powerDppler_data = self.getFiles(os.path.join(powerDpplerPath, name), matName)
            powerDppler_data = torch.unsqueeze(powerDppler_data, dim=0)
            powerDppler_data = torch.unsqueeze(powerDppler_data, dim=0)
            out = self.model(powerDppler_data).to('cpu').detach().numpy()
            save_path = os.path.join(filePath, str(epoch) + '_' + name)
            with h5py.File(save_path, 'w') as f:
                f.create_dataset("images", data=out)
        print('保存完毕')

    def getLR(self):
        return {"model_LR": self.lr_decay.get_lr()[0], "discriminator_LR": self.lr_decay_discriminator.get_lr()[0]}
