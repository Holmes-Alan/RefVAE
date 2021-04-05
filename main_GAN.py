from __future__ import print_function
import argparse
from math import log10

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from laploss import LapLoss
from torch.utils.data import DataLoader
import torch.nn.functional as F
from model import *
from network import encoder4, decoder4
from image_utils import TVLoss
from data import get_training_set
import numpy as np
from pytorch_msssim import SSIM as pytorch_ssim
import time
from lpips import lpips

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--up_factor', type=int, default=4, help='upsampling factor')
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')
parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=0.0001')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=2, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='/home/server2/ZSLiu/NTIRE2021/data/SR/')
parser.add_argument("--ref_dir", type=str, default="/home/server2/ZSLiu/style_transfer/Data/wikiart",
                    help='path to wikiArt dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--model_type', type=str, default='GAN')
parser.add_argument('--patch_size', type=int, default=64, help='Size of cropped LR image')
parser.add_argument('--pretrained_G_model', default='GAN_generator_v18.pth', help='pretrained G model')
parser.add_argument('--pretrained_D_model', default='GAN_discriminator_0.pth', help='pretrained D model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--save_folder', default='models/', help='Location to save checkpoint models')
parser.add_argument("--encoder_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained encoder path')


opt = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train(epoch):
    G_epoch_loss = 0
    D_epoch_loss = 0
    G.train()
    D.train()
    enc.eval()
    dec.eval()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target, ref = batch[0], batch[1], batch[2]

        minibatch = input.size()[0]
        real_label = torch.ones((minibatch, 1344))
        fake_label = torch.zeros((minibatch, 1344))

        input = input.to(device)
        target = target.to(device)
        ref = ref.to(device)
        real_label = real_label.to(device)
        fake_label = fake_label.to(device)

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = False

        G_optimizer.zero_grad()

        # encoder
        bic = F.interpolate(input, scale_factor=opt.up_factor, mode='bicubic')
        # ref_feat = enc(ref)
        tar_feat = enc(target)
        LR_feat = enc(bic)

        predict, KL = G(input, LR_feat['r41'], tar_feat['r41'])

        # predict = dec(LR_feat)

        pre_LR = F.interpolate(predict, scale_factor=1.0 / opt.up_factor, mode='bicubic')

        # Reconstruction loss
        LR_loss = L1_criterion(pre_LR, input)

        # pre_feat = enc(predict)
        # tar_feat = enc(target)
        # SR_loss = L1_criterion(pre_feat['r41'], tar_feat['r41'])
        SR_L1 = L1_criterion(predict, target) + \
                L1_criterion(F.interpolate(predict, scale_factor=0.5, mode='bicubic'), F.interpolate(target, scale_factor=0.5, mode='bicubic')) + \
                L1_criterion(F.interpolate(predict, scale_factor=0.25, mode='bicubic'), F.interpolate(target, scale_factor=0.25, mode='bicubic'))
        # SR_lap = lap_loss(2*predict-1, 2*target-1)

        KL_loss = KL.mean()
        # RE = log_Logistic_256(target, pre_mean, pre_logvar, average=False, dim=1)
        # RE = RE.mean()

        # ssim_loss = 1 - ssim(predict, target)
        # lap_recon = lap_loss(predict, target)
        # TV_loss = TV(predict)
        # PD_feat, mean_var_loss = PD_loss(predict, target)
        ContentLoss, StyleLoss = VGG_feat(predict, target)
        # VGG_loss = L1_criterion(pre_feat4, tar_feat4)
        lpips_sp = loss_fn_alex_sp(2 * predict - 1, 2 * target - 1)
        lpips_sp = lpips_sp.mean()

        D_fake_feat, D_fake_decision = D(predict)
        D_real_feat, D_real_decision = D(target)
        GAN_feat_loss = L1_criterion(D_fake_feat, D_real_feat)

        GAN_loss = L1_criterion(D_fake_decision, real_label)

        G_loss = 100 * LR_loss + 1 * SR_L1 + 0.01 * GAN_feat_loss + 1 * GAN_loss + 0.01 * ContentLoss + 10*StyleLoss + 1*KL_loss + 1e4*lpips_sp

        G_loss.backward()
        G_optimizer.step()

        # Reset gradient
        for p in D.parameters():
            p.requires_grad = True

        D_optimizer.zero_grad()

        _, D_fake_decision = D(predict.detach())
        _, D_real_decision = D(target)

        real = real_label * np.random.uniform(0.7, 1.2)
        fake = fake_label + np.random.uniform(0.0, 0.3)


        Dis_loss = (L1_criterion(D_real_decision, real)
                    + L1_criterion(D_fake_decision, fake)) / 2.0

        # Back propagation
        D_loss = Dis_loss
        D_loss.backward()
        D_optimizer.step()

        G_epoch_loss += G_loss.data
        D_epoch_loss += D_loss.data

        print("===> Epoch[{}]({}/{}): G_loss: {:.4f} || "
              "D_loss: {:.4f} || "
              "LR_loss: {:.4f} || "
              "SR_L1: {:.4f} || "
              "GAN_loss: {:.4f} || "
              "KL_loss: {:.4f} || "
              "ContentLoss: {:.4f} || "
              "StyleLoss: {:.4f} ||"
              "lpips_sp: {:.4f} ||"
              "GAN_feat_loss: {:.4f} ||".format(epoch, iteration,
                                          len(training_data_loader),
                                          G_loss.data,
                                          D_loss.data,
                                          LR_loss.data,
                                          SR_L1.data,
                                          GAN_loss.data,
                                          KL_loss.data,
                                          ContentLoss.data,
                                          StyleLoss.data,
                                          lpips_sp.data,
                                          GAN_feat_loss.data))

    print("===> Epoch {} Complete: Avg. G Loss: {:.4f} || D Loss: {:.4f}".format(epoch, G_epoch_loss / len(training_data_loader),
                                                                                 D_epoch_loss / len(training_data_loader)))


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def checkpoint(epoch):
    G_model_out_path = opt.save_folder + opt.model_type + "_generator_{}.pth".format(epoch)
    D_model_out_path = opt.save_folder + opt.model_type + "_discriminator_{}.pth".format(epoch)
    torch.save(G.state_dict(), G_model_out_path)
    torch.save(D.state_dict(), D_model_out_path)
    print("Checkpoint saved to {} and {}".format(G_model_out_path, D_model_out_path))



print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.patch_size, opt.up_factor,
                             opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

print('===> Building model ', opt.model_type)

# model = RCAN(num_in_ch=3, num_out_ch=3, up_factor=opt.up_factor)
enc = encoder4()
dec = decoder4()
G = VAE_v3_4x(up_factor=opt.up_factor)
D = discriminator_v2(num_channels=3, base_filter=32)
L1_criterion = nn.L1Loss(size_average=False)
L2_criterion = nn.MSELoss(size_average=False)
TV = TVLoss()
ssim = pytorch_ssim()
lap_loss = LapLoss(max_levels=5, k_size=5, sigma=2.0)
# PD_loss = PDLoss(device, l1_lambda=1.5, w_lambda=0.01)
VGG_feat = Vgg19_feat(device)
loss_fn_alex_sp = lpips.LPIPS(spatial=True)

print('---------- Generator architecture -------------')
print_network(G)
print('----------------------------------------------')
print('---------- Discriminator architecture -------------')
print_network(D)
print('----------------------------------------------')


if os.path.exists(opt.encoder_dir):
    enc.load_state_dict(torch.load(opt.encoder_dir))
    print('encoder model is loaded!')

if os.path.exists(opt.decoder_dir):
    dec.load_state_dict(torch.load(opt.decoder_dir))
    print('decoder model is loaded!')


for param in enc.parameters():
    param.requires_grad = False

for param in dec.parameters():
    param.requires_grad = False


if opt.pretrained:
    G_model_name = os.path.join(opt.save_folder + opt.pretrained_G_model)
    D_model_name = os.path.join(opt.save_folder + opt.pretrained_D_model)
    if os.path.exists(G_model_name):
        pretrained_dict = torch.load(G_model_name, map_location=lambda storage, loc: storage)
        model_dict = G.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        G.load_state_dict(model_dict)
        # G.load_state_dict(torch.load(G_model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained Generator is loaded.')
    if os.path.exists(D_model_name):
        D.load_state_dict(torch.load(D_model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained Discriminator is loaded.')

# if torch.cuda.device_count() > 1:
#     G = torch.nn.DataParallel(G)
#     D = torch.nn.DataParallel(D)


enc = enc.to(device)
dec = dec.to(device)
G = G.to(device)
D = D.to(device)
VGG_feat = VGG_feat.to(device)
L1_criterion = L1_criterion.to(device)
ssim = ssim.to(device)
lap_loss = lap_loss.to(device)
TV = TV.to(device)
loss_fn_alex_sp = loss_fn_alex_sp.to(device)


G_optimizer = optim.Adam(G.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)
D_optimizer = optim.Adam(D.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


for epoch in range(opt.start_iter, opt.nEpochs + 1):
    train(epoch)

    if epoch % (opt.nEpochs / 2) == 0:
        for param_group in G_optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(G_optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        checkpoint(epoch)
