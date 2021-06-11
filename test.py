from __future__ import print_function
import argparse

import os
import torch
import cv2
from model import *
import torchvision.transforms as transforms
from collections import OrderedDict
import numpy as np
from os.path import join
import time
from network import encoder4, decoder4
import numpy
from dataset import is_image_file
from image_utils import *
from PIL import Image, ImageOps
from os import listdir
import torch.utils.data as utils
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--testBatchSize', type=int, default=8, help='testing batch size')
parser.add_argument('--up_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=True)
parser.add_argument('--patch_size', type=int, default=64, help='0 to use original frame size')
parser.add_argument('--stride', type=int, default=4, help='0 to use original patch size')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--image_dataset', type=str, default='data/SR/Set5/')
parser.add_argument('--model_type', type=str, default='VAE')
parser.add_argument('--distortion', type=int, default=1)
parser.add_argument('--model', default='GAN_generator_50.pth', help='sr pretrained base model')
parser.add_argument("--encoder_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained encoder path')

opt = parser.parse_args()

print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('===> Building model ', opt.model_type)


# def apply_dropout(m):
#     if type(m) == nn.Dropout:
#         m.train()


model = VAE_v3_4x(up_factor=opt.up_factor)

enc = encoder4()
dec = decoder4()

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

# model_name = 'models/' + opt.model
# if os.path.exists(model_name):
#     model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
#     print(model_name)

# model = torch.nn.DataParallel(model, device_ids=gpus_list)

# mat_ncc = mat_ncc.to(device)
model = model.to(device)
enc = enc.to(device)
dec = dec.to(device)


print('===> Loading datasets')


def eval(i):
    model.eval()
    enc.eval()
    dec.eval()

    model_name = 'models/GAN_generator_'+str(i)+'.pth'
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print(model_name)

    HR_filename = os.path.join(opt.image_dataset, 'HR')
    Ref_filename = os.path.join(opt.image_dataset, 'Ref')
    # LR_filename = os.path.join(opt.image_dataset, 'hazy')
    SR_filename = os.path.join(opt.image_dataset, 'SR')
    SLR_filename = os.path.join(opt.image_dataset, 'SLR')

    gt_image = [join(HR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    ref_image = [join(Ref_filename, x) for x in listdir(Ref_filename) if is_image_file(x)]
    ref_image = sorted(ref_image)
    output_image = [join(SR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]
    slr_output_image = [join(SLR_filename, x) for x in listdir(HR_filename) if is_image_file(x)]

    count = 0
    avg_psnr_predicted = 0.0
    avg_ssim_predicted = 0.0
    avg_psnr_LR = 0.0
    avg_ssim_LR = 0.0
    t0 = time.time()
    # ran_patch = torch.randint(896, (2,))
    for i in range(gt_image.__len__()):

        HR = Image.open(gt_image[i]).convert('RGB')
        HR = modcrop(HR, opt.up_factor)
        Ref = Image.open(ref_image[2]).convert('RGB')
        Ref = modcrop(Ref, opt.up_factor)
        LR = rescale_img(HR, 1.0/opt.up_factor)
        with torch.no_grad():
            pre_LR, prediction = chop_forward(Ref, LR)

        # print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))
        prediction = prediction.data[0].cpu().permute(1, 2, 0)
        pre_LR = pre_LR.data[0].cpu().permute(1, 2, 0)

        prediction = prediction * 255.0
        pre_LR = pre_LR * 255.0
        prediction = prediction.clamp(0, 255)
        pre_LR = pre_LR.clamp(0, 255)

        Image.fromarray(np.uint8(prediction)).save(output_image[i])
        Image.fromarray(np.uint8(pre_LR)).save(slr_output_image[i])

        GT = np.array(HR).astype(np.float32)
        GT_Y = rgb2ycbcr(GT)
        LR = np.array(LR).astype(np.float32)
        LR_Y = rgb2ycbcr(LR)
        prediction = np.array(prediction).astype(np.float32)
        pre_LR = np.array(pre_LR).astype(np.float32)
        prediction_Y = rgb2ycbcr(prediction)
        pre_LR_Y = rgb2ycbcr(pre_LR)
        psnr_predicted = PSNR(prediction_Y, GT_Y, shave_border=opt.up_factor)
        ssim_predicted = SSIM(prediction_Y, GT_Y, shave_border=opt.up_factor)
        avg_psnr_predicted += psnr_predicted
        avg_ssim_predicted += ssim_predicted
        psnr_predicted = PSNR(pre_LR_Y, LR_Y, shave_border=1)
        ssim_predicted = SSIM(pre_LR_Y, LR_Y, shave_border=1)
        avg_psnr_LR += psnr_predicted
        avg_ssim_LR += ssim_predicted
        count += 1

    t1 = time.time()

    avg_psnr_predicted = avg_psnr_predicted / count
    avg_ssim_predicted = avg_ssim_predicted / count
    avg_psnr_LR = avg_psnr_LR / count
    avg_ssim_LR = avg_ssim_LR / count
    avg_time_predicted = t1 - t0
    print("PSNR_predicted= {:.4f} || "
          "SSIM_predicted= {:.4f} || "
          "PSNR_LR= {:.4f} || "
          "SSIM_LR= {:.4f} || Time= {:.4f} ".format(
                                                avg_psnr_predicted,
                                                avg_ssim_predicted,
                                                avg_psnr_LR,
                                                avg_ssim_LR,
                                                avg_time_predicted))



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(ref, img):

    img = transform(img).unsqueeze(0)
    ref = transform(ref).unsqueeze(0)

    testset = utils.TensorDataset(ref, img)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)
    std_z = torch.from_numpy(np.random.normal(0, 1, (1, 256))).float()
    z_q = std_z.to(device)

    for iteration, batch in enumerate(test_dataloader, 1):
        ref, input = batch[0].to(device), batch[1].to(device)
        batch_size, channels, img_height, img_width = input.size()
        # eps = torch.randn(input.shape[0], 64, 1, 1).to(device)

        # eps = torch.from_numpy(np.random.normal(0, 1, (input.shape[0], 3, 256, 256))).float()
        # eps = eps.to(device)
        #
        # LR_patches = patchify_tensor(input, patch_size=opt.patch_size, overlap=opt.stride)
        # n_patches = LR_patches.size(0)
        # out_box = []
        # with torch.no_grad():
        #     for p in range(n_patches):
        #         LR_input = LR_patches[p:p + 1]
        #         LR_feat = enc(F.interpolate(LR_input, scale_factor=opt.up_factor, mode='bicubic'))
        #         ref_feat = enc(ref)
        #         SR, _ = model(LR_input, LR_feat['r41'], ref_feat['r41'])
        #         out_box.append(SR)
        #
        #     out_box = torch.cat(out_box, 0)
        #     SR = recompose_tensor(out_box, opt.up_factor * img_height, opt.up_factor * img_width,
        #                                       overlap=opt.up_factor * opt.stride)

        LR_feat = enc(F.interpolate(input, scale_factor=opt.up_factor, mode='bicubic'))
        ref_feat = enc(ref)
        SR, _ = model(input, LR_feat['r41'], ref_feat['r41'])

        LR = F.interpolate(SR, scale_factor=1/opt.up_factor, mode='bicubic')

    return LR, SR




##Eval Start!!!!
for i in range(5, 475, 5):
    eval(i)
