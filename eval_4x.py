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
from dataset import is_image_file, rescale_img
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
parser.add_argument('--use_img_self', action='store_true', help='using LR image itself or not')
parser.add_argument('--use_ref', action='store_true', help='using external reference images or not')
parser.add_argument('--num_sample', type=int, default=10, help='number of SR images')
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument('--input_dataset', type=str, default='input')
parser.add_argument('--output_dataset', type=str, default='result')
parser.add_argument('--model_type', type=str, default='ConVAE')
parser.add_argument('--model', default='models/ConVAE_4x.pth', help='sr pretrained base model')
parser.add_argument("--encoder_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained encoder path')

opt = parser.parse_args()

print(opt)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print('===> Building model ', opt.model_type)


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



model = model.to(device)
enc = enc.to(device)
dec = dec.to(device)


print('===> Loading datasets')


def eval():
    model.eval()
    enc.eval()
    dec.eval()

    if os.path.exists(opt.model):
        model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
        print(opt.model)

    Ref_filename = os.path.join(opt.input_dataset, 'Ref')
    LR_filename = os.path.join(opt.input_dataset, 'LR')
    SR_filename = opt.output_dataset

    lr_image = [join(LR_filename, x) for x in listdir(LR_filename) if is_image_file(x)]
    lr_image = sorted(lr_image)
    ref_image = [join(Ref_filename, x) for x in listdir(Ref_filename) if is_image_file(x)]
    ref_image = sorted(ref_image)

    for i in range(len(lr_image)):

        LR = Image.open(lr_image[i]).convert('RGB')
        LR = modcrop(LR, opt.up_factor)

        if len(ref_image) != 0 and opt.use_ref:
            print("using ref images for SR")
            for j in range(len(ref_image)):
                Ref = Image.open(ref_image[j]).convert('RGB')

                with torch.no_grad():
                    prediction = chop_forward(Ref, LR)

                prediction = prediction.data[0].cpu().permute(1, 2, 0)

                prediction = prediction * 255.0
                prediction = prediction.clamp(0, 255)
                lr_name = lr_image[i][-8:-4]
                output_name = SR_filename + '/' + lr_name.zfill(6) + '_sample' + str(j).zfill(5) + '.png'
                Image.fromarray(np.uint8(prediction)).save(output_name)

        else:
            if opt.use_img_self:
                print("using LR images itself for SR")
                Ref = LR.resize((256, 256))

                with torch.no_grad():
                    prediction = chop_forward(Ref, LR)

                prediction = prediction.data[0].cpu().permute(1, 2, 0)

                prediction = prediction * 255.0
                prediction = prediction.clamp(0, 255)
                lr_name = lr_image[i][-8:-4]
                output_name = SR_filename + '/' + lr_name.zfill(6) + '_sample0.png'
                Image.fromarray(np.uint8(prediction)).save(output_name)

            else:
                print("using random noise for SR")
                for j in range(opt.num_sample):
                    a = np.random.rand(256, 256, 3)
                    Ref = Image.fromarray(np.uint8(a * 128))

                    with torch.no_grad():
                        prediction = chop_forward(Ref, LR)

                    prediction = prediction.data[0].cpu().permute(1, 2, 0)

                    prediction = prediction * 255.0
                    prediction = prediction.clamp(0, 255)
                    lr_name = lr_image[i][-8:-4]
                    output_name = SR_filename + '/' + lr_name.zfill(6) + '_sample' + str(j).zfill(5) + '.png'
                    print("random SR: {}".format(j))
                    Image.fromarray(np.uint8(prediction)).save(output_name)

            # pre_LR = F.interpolate(prediction, scale_factor=1 / opt.up_factor, mode='bicubic')
            #
            # prediction = prediction.data[0].cpu().permute(1, 2, 0)
            # pre_LR = pre_LR.data[0].cpu().permute(1, 2, 0)
            #
            # prediction = prediction * 255.0
            # prediction = prediction.clamp(0, 255)
            # lr_name = lr_image[i][-8:-4]
            # output_name = SR_filename + '/' + lr_name.zfill(6) + '_sample0.png'
            # Image.fromarray(np.uint8(prediction)).save(output_name)
            #
            # pre_LR = pre_LR * 255.0
            # pre_LR = pre_LR.clamp(0, 255)
            #
            # LR = np.array(LR).astype(np.float32)
            # LR_Y = rgb2ycbcr(LR)
            # pre_LR = np.array(pre_LR).astype(np.float32)
            # pre_LR_Y = rgb2ycbcr(pre_LR)
            #
            # psnr_predicted = PSNR(pre_LR_Y, LR_Y, shave_border=1)
            # ssim_predicted = SSIM(pre_LR_Y, LR_Y, shave_border=1)



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

style_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(ref, img):

    img = transform(img).unsqueeze(0)
    ref = style_transform(ref).unsqueeze(0)

    testset = utils.TensorDataset(ref, img)
    test_dataloader = utils.DataLoader(testset, num_workers=opt.threads,
                                       drop_last=False, batch_size=opt.testBatchSize, shuffle=False)

    for iteration, batch in enumerate(test_dataloader, 1):
        ref, input = batch[0].to(device), batch[1].to(device)

        LR_feat = enc(F.interpolate(input, scale_factor=opt.up_factor, mode='bicubic'))
        ref_feat = enc(ref)
        SR, _ = model(input, LR_feat['r41'], ref_feat['r41'])

    return SR




##Eval Start!!!!
eval()

