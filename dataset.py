import torch.utils.data as data
import torch
import numpy as np
import os
from os import listdir
from os.path import join
from PIL import Image, ImageOps, ImageEnhance
import random
from torchvision import transforms
from glob import glob
from imresize import imresize


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".png", ".jpg", ".jpeg"])


def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    # y, _, _ = img.split()
    return img


# def rescale_img(img_in, scale):
#     size_in = img_in.size
#     new_size_in = tuple([int(x * scale) for x in size_in])
#     img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
#     return img_in
def rescale_img(img_in, size):
    # size_in = img_in.size
    new_size_in = tuple([size, size])
    img_in = img_in.resize(new_size_in, resample=Image.BICUBIC)
    return img_in


def get_patch(img_in, img_tar, patch_size, scale, ix=-1, iy=-1):
    (ih, iw) = img_in.size
    #(th, tw) = (scale * ih, scale * iw)

    patch_mult = scale  # if len(scale) > 1 else 1
    tp = patch_mult * patch_size
    ip = tp // scale

    if ix == -1:
        ix = random.randrange(0, iw - ip + 1)
        # ix = torch.randint(0, iw - ip + 1, (1,)).item()
    if iy == -1:
        iy = random.randrange(0, ih - ip + 1)
        # iy = torch.randint(0, ih - ip + 1, (1,)).item()


    (tx, ty) = (scale * ix, scale * iy)

    img_in = img_in.crop((iy, ix, iy + ip, ix + ip))
    img_tar = img_tar.crop((ty, tx, ty + tp, tx + tp))
    # img_ref = img_ref.crop((ty, tx, ty + tp, tx + tp))

    #info_patch = {
    #    'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}

    return img_in, img_tar




def augment(img_in, img_tar, img_ref, flip_h=True, flip_v=True, rot=True):


    if torch.rand(1).item() < 0.5 and flip_h:
        img_in = ImageOps.flip(img_in)
        img_tar = ImageOps.flip(img_tar)
        img_ref = ImageOps.flip(img_ref)

    if torch.rand(1).item() < 0.5 and flip_v:
        img_in = ImageOps.mirror(img_in)
        img_tar = ImageOps.mirror(img_tar)
        img_ref = ImageOps.mirror(img_ref)

    if torch.rand(1).item() < 0.5 and rot:
        rot = torch.randint(1, 3, (1,)).item() * 90
        img_in = img_in.rotate(rot)
        img_tar = img_tar.rotate(rot)
        img_ref = img_ref.rotate(rot)


    return img_in, img_tar, img_ref

def rgb_permute(im1, im2):

    im1 = np.array(im1)
    im2 = np.array(im2)
    # if np.random.rand(1) >= prob:
    #     return im1, im2

    perm = np.random.permutation(3)
    im1 = im1[:, :, perm]
    im2 = im2[:, :, perm]
    im1 = Image.fromarray(im1)
    im2 = Image.fromarray(im2)

    return im1, im2

def color_shift(img_in, img_tar):

    color_factor = random.uniform(1, 1.5)
    contrast_factor = random.uniform(1, 1.5)
    bright_factor = random.uniform(1, 1.5)
    # sharp_factor = random.uniform(0.5, 1)

    if torch.rand(1).item() < 0.5:
        img_tar = ImageEnhance.Color(img_tar).enhance(color_factor)
        img_in = ImageEnhance.Color(img_in).enhance(color_factor)

    if torch.rand(1).item() < 0.5:
        img_tar = ImageEnhance.Contrast(img_tar).enhance(contrast_factor)
        img_in = ImageEnhance.Contrast(img_in).enhance(contrast_factor)

    if torch.rand(1).item() < 0.5:
        img_tar = ImageEnhance.Brightness(img_tar).enhance(bright_factor)
        img_in = ImageEnhance.Brightness(img_in).enhance(bright_factor)
        # img_in = ImageEnhance.Sharpness(img_in).enhance(sharp_factor)

    return img_in, img_tar




class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir1, data_dir2, patch_size, up_factor, data_augmentation, transform=None):
        super(DatasetFromFolder, self).__init__()
        GT_dir = join(data_dir1, 'HR')
        input_dir = join(data_dir1, 'LR')
        self.gt_image_filenames = [join(GT_dir, x) for x in listdir(GT_dir) if is_image_file(x)]
        self.input_image_filenames = [join(input_dir, x) for x in listdir(input_dir) if is_image_file(x)]
        # GT_dir = join(data_dir2, 'HR')
        # input_dir = join(data_dir2, 'LR')
        # self.gt_image_filenames += [join(GT_dir, x) for x in listdir(GT_dir) if is_image_file(x)]
        # self.input_image_filenames += [join(input_dir, x) for x in listdir(input_dir) if is_image_file(x)]
        # ref_dir = '/home/server2/ZSLiu/style_transfer/Data/wikiart'
        # self.ref_image_filenames = [join(ref_dir, x) for x in listdir(ref_dir) if is_image_file(x)]

        self.gt_image_filenames = sorted(self.gt_image_filenames)
        self.input_image_filenames = sorted(self.input_image_filenames)
        self.patch_size = patch_size
        self.up_factor = up_factor
        self.transform = transform
        self.data_augmentation = data_augmentation


    def __getitem__(self, index):

        target = load_img(self.gt_image_filenames[index])

        input = load_img(self.input_image_filenames[index])


        rand_no = torch.randint(0, len(self.gt_image_filenames), (1,)).item()
        ref = load_img(self.gt_image_filenames[rand_no])
        ref = rescale_img(ref, 256)

        target = rescale_img(target, 288)
        input = rescale_img(input, 288//self.up_factor)

        input, target = get_patch(input, target, self.patch_size, scale=self.up_factor)


        if self.data_augmentation:
            input, target, ref = augment(input, target, ref)
            # input, target = color_shift(input, target)


        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
            ref = self.transform(ref)


        return input, target, ref

    def __len__(self):
        return len(self.gt_image_filenames)


class DatasetFromFolder_new(data.Dataset):
    def __init__(self, data_dir1, data_dir2, patch_size, up_factor, data_augmentation, transform=None):
        super(DatasetFromFolder_new, self).__init__()
        GT_dir = join(data_dir1, 'HR')
        input_dir = join(data_dir1, 'LR')
        self.gt_image_filenames = [join(GT_dir, x) for x in listdir(GT_dir) if is_image_file(x)]
        self.input_image_filenames = [join(input_dir, x) for x in listdir(input_dir) if is_image_file(x)]
        # GT_dir = join(data_dir2, 'HR')
        # input_dir = join(data_dir2, 'LR')
        # self.gt_image_filenames += [join(GT_dir, x) for x in listdir(GT_dir) if is_image_file(x)]
        # self.input_image_filenames += [join(input_dir, x) for x in listdir(input_dir) if is_image_file(x)]
        # ref_dir = '/home/server2/ZSLiu/style_transfer/Data/wikiart'
        # self.ref_image_filenames = [join(ref_dir, x) for x in listdir(ref_dir) if is_image_file(x)]

        self.gt_image_filenames = sorted(self.gt_image_filenames)
        self.input_image_filenames = sorted(self.input_image_filenames)
        self.patch_size = patch_size
        self.up_factor = up_factor
        self.transform = transform
        self.data_augmentation = data_augmentation


    def __getitem__(self, index):

        target = load_img(self.gt_image_filenames[index])

        input = imresize(np.array(target), 0.125)
        input = Image.fromarray(np.unit8(input))


        rand_no = torch.randint(0, len(self.gt_image_filenames), (1,)).item()
        ref = load_img(self.gt_image_filenames[rand_no])
        ref = rescale_img(ref, 256)

        target = rescale_img(target, 288)
        input = rescale_img(input, 36)
        # ref = rescale_img(target, 256)
        input, target = get_patch(input, target, self.patch_size, scale=self.up_factor)


        if self.data_augmentation:
            input, target, ref = augment(input, target, ref)
            # input, target = color_shift(input, target)


        if self.transform:
            input = self.transform(input)
            target = self.transform(target)
            ref = self.transform(ref)


        return input, target, ref

    def __len__(self):
        return len(self.gt_image_filenames)
class DatasetFromFolderEval(data.Dataset):
    def __init__(self, data_dir, transform=None):
        super(DatasetFromFolderEval, self).__init__()
        data_dir = data_dir + 'hazy'
        self.image_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        self.transform = transform

    def __getitem__(self, index):
        input = load_img(self.image_filenames[index])


        if self.transform:
            input = self.transform(input)

        return input

    def __len__(self):
        return len(self.image_filenames)
