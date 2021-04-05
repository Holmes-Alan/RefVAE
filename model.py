import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from torchvision import models



class ncc_test(nn.Module):
    """Residual Channel Attention Networks.
    Paper: Image Super-Resolution Using Very Deep Residual Channel Attention
        Networks
    Ref git repo: https://github.com/yulunzhang/RCAN.
    Args:
        num_in_ch (int): Channel number of inputs.
        num_out_ch (int): Channel number of outputs.
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        num_group (int): Number of ResidualGroup. Default: 10.
        num_block (int): Number of RCAB in ResidualGroup. Default: 16.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        upscale (int): Upsampling factor. Support 2^n and 3.
            Default: 4.
        res_scale (float): Used to scale the residual in residual block.
            Default: 1.
        img_range (float): Image range. Default: 255.
        rgb_mean (tuple[float]): Image mean in RGB orders.
            Default: (0.4488, 0.4371, 0.4040), calculated from DIV2K dataset.
    """

    def __init__(self,
                 k,
                 patch_size,):
        super(ncc_test, self).__init__()


        self.k = k
        self.patch_size = patch_size

        self.unfold = nn.Unfold(kernel_size=(3, 3), stride=2)
        self.fold = nn.Fold(output_size=(self.patch_size, self.patch_size), kernel_size=(3, 3), stride=2)
        self.max = nn.MaxPool2d(3, stride=1, padding=1)

    def forward(self, x, ref):
        # x_gray = torch.min(x, dim=1, keepdim=True)[0]
        x_gray = torch.max(self.max(x), dim=1, keepdim=True)[0] - torch.min(x, dim=1, keepdim=True)[0]
        ref_gray = torch.mean(ref, dim=1, keepdim=True)
        input_patch = self.unfold(x_gray)
        ref_patch = self.unfold(ref_gray)
        input_mu = torch.mean(input_patch, dim=2, keepdim=True)
        ref_mu = torch.mean(ref_patch, dim=2, keepdim=True)

        input_norm = input_patch - input_mu
        input_len = input_patch.norm(dim=1, keepdim=True)
        input_norm = input_norm / input_len
        input_norm_t = input_norm.permute(0, 2, 1)
        ref_norm = ref_patch - ref_mu
        ref_len = ref_patch.norm(dim=1, keepdim=True)
        ref_norm = ref_norm / ref_len
        ncc = torch.bmm(input_norm_t, ref_norm)
        # idx = torch.argmax(ncc, dim=2)
        idx = torch.topk(ncc, k=self.k, dim=2)[1]
        x_rec = torch.zeros(input_patch.shape[0], input_patch.shape[1], input_patch.shape[2]).type_as(input_patch)
        for i in range(self.k):
            for j in range(input_patch.shape[0]):
                t = idx[j:j+1, :, i:i+1].squeeze(2)
                x_rec[j:j+1, :, :] = ref_patch[j:j+1, :, t].squeeze(2)

            x = torch.cat((x, self.fold(x_rec)), dim=1)

        return x


def norm_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    normalized_feat = (feat - feat_mean.expand(
        size)) / feat_std.expand(size)

    return normalized_feat



class VAE_v1(nn.Module):
    def __init__(self, up_factor):
        super(VAE_v1, self).__init__()

        self.up_factor = up_factor
        self.init_feat = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(128, 256, 3, 1, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(256, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
            nn.Conv2d(512, 512, 3, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(),
            nn.MaxPool2d(2, 2, 0),
        )

        self.VAE_encoder = nn.Sequential(
            nn.Linear(8192, 512),
            # nn.ReLU(),
            # nn.Linear(2048, 512),
        )
        self.VAE_decoder = nn.Sequential(
            nn.Linear(256, 16384),
        )

        self.up1 = nn.Sequential(
            nn.Conv2d(16, 256, 3, 1, 1),
            nn.LeakyReLU(),
            ResnetBlock(256, 3, 1, 1),
            nn.InstanceNorm2d(256)
        )
        self.mask1 = nn.Sequential(
            nn.Conv2d(3, 256, 3, 1, 1),
            # nn.InstanceNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 512, 1, 1, 0)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
            ResnetBlock(128, 3, 1, 1),
            nn.InstanceNorm2d(128)
        )
        self.mask2 = nn.Sequential(
            nn.Conv2d(3, 128, 3, 1, 1),
            # nn.InstanceNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 256, 1, 1, 0)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            ResnetBlock(64, 3, 1, 1),
            nn.InstanceNorm2d(64)
        )
        self.mask3 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 1, 1, 0)
        )
        self.up4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            ResnetBlock(64, 3, 1, 1),
            nn.InstanceNorm2d(64)
        )
        self.mask4 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            # nn.InstanceNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 128, 1, 1, 0)
        )

        self.recon = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Tanh()
        )
        self.pix_up = nn.Upsample(scale_factor=2, mode='bilinear')

        self.mu_act = nn.Sigmoid()
        self.var_act = nn.Hardtanh(min_val=-4.5, max_val=0)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std) + mu

    def encode(self, HR):
        # initial
        HR_feat = self.init_feat(HR)
        # encoder
        z_q_mu, z_q_logvar = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1)).chunk(2, dim=1)

        return z_q_mu, z_q_logvar

    def decode(self, LR, z_q):
        dec_feat = self.VAE_decoder(z_q).view(LR.size(0), 16, 32, 32)

        # reconstruction
        up1 = self.up1(dec_feat)
        # up1 = norm_mean_std(up1)
        up1 = F.interpolate(up1, size=[LR.shape[2], LR.shape[3]], mode='bicubic')
        mu1, var1 = self.mask1(LR).chunk(2, dim=1)
        up1 = up1 * (1 + var1) + mu1
        up1 = self.pix_up(up1)

        up2 = self.up2(up1)
        # up2 = norm_mean_std(up2)
        mu2, var2 = self.mask2(F.interpolate(LR, scale_factor=2, mode='bicubic')).chunk(2, dim=1)
        up2 = up2 * (1 + var2) + mu2
        up2 = self.pix_up(up2)

        up3 = self.up3(up2)
        # up3 = norm_mean_std(up3)
        mu3, var3 = self.mask3(F.interpolate(LR, scale_factor=4, mode='bicubic')).chunk(2, dim=1)
        up3 = up3 * (1 + var3) + mu3
        up3 = self.pix_up(up3)

        up4 = self.up4(up3)
        # up4 = norm_mean_std(up4)
        mu4, var4 = self.mask4(F.interpolate(LR, scale_factor=8, mode='bicubic')).chunk(2, dim=1)
        up4 = up4 * (1 + var4) + mu4

        SR = self.recon(up4)


        return SR


    def forward(self, LR, HR=None, z_q=None):
        # encode
        if z_q is None:
            bic = F.interpolate(LR, scale_factor=self.up_factor, mode='bicubic')
            z_q_mu, z_q_logvar= self.encode(HR - bic)
            z_q = self.reparameterize(z_q_mu, z_q_logvar)
            KL = -0.5 * torch.sum(1 + z_q_logvar - z_q_mu.pow(2) - z_q_logvar.exp())
            # decode
            SR = self.decode(LR, z_q)

            return SR, KL
        else:
            SR= self.decode(LR, z_q)

            return SR



class VAE_v2(nn.Module):
    def __init__(self, up_factor):
        super(VAE_v2, self).__init__()

        self.up_factor = up_factor
        self.init_feat = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.VAE_encoder = nn.Sequential(
            nn.Linear(16384, 512),
            # nn.ReLU(),
            # nn.Linear(2048, 512),
        )
        self.VAE_decoder = nn.Sequential(
            nn.Linear(256, 16384),
        )

        self.dec_feat = nn.Sequential(
            nn.Conv2d(64, 512, 1, 1, 0),
            # nn.LeakyReLU(),
            ResnetBlock(512, 3, 1, 1),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            # nn.Dropout2d(p=0.5),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            nn.Conv2d(512, 1024, 1, 1, 0),
        )


        self.mu_act = nn.Sigmoid()
        self.var_act = nn.Hardtanh(min_val=-4.5, max_val=0)
        self.bn = nn.BatchNorm1d(256)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std) + mu

    def encode(self, HR_feat):
        # initial
        HR_feat = self.init_feat(HR_feat)
        # encoder
        HR_feat = F.interpolate(HR_feat, size=[16, 16], mode='bilinear')
        z_q_mu, z_q_logvar = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1)).chunk(2, dim=1)
        # z_q_mu = self.bn(z_q_mu)

        return z_q_mu, z_q_logvar

    def decode(self, LR_feat, z_q):
        dec_feat = self.VAE_decoder(z_q).view(LR_feat.size(0), 64, 16, 16)

        # reconstruction
        dec_feat = F.interpolate(dec_feat, size=[LR_feat.shape[2], LR_feat.shape[3]], mode='bilinear')
        feat = self.dec_feat(dec_feat)
        mu, var = self.mask(LR_feat).chunk(2, dim=1)
        feat = feat * (1 + var) + mu

        return feat


    def forward(self, LR_feat, HR_feat=None, z_q=None):
        # encode
        if z_q is None:
            z_q_mu, z_q_logvar= self.encode(HR_feat)
            z_q = self.reparameterize(z_q_mu, z_q_logvar)
            KL = -0.5 * torch.sum(1 + z_q_logvar - z_q_mu.pow(2) - z_q_logvar.exp())
            # decode
            SR = self.decode(LR_feat, z_q)

            return SR, KL
        else:
            SR= self.decode(LR_feat, z_q)

            return SR


class VAE_v3_8x(nn.Module):
    def __init__(self, up_factor):
        super(VAE_v3_8x, self).__init__()

        self.up_factor = up_factor
        self.init_feat = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.VAE_encoder = nn.Sequential(
            nn.Linear(16384, 512),
            # nn.ReLU(),
            # nn.Linear(2048, 512),
        )
        self.VAE_decoder = nn.Sequential(
            nn.Linear(256, 16384),
        )

        self.dec_feat = nn.Sequential(
            nn.Conv2d(64, 512, 1, 1, 0),
            # nn.LeakyReLU(),
            ResnetBlock(512, 3, 1, 1),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            # nn.Dropout2d(p=0.5),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            nn.Conv2d(512, 1024, 1, 1, 0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )


        self.mu_act = nn.Sigmoid()
        self.var_act = nn.Hardtanh(min_val=-4.5, max_val=0)
        self.bn = nn.BatchNorm1d(256)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std) + mu

    def encode(self, HR_feat):
        # initial
        HR_feat = self.init_feat(HR_feat)
        # encoder
        HR_feat = F.interpolate(HR_feat, size=[16, 16], mode='bilinear')
        z_q_mu, z_q_logvar = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1)).chunk(2, dim=1)
        # z_q_mu = self.bn(z_q_mu)

        return z_q_mu, z_q_logvar

    def decode(self, LR, LR_feat, z_q):
        dec_feat = self.VAE_decoder(z_q).view(LR_feat.size(0), 64, 16, 16)

        # reconstruction
        dec_feat = F.interpolate(dec_feat, size=[LR_feat.shape[2], LR_feat.shape[3]], mode='bilinear')
        feat = self.dec_feat(dec_feat)
        mu, var = self.mask(LR_feat).chunk(2, dim=1)
        feat = feat * (1 + var) + mu

        SR = self.decoder(feat) + F.interpolate(LR, scale_factor=8, mode='bicubic')

        return SR


    def forward(self, LR, LR_feat, HR_feat=None, z_q=None):
        # encode
        if z_q is None:
            z_q_mu, z_q_logvar= self.encode(HR_feat)
            z_q = self.reparameterize(z_q_mu, z_q_logvar)
            KL = -0.5 * torch.sum(1 + z_q_logvar - z_q_mu.pow(2) - z_q_logvar.exp())
            # decode
            SR = self.decode(LR, LR_feat, z_q)

            return SR, KL
        else:
            SR= self.decode(LR, LR_feat, z_q)

            return SR



class VAE_v3_4x(nn.Module):
    def __init__(self, up_factor):
        super(VAE_v3_4x, self).__init__()

        self.up_factor = up_factor
        self.init_feat = nn.Sequential(
            nn.Conv2d(512, 64, 1, 1, 0),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
        )

        self.VAE_encoder = nn.Sequential(
            nn.Linear(16384, 512),
            # nn.ReLU(),
            # nn.Linear(2048, 512),
        )
        self.VAE_decoder = nn.Sequential(
            nn.Linear(256, 16384),
        )

        self.dec_feat = nn.Sequential(
            nn.Conv2d(64, 512, 1, 1, 0),
            # nn.LeakyReLU(),
            ResnetBlock(512, 3, 1, 1),
        )
        self.mask = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, 0),
            # nn.Dropout2d(p=0.5),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            ResnetBlock(512, 3, 1, 1),
            nn.Conv2d(512, 1024, 1, 1, 0),
        )

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, 1, 1, 0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 3, 3, 1, 1),
        )


        self.mu_act = nn.Sigmoid()
        self.var_act = nn.Hardtanh(min_val=-4.5, max_val=0)
        self.bn = nn.BatchNorm1d(256)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps.mul(std) + mu

    def encode(self, HR_feat):
        # initial
        HR_feat = self.init_feat(HR_feat)
        # encoder
        HR_feat = F.interpolate(HR_feat, size=[16, 16], mode='bilinear')
        z_q_mu, z_q_logvar = self.VAE_encoder(HR_feat.view(HR_feat.size(0), -1)).chunk(2, dim=1)
        # z_q_mu = self.bn(z_q_mu)

        return z_q_mu, z_q_logvar

    def decode(self, LR, LR_feat, z_q):
        dec_feat = self.VAE_decoder(z_q).view(LR_feat.size(0), 64, 16, 16)

        # reconstruction
        dec_feat = F.interpolate(dec_feat, size=[LR_feat.shape[2], LR_feat.shape[3]], mode='bilinear')
        feat = self.dec_feat(dec_feat)
        mu, var = self.mask(LR_feat).chunk(2, dim=1)
        feat = feat * (1 + var) + mu

        SR = self.decoder(feat) + F.interpolate(LR, scale_factor=4, mode='bicubic')

        return SR


    def forward(self, LR, LR_feat, HR_feat=None, z_q=None):
        # encode
        if z_q is None:
            z_q_mu, z_q_logvar= self.encode(HR_feat)
            z_q = self.reparameterize(z_q_mu, z_q_logvar)
            KL = -0.5 * torch.sum(1 + z_q_logvar - z_q_mu.pow(2) - z_q_logvar.exp())
            # decode
            SR = self.decode(LR, LR_feat, z_q)

            return SR, KL
        else:
            SR= self.decode(LR, LR_feat, z_q)

            return SR




class generator(nn.Module):
    def __init__(self, input_num, base_filter):
        super(generator, self).__init__()

        # backbone
        self.input_conv = nn.Linear(256, 16384)
        self.conv1 = nn.Conv2d(16, base_filter * 4, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(base_filter * 4)
        self.res1 = ResnetBlock(base_filter * 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(base_filter * 4, base_filter * 2, 3, 1, 1)
        self.norm2 = nn.InstanceNorm2d(base_filter * 2)
        self.res2 = ResnetBlock(base_filter * 2, 3, 1, 1)
        self.conv3 = nn.Conv2d(base_filter * 2, base_filter, 3, 1, 1)
        self.norm3 = nn.InstanceNorm2d(base_filter)
        self.res3 = ResnetBlock(base_filter, 3, 1, 1)
        self.act = nn.LeakyReLU()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear')

        # condition
        self.feat = nn.Conv2d(input_num, base_filter * 4, 3, 1, 1)



        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        feat = self.act(self.input_conv(x))
        out1 = self.act(self.norm1(self.conv1(feat)))
        out2 = self.act(self.norm2(self.conv2(out1)))
        out3 = self.act(self.norm3(self.conv3(out2)))

        prob = self.weight(out3)

        b = feat.shape[0]

        prob = prob.view(b, -1)

        feat = feat.view(b, -1)
        out1 = out1.view(b, -1)
        out2 = out2.view(b, -1)
        out3 = out3.view(b, -1)

        out = torch.cat((feat, out1, out2, out3), 1)

        return out, prob

    def forward(self, x):
        # x = torch.cat((x, y), 1)
        feat1, prob1 = self.encode(x)
        x = self.down(x)
        feat2, prob2 = self.encode(x)
        x = self.down(x)
        feat3, prob3 = self.encode(x)

        feat_out = torch.cat((feat1, feat2, feat3), 1)
        prob_out = torch.cat((prob1, prob2, prob3), 1)

        return feat_out, prob_out

class discriminator_v2(nn.Module):
    def __init__(self, num_channels, base_filter):
        super(discriminator_v2, self).__init__()

        self.input_conv = nn.Conv2d(num_channels, base_filter, 3, 1, 1)#512*256
        self.conv1 = nn.Conv2d(base_filter, base_filter * 2, 4, 2, 1)
        self.norm1 = nn.InstanceNorm2d(base_filter * 2)
        self.conv2 = nn.Conv2d(base_filter * 2, base_filter * 4, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(base_filter * 4)
        self.conv3 = nn.Conv2d(base_filter * 4, base_filter * 8, 4, 2, 1)
        self.norm3 = nn.InstanceNorm2d(base_filter * 8)
        self.act = nn.LeakyReLU(0.2, False)

        self.weight = nn.Conv2d(base_filter * 8, 1, 3, 1, 1)

        self.down = nn.MaxPool2d(3, stride=2, padding=[1, 1])


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

    def encode(self, x):
        feat = self.act(self.input_conv(x))
        out1 = self.act(self.norm1(self.conv1(feat)))
        out2 = self.act(self.norm2(self.conv2(out1)))
        out3 = self.act(self.norm3(self.conv3(out2)))

        prob = self.weight(out3)

        b = feat.shape[0]

        prob = prob.view(b, -1)

        feat = feat.view(b, -1)
        out1 = out1.view(b, -1)
        out2 = out2.view(b, -1)
        out3 = out3.view(b, -1)

        out = torch.cat((feat, out1, out2, out3), 1)

        return out, prob

    def forward(self, x):
        # x = torch.cat((x, y), 1)
        feat1, prob1 = self.encode(x)
        x = self.down(x)
        feat2, prob2 = self.encode(x)
        x = self.down(x)
        feat3, prob3 = self.encode(x)

        feat_out = torch.cat((feat1, feat2, feat3), 1)
        prob_out = torch.cat((prob1, prob2, prob3), 1)

        return feat_out, prob_out


class discriminator_v3(nn.Module):
    def __init__(self, num_channels, base_filter, down_factor):
        super(discriminator_v3, self).__init__()
        self.down_factor = down_factor

        self.input_conv = nn.Conv2d(num_channels*down_factor*down_factor, base_filter, 1, 1, 0)#512*256
        self.bn = nn.InstanceNorm2d(base_filter)
        self.conv1 = nn.Conv2d(base_filter, base_filter, 3, 1, 1)
        self.norm1 = nn.InstanceNorm2d(base_filter)
        self.conv2 = nn.Conv2d(base_filter, base_filter, 4, 2, 1)
        self.norm2 = nn.InstanceNorm2d(base_filter)
        self.conv3 = nn.Conv2d(base_filter, base_filter, 4, 2, 1)
        self.norm3 = nn.InstanceNorm2d(base_filter)
        self.conv4 = nn.Conv2d(base_filter, base_filter, 4, 2, 1)
        self.norm4 = nn.InstanceNorm2d(base_filter)
        self.act = nn.LeakyReLU(0.2, False)

        self.weight = nn.Conv2d(base_filter, base_filter, 3, 1, 1)


        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


    def forward(self, x):
        # x = torch.cat((x, y), 1)
        x = pixel_unshuffle(x, downscale_factor=self.down_factor)
        feat = self.bn(self.input_conv(x))
        out1 = self.act(self.norm1(self.conv1(feat)))
        out2 = self.act(self.norm2(self.conv2(out1)))
        out3 = self.act(self.norm3(self.conv3(out2)))
        out4 = self.act(self.norm4(self.conv4(out3)))

        prob = self.weight(out4)

        b = feat.shape[0]

        prob = prob.view(b, -1)

        out1 = out1.view(b, -1)
        out2 = out2.view(b, -1)
        out3 = out3.view(b, -1)

        feat_out = torch.cat((out1, out2, out3), 1)

        return feat_out, prob


### Define Vgg19 for projected distribution loss
class Vgg19_feat(nn.Module):
    def __init__(self, device, requires_grad=False):
        super(Vgg19_feat, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        self.L1_loss = nn.L1Loss(size_average=False).to(device)
        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def feat_extract(self, x):
        x = (x - self.mean) / self.std
        h = self.slice1(x)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h

        return h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1

    def forward(self, x, y):
        x_feat1, x_feat2, x_feat3, x_feat4 = self.feat_extract(x)
        y_feat1, y_feat2, y_feat3, y_feat4 = self.feat_extract(y)

        ContentLoss = self.L1_loss(x_feat4, y_feat4)

        # style loss
        StyleLoss = 0
        mean_x, var_x = calc_mean_std(x_feat1)
        mean_style, var_style = calc_mean_std(y_feat1)
        StyleLoss = StyleLoss + self.L1_loss(mean_x, mean_style)
        StyleLoss = StyleLoss + self.L1_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(x_feat2)
        mean_style, var_style = calc_mean_std(y_feat2)
        StyleLoss = StyleLoss + self.L1_loss(mean_x, mean_style)
        StyleLoss = StyleLoss + self.L1_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(x_feat3)
        mean_style, var_style = calc_mean_std(y_feat3)
        StyleLoss = StyleLoss + self.L1_loss(mean_x, mean_style)
        StyleLoss = StyleLoss + self.L1_loss(var_x, var_style)

        mean_x, var_x = calc_mean_std(x_feat4)
        mean_style, var_style = calc_mean_std(y_feat4)
        StyleLoss = StyleLoss + self.L1_loss(mean_x, mean_style)
        StyleLoss = StyleLoss + self.L1_loss(var_x, var_style)

        return ContentLoss, StyleLoss


class Vgg19Conv2(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19Conv2, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        # self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()

        # for x in range(2):
        #     self.slice1.add_module(str(x), vgg_pretrained_features[x])

        for x in range(23):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        # fixed pretrained vgg19 model for feature extraction
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        # out1 = self.slice1(x)
        out2 = self.slice2(x)

        return out2

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


class PDLoss(nn.Module):
    def __init__(self, device, l1_lambda=1.5, w_lambda=0.01, average=True):
        super(PDLoss, self).__init__()
        self.vgg = Vgg19Conv2().to(device)
        self.criterionL1 = nn.L1Loss(size_average=average)
        self.w_lambda = w_lambda
        self.l1_lambda = l1_lambda

        mean = torch.Tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
        std = torch.Tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)

    def w_distance(self, xvgg, yvgg):
        x_mean, x_var = calc_mean_std(xvgg)
        y_mean, y_var = calc_mean_std(yvgg)
        xvgg = xvgg / (torch.sum(xvgg, dim=(2, 3), keepdim=True) + 1e-14)
        yvgg = yvgg / (torch.sum(yvgg, dim=(2, 3), keepdim=True) + 1e-14)

        xvgg = xvgg.view(xvgg.size()[0], xvgg.size()[1], -1)
        yvgg = yvgg.view(yvgg.size()[0], yvgg.size()[1], -1)

        cdf_xvgg = torch.cumsum(xvgg, dim=-1)
        cdf_yvgg = torch.cumsum(yvgg, dim=-1)

        cdf_distance = torch.sum(torch.abs(cdf_xvgg - cdf_yvgg), dim=-1)
        # cdf_loss = cdf_distance.mean()
        cdf_loss = cdf_distance.sum()

        mean_distance = torch.sum(torch.abs(x_mean - y_mean), dim=-1)
        var_distance = torch.sum(torch.abs(x_var - y_var), dim=-1)

        mean_var_loss = mean_distance.sum() + var_distance.sum()

        return cdf_loss, mean_var_loss

    def forward(self, x, y):
        # L1loss = self.criterionL1(x, y) * self.l1_lambda
        # L1loss = 0
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        x_vgg1 = self.vgg(x)
        y_vgg1 = self.vgg(y)

        WdLoss, mean_var_loss = self.w_distance(x_vgg1, y_vgg1)
        WdLoss = WdLoss * self.w_lambda
        # WdLoss_img = self.w_distance(x, y)

        return WdLoss, mean_var_loss

############################################################################################
# Base models
############################################################################################
def rgb2ycbcr(img):

    Y = 0. + .299 * img[:, 0] + .587 * img[:, 1] + .114 * img[:, 2]

    return Y.unsqueeze(1)

class GatedDense(nn.Module):
    def __init__(self, input_size, output_size, activation=None):
        super(GatedDense, self).__init__()

        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        self.h = nn.Linear(input_size, output_size)
        self.g = nn.Linear(input_size, output_size)

    def forward(self, x):
        h = self.h(x)
        if self.activation is not None:
            h = self.activation( self.h( x ) )

        g = self.sigmoid( self.g( x ) )

        return h * g

class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=False)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=False)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=False)
        self.local_weight1 = nn.Conv2d(input_size, 2 * output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.local_weight2 = nn.Conv2d(output_size, 2 * output_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        mean, var = self.local_weight1(x).chunk(2, dim=1)
        residue = mean + lr * (1 + var)
        h_residue = self.conv3(residue)
        mean, var = self.local_weight2(hr).chunk(2, dim=1)
        return mean + h_residue * (1 + var)


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=False)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=False)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=False)
        self.local_weight1 = nn.Conv2d(input_size, 2 * output_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.local_weight2 = nn.Conv2d(output_size, 2 * output_size, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        mean, var = self.local_weight1(x).chunk(2, dim=1)
        residue = mean + hr * (1 + var)
        l_residue = self.conv3(residue)
        mean, var = self.local_weight2(lr).chunk(2, dim=1)
        return mean + l_residue * (1 + var)


class ResnetBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResnetBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)
        self.conv2 = torch.nn.Conv2d(num_filter, num_filter, kernel_size, stride, padding, bias=bias)

        self.act1 = torch.nn.LeakyReLU()
        self.act2 = torch.nn.LeakyReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)
        out = self.conv2(out)
        out = out + x
        out = self.act2(out)

        return out


class Self_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Self_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale

        self.K = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)

        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(x)
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W

        return output




class Space_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Space_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale

        self.K = torch.nn.Conv2d(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)

    def forward(self, x, y):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        if self.scale > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(y)
        if self.scale > 1:
            V = F.interpolate(V, scale_factor=1 / self.scale, mode='bicubic')
        else:
            V = V

        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)

        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.bmm(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)
        vector = torch.bmm(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        output = y + self.local_weight(O)

        return output


class Cross_attention(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale, t=0):
        super(Cross_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale
        self.t = t
        self.num = output_size // input_size

        self.K = torch.nn.Conv2d(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = torch.nn.Conv2d(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)

        if kernel_size == 1:
            self.local_weight = torch.nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = torch.nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)

    def forward(self, x, y):
        batch_size = y.size(0)
        K = self.K(x)
        if self.t == 0:
            if self.scale > 1:
                K = self.pool(K)
        Q = self.Q(x)
        if self.t == 0:
            if self.scale > 1:
                Q = self.pool(Q)

        V = self.V(y)
        if self.t == 1:
            if self.scale > 1:
                V = self.pool(V)

        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)

        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.bmm(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.bmm(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()

        if self.t == 1:
            O = vector_reshape.view(batch_size, self.output_size, x.size(2), x.size(3))
            O = F.interpolate(O, scale_factor=self.scale, mode='nearest')
        else:
            O = vector_reshape.view(batch_size, self.output_size, y.size(2), y.size(3))

        output = y + self.local_weight(O)

        return output





######################################################################################
def pixel_unshuffle(input, downscale_factor):
    '''
    input: batchSize * c * k*w * k*h
    kdownscale_factor: k
    batchSize * c * k*w * k*h -> batchSize * k*k*c * w * h
    '''
    c = input.shape[1]

    kernel = torch.zeros(size=[downscale_factor * downscale_factor * c,
                               1, downscale_factor, downscale_factor],
                         device=input.device)
    for y in range(downscale_factor):
        for x in range(downscale_factor):
            kernel[x + y * downscale_factor::downscale_factor*downscale_factor, 0, y, x] = 1
    return F.conv2d(input, kernel, stride=downscale_factor, groups=c)


class Upsample(nn.Sequential):
    """Upsample module.
    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. '
                             'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.
    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.
    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)

class ChannelAttention(nn.Module):
    """Channel attention used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
    """

    def __init__(self, num_feat, squeeze_factor=16):
        super(ChannelAttention, self).__init__()
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_feat, num_feat // squeeze_factor, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_feat // squeeze_factor, num_feat, 1, padding=0),
            nn.Sigmoid())

    def forward(self, x):
        y = self.attention(x)
        return x * y


class RCAB(nn.Module):
    """Residual Channel Attention Block (RCAB) used in RCAN.
    Args:
        num_feat (int): Channel number of intermediate features.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, squeeze_factor=16, res_scale=1):
        super(RCAB, self).__init__()
        self.res_scale = res_scale

        self.rcab = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            ChannelAttention(num_feat, squeeze_factor))

    def forward(self, x):
        res = self.rcab(x) * self.res_scale
        return res + x


class ResidualGroup(nn.Module):
    """Residual Group of RCAB.
    Args:
        num_feat (int): Channel number of intermediate features.
        num_block (int): Block number in the body network.
        squeeze_factor (int): Channel squeeze factor. Default: 16.
        res_scale (float): Scale the residual. Default: 1.
    """

    def __init__(self, num_feat, num_block, squeeze_factor=16, res_scale=1):
        super(ResidualGroup, self).__init__()

        self.residual_group = make_layer(
            RCAB,
            num_block,
            num_feat=num_feat,
            squeeze_factor=squeeze_factor,
            res_scale=res_scale)
        self.conv = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

    def forward(self, x):
        res = self.conv(self.residual_group(x))
        return res + x


class one_conv(nn.Module):
    def __init__(self, G0, G):
        super(one_conv, self).__init__()
        self.conv = nn.Conv2d(G0, G, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu = nn.LeakyReLU(0.1, inplace=True)
    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), dim=1)


class RDB(nn.Module):
    def __init__(self, G0, C, G):
        super(RDB, self).__init__()
        convs = []
        for i in range(C):
            convs.append(one_conv(G0+i*G, G))
        self.conv = nn.Sequential(*convs)
        self.LFF = nn.Conv2d(G0+C*G, G0, kernel_size=1, stride=1, padding=0, bias=True)
    def forward(self, x):
        out = self.conv(x)
        lff = self.LFF(out)
        return lff + x


class RDG(nn.Module):
    def __init__(self, G0, C, G, n_RDB):
        super(RDG, self).__init__()
        self.n_RDB = n_RDB
        RDBs = []
        for i in range(n_RDB):
            RDBs.append(RDB(G0, C, G))
        self.RDB = nn.Sequential(*RDBs)
        self.conv = nn.Conv2d(G0*n_RDB, G0, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        buffer = x
        temp = []
        for i in range(self.n_RDB):
            buffer = self.RDB[i](buffer)
            temp.append(buffer)
        buffer_cat = torch.cat(temp, dim=1)
        out = self.conv(buffer_cat)
        return out


class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel//16, 1, padding=0, bias=True),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Conv2d(channel//16, channel, 1, padding=0, bias=True),
                nn.Sigmoid())

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class ResB(nn.Module):
    def __init__(self, channels):
        super(ResB, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, groups=4, bias=True),
        )
    def __call__(self,x):
        out = self.body(x)
        return out + x
