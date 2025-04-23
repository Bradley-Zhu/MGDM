#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from pytorch_msssim import ssim
import lpips

#loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
loss_fn_vgg = lpips.LPIPS(net='vgg') # closer to "traditional" perceptual loss, when used for optimization


def mse_calc(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def psnr_calc(img1, img2):
    mse = (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim_3dg(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
    

def tslist2list(tslist):
    return [ele.item() for ele in tslist]


def mse_psnr_ssim(ts1_r,ts2_r):
    msevalue = mse_calc(ts1_r,ts2_r)

    # Calculate the Peak Signal-to-Noise Ratio (PSNR)
    psnrvalue = psnr_calc(ts1_r,ts2_r)

    # Calculate SSIM
    ssimvalue = ssim(ts1_r,ts2_r,size_average=False)
    #ssimvalue = ssim(ts1.view(1,1,ts1.shape[0],ts1.shape[1]), ts2.view(1,1,ts2.shape[0],ts2.shape[1]))
    #ssim = 0
    ts1 = ts1_r.cpu()
    ts2 = ts2_r.cpu()
    if ts1.shape[1] == 2:
        ts1stacked = torch.cat((ts1[:,0:1],ts1[:,1:2],torch.sqrt(ts1[:,0:1]**2+ts1[:,1:2]**2)), dim=1)
        ts2stacked = torch.cat((ts2[:,0:1],ts2[:,1:],torch.sqrt(ts2[:,0:1]**2+ts2[:,1:]**2)), dim=1)
        lpipsvalue = loss_fn_vgg(ts1stacked, ts2stacked)
    elif ts1.shape == 1:
        ts1stacked = torch.cat((ts1[:,0:],ts1[:,0:],ts1[:,0:]), dim=1)
        ts2stacked = torch.cat((ts2[:,0:],ts2[:,0:],ts2[:,0:]), dim=1)
        lpipsvalue = loss_fn_vgg(ts1stacked, ts2stacked)
    else:
        lpipsvalue = loss_fn_vgg(ts1, ts2)
    #print(lpipsvalue)
    #print(lpipsvalue.shape)
    #lpips = LPIPS(ts1, ts2)
    #print('-'*10)
    #print(ssimvalue)
    #print(torch.mean(ssimvalue))
    #print(torch.std(ssimvalue))
    #print(ssimvalue.shape)
    return tslist2list([torch.mean(msevalue), torch.mean(psnrvalue), torch.mean(ssimvalue), torch.mean(lpipsvalue)]),\
    tslist2list([torch.std(msevalue), torch.std(psnrvalue), torch.std(ssimvalue), torch.std(lpipsvalue)])

if __name__ == "__main__":
    a = torch.randn(32,1,128,64)
    b = torch.randn(32,1,128,64)
    print(mse_psnr_ssim(a,b))