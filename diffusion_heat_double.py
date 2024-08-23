
from pathlib import Path
from random import random
import matplotlib.pyplot as plt

from collections import namedtuple
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import nn

from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

import torchvision
from torchvision import transforms as T, utils
from torchvision.io import read_video
from torchvision.transforms import v2
from einops import rearrange, reduce, repeat

from PIL import Image
from tqdm.auto import tqdm

from architectures import Unet
from slice_video import slice_all_and_save, sampling, sampling_nophysics, sample_video, conditional_joint_sampling_img
# gaussian diffusion trainer class

import sys
#sys.path.append('playground/')
import playground.heat_fit_conductance as hf
from playground.optical_flow_fit import unet_inference
from playground.optical_flow import Optical_Flow
from evaluations import mse_psnr_ssim


def heatsource_to_pde(heatsource, dinvmat, lookback=5):
    physics_context = []
    b,dt,w,h = heatsource.shape
    for i in range(0,dt):                
        for j in range(0,lookback):
            if i - j < 0:
                break
            if j == 0:
                predict = dinvmat[3]* torch.sigmoid(dinvmat[2])**j * hf.greensfunction(heatsource[:,i-j], dinvmat[0], dinvmat[1], 1 + j)
            else:
                predict += dinvmat[3]* torch.sigmoid(dinvmat[2])**j * hf.greensfunction(heatsource[:,i-j], dinvmat[0], dinvmat[1], 1 + j)
                    #print(predict.shape)
                    #return
        # predict has shape batch x w x h
        physics_context.append(predict)
    # physics_context has shape batch x ( length x w x h )
    physics_context = torch.stack(physics_context,dim=1)#.unsqueeze(1)
    return physics_context


class NISTDataset(Dataset):
    def __init__(
        self,
        args=dict(),
        lookback=5,
        data_time_window=10,
    ):
        super().__init__()
        statedict = torch.load('playground/intermediate/alldata_withof.pt')
        fulldata = statedict['augmenteddata']
        _, dinvmat = hf.find_conductance(True, False)

        args['dinvmat'] = dinvmat
        self.dinvmat = dinvmat
        self.lookback = lookback
        self.data_time_window = data_time_window

        n,w,h = fulldata[0][0].shape  
        # shape is [10 for data, 10 for heatsource, 4x2 for optical flow] x w x h     
        self.dataset = [torch.cat((dt[0],dt[1]), dim=0) for dt in fulldata]
    
        print('Done preprocessing, dataset contains %s clips'%(len(self.dataset)))

    def slice_data(self, rdata, device):
        data = rdata[:,:self.data_time_window].to(device)
        heatsource = rdata[:,self.data_time_window:2*self.data_time_window].to(device) 
        # physics_context has shape batch x ( length x w x h )
        physics_context = heatsource_to_pde(heatsource, self.dinvmat, self.lookback)
            
        # the line following is repeated in functions where it is called
        #physics_context = rearrange(physics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 
        
        return data, heatsource, physics_context
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return self.dataset[index]

def of_loss(xt_folded, physics_folded, v, of, time_window=5):
    xt = rearrange(xt_folded, '(b d1) c w h -> b (d1 c) w h ', d1=time_window, c=1) 
    bs = len(xt)
    # rearrange
    xt_new = rearrange(xt[:,:-1], 'b t w h -> (b t) 1 w h ') 
    v_new  = rearrange(v, 'b t w h c-> (b t) w h c', c=2) 

    predicted_xt = of.forward(xt_new,v_new) 
    predicted_xt = rearrange(predicted_xt, '(b t) 1 w h -> b t w h ', t=time_window-1) 
    physics = rearrange(physics_folded, '(b t) 1 w h -> b t w h ', b = bs) 
    physics_mask = physics < 0.01
    physics_mask = physics_mask[:,-(time_window-1):]
    predicted_xt = predicted_xt.detach()
    predict_mask = predicted_xt > -0.01
    return torch.sum((physics_mask*predict_mask*torch.abs(xt[:,1:]-predicted_xt))**2)


def genmask(image,dh,dw):
    # Create a sample tensor image
    #image = torch.randn(2, 3, 4, 4)  # b x c x h x w
    b,c,w,h = image.shape
    # Find the index of the largest element in each (b x c) dimensions
    max_values, max_indices = torch.max(image.view(image.size(0), image.size(1), -1), dim=2)

    # Create a mask tensor with dh x dw boxes around the largest element
    #dh, dw = 1, 1  # Height and width of the box around the largest element

    # Create indices for selecting the surrounding box around the max element
    b_idx, c_idx = torch.meshgrid(torch.arange(image.size(0)), torch.arange(image.size(1)))
    h_idx, w_idx = max_indices // image.size(-1), max_indices % image.size(-1)

    
    # Create masks for the surrounding box to be zero and all others to be one
    mask = torch.ones_like(image)
    h_start = torch.clamp(h_idx - dh, 0, image.size(2)-1)#.unsqueeze(2)
    h_end = torch.clamp(h_idx + dh + 1, 0, image.size(2))#.unsqueeze(2)
    w_start = torch.clamp(w_idx - dw, 0, image.size(3)-1)#.unsqueeze(2)
    w_end = torch.clamp(w_idx + dw + 1, 0, image.size(3))#.unsqueeze(2)

    for i in range(b):
        for j in range(c):
            mask[i, j, h_start[i, j]:h_end[i, j], w_start[i, j]:w_end[i, j]] = 0
    return mask

def of_loss_small(xt_original, xt_new, physics_mask, v, of):
    xt = xt_original #rearrange(xt_folded, '(b d1) c w h -> b (d1 c) w h ', d1=time_window, c=1) 
    bs = len(xt)    

    predicted_xt = of.forward(xt_original,v) 
    
    #predicted_xt = rearrange(predicted_xt, '(b t) 1 w h -> b t w h ', t=time_window-1) 
    #physics_mask = physics < 0.001
    
    return torch.sum((physics_mask*torch.abs(xt_new-predicted_xt))**2)


def sequential_sample(unet,flownet,beta,alpha,baralpha,timelist,epoch,rdata,dinvmat,ds,args):
    device=args['device']
    lookback = ds.lookback
    data_time_window = ds.data_time_window
    
    data, heatsource, rphysics_context = ds.slice_data(rdata,device)
    
    # calculate the predicted flow
    predicted_flow = unet_inference(flownet,rphysics_context)
    # predicted flow has a shape of batch x 4 x w x h x 2
    predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> b t w h d1', d1=2) 

    of = Optical_Flow()

    # we only predict the second half
    physics_context = rearrange(rphysics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 
    
    bs, dt, w, h = data.shape   
    
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    xtlist = []
    # (b dt) 1 w h
    for j in range(data_time_window//2):
        xt = torch.randn((bs,1,w,h), device=beta.device)
        init_context = torch.cat([data[i:i+1,:2].repeat(1, 1, 1, 1) for i in range(bs)])
        
        #print('Sampling in epoch %s ...'%epoch)
        strength = 0.005
        sub_physics_context = torch.cat([physics_context[ii*data_time_window//2+j:ii*data_time_window//2+(j+1),:,:,:] for ii in range(bs)], dim=0)
        sub_physics_mask = genmask(sub_physics_context, 10,20).detach()
        
        for tid in range(Time-1,0,-1):
            t = timelist[tid:(tid+1)]
            if not args['physics_guided']:
                physics_context *= 0
            inputts = torch.cat((xt, sub_physics_context, init_context,) ,dim=1)
            with torch.no_grad():
                epsilon = unet(inputts,t)
                x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            if j > 0:
                for zz in range(3):
                    # apply optical flow loss
                    tsxt = torch.zeros(x0.shape, device = x0.device, requires_grad = True)
                    with  torch.no_grad():
                        tsxt.data = tsxt.data + x0.data
                    #print(xtlist[-1].shape)
                    #print(tsxt.shape)
                    #print(sub_physics_context.shape)
                    
                    loss = of_loss_small(xtlist[-1], tsxt, sub_physics_mask, predicted_flow[:,j-1].detach(), of)
                    loss.backward()
                    grad_x = tsxt.grad * strength
                    #beta2 = 0.1
                    with torch.no_grad():
                        x0 = x0 + (-grad_x)#1/torch.sqrt(alpha[t])*(- (1-alpha[t])*grad_x)
                        #xt = rearrange(xt, '(b d1) c w h -> b (d1 c) w h ', d1=data_time_window//2, c=1) 
                        #xt[:,1:] = (1 - beta2) * xt[:,1:] + beta2*predicted_from_of
                        #xt = rearrange(xt, 'b (d1 c) w h -> (b d1) c w h', d1=data_time_window//2, c=1) 
            with torch.no_grad():
                xt = torch.sqrt(baralpha[t-1])*x0 + torch.sqrt(1-baralpha[t-1])*epsilon#1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*)
                #xt += sigma[t] * torch.randn((bs*data_time_window//2,1,w,h), device=data.device)
            
        x0[x0>1] = 1
        x0[x0<0] = 0
        xtlist.append(x0)
    # xtlist is a list of tensors, indexed by time
    # each element has dimension batch x 1 x w x h
    # data has shape bs x timelength (double prediction window) x w x h
    prediction = torch.stack([xtlist[i][:,0] for i in range(len(xtlist))])
 
    prediction = prediction.permute(1, 0, 2, 3)
    sourcext = rearrange(prediction[:,:-1], 'b t w h -> (b t) 1 w h') 
    targetxt = rearrange(prediction[:,1:], 'b t w h -> (b t) 1 w h') 
    #print(sourcext.shape)
    #print(targetxt.shape)
    predicted_flow_1 = rearrange(predicted_flow, ' b t w h d1 -> (b t) w h d1') 
    #print(sub_physics_mask.shape)
    #print(predicted_flow.shape)
    #sub_physics_mask = rearrange(sub_physics_mask[:,data_time_window//2:-1], 'b t w h -> (b t) 1 w h')
    result_dict, _ = mse_psnr_ssim(data[:,data_time_window//2:].reshape(-1,1,*prediction.shape[-2:]), prediction.reshape(-1,1,*prediction.shape[-2:]))
    # the first element should have dimension (bs* time) x 1 x w x h, same as the second
    # mask should have dimension the same
    #consistency_loss = of_loss_small(prediction[:,:-1], prediction[:,1:], sub_physics_mask, predicted_flow.detach(), of).detach()
    sub_physics_context_new = rphysics_context#torch.cat([physics_context[ii*data_time_window//2+j:ii*data_time_window//2+(j+1),:,:,:] for ii in range(bs)], dim=0)
    sub_physics_mask_new = genmask(sub_physics_context_new, 10,20).detach()
    sub_physics_mask_new = rearrange(sub_physics_mask_new[:,data_time_window//2:-1], 'b t w h -> (b t) 1 w h')

    consistency_loss = of_loss_small(sourcext, targetxt, sub_physics_mask_new, predicted_flow_1, of)
    consistency_loss = consistency_loss.item()/(1e-10+torch.sum(torch.sum((sub_physics_mask_new*torch.abs(sourcext))**2)))
    result_dict.append(consistency_loss)

    SAVEPLOT = True
    if SAVEPLOT:
        plt.clf()
        plt.close('all')
        x = np.arange(64)
        y = np.arange(32)
        X, Y = np.meshgrid(x, y)

        num_tstamp = data_time_window//2
        delta = data_time_window//2 // num_tstamp
        fig, axes = plt.subplots(3 * bs, num_tstamp, figsize=(num_tstamp* 4, 1.5*bs * 4))

        # Iterate over the tensors and plot them as greyscale/colorful images
        for i in range(bs):
            for j in range(num_tstamp):
                image = data[i,data_time_window//2 + j*delta].detach().cpu().numpy()
                axes[3*i,j].imshow(image, cmap='afmhot')
                axes[3*i,j].axis('off')

                image = physics_context[i*data_time_window//2 + j*delta,0].detach().cpu().numpy()
                axes[3*i+1,j].imshow(image,)
                axes[3*i+1,j].axis('off') 
                if j < -10:#num_tstamp-1:#
                    axes[3*i+1,j].quiver(X, Y, 
                                predicted_flow[i,j,:,:,0].detach().cpu().numpy(), 
                                predicted_flow[i,j,:,:,1].detach().cpu().numpy(), 
                                scale=10) 
        
                image = xtlist[j][i,0].detach().cpu().numpy()
                axes[3*i+2,j].imshow(image, cmap='afmhot')
                axes[3*i+2,j].axis('off')   
        plt.savefig('samples/double_physics/epoch_%s.png'%epoch,bbox_inches='tight')
    RETURNSAMPLES = True
    if RETURNSAMPLES:
        return prediction
    else:
        return result_dict   


def sample_std(unet,flownet,beta,alpha,baralpha,timelist,epoch,rdata,dinvmat,ds,args):
    device=args['device']
    lookback = ds.lookback
    data_time_window = ds.data_time_window
    
    batchsize,c,w,h = rdata.shape
    with torch.no_grad():
        for i in range(1,batchsize):
            rdata[i] *= 0
            rdata[i] += rdata[0]
        rdata = torch.cat((rdata,rdata,rdata,rdata), dim=0)
    data, heatsource, rphysics_context = ds.slice_data(rdata,device)
    
    # calculate the predicted flow
    predicted_flow = unet_inference(flownet,rphysics_context)
    # predicted flow has a shape of batch x 4 x w x h x 2
    predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> b t w h d1', d1=2) 

    of = Optical_Flow()

    # we only predict the second half
    physics_context = rearrange(rphysics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 
    
    bs, dt, w, h = data.shape   
    
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    xtlist = []
    # (b dt) 1 w h
    for j in range(data_time_window//2):
        xt = torch.randn((bs,1,w,h), device=beta.device)
        init_context = torch.cat([data[i:i+1,:2].repeat(1, 1, 1, 1) for i in range(bs)])
        
        #print('Sampling in epoch %s ...'%epoch)
        strength = 0.005
        sub_physics_context = torch.cat([physics_context[ii*data_time_window//2+j:ii*data_time_window//2+(j+1),:,:,:] for ii in range(bs)], dim=0)
        sub_physics_mask = genmask(sub_physics_context, 10,20).detach()
        
        for tid in range(Time-1,0,-1):
            t = timelist[tid:(tid+1)]
            if not args['physics_guided']:
                physics_context *= 0
            inputts = torch.cat((xt, sub_physics_context, init_context,) ,dim=1)
            with torch.no_grad():
                epsilon = unet(inputts,t)
                x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            if j > 0:
                for zz in range(0):
                    # apply optical flow loss
                    tsxt = torch.zeros(x0.shape, device = x0.device, requires_grad = True)
                    with  torch.no_grad():
                        tsxt.data = tsxt.data + x0.data
                    #print(xtlist[-1].shape)
                    #print(tsxt.shape)
                    #print(sub_physics_context.shape)
                    
                    loss = of_loss_small(xtlist[-1], tsxt, sub_physics_mask, predicted_flow[:,j-1].detach(), of)
                    loss.backward()
                    grad_x = tsxt.grad * strength
                    #beta2 = 0.1
                    with torch.no_grad():
                        x0 = x0 + (-grad_x)#1/torch.sqrt(alpha[t])*(- (1-alpha[t])*grad_x)
                        #xt = rearrange(xt, '(b d1) c w h -> b (d1 c) w h ', d1=data_time_window//2, c=1) 
                        #xt[:,1:] = (1 - beta2) * xt[:,1:] + beta2*predicted_from_of
                        #xt = rearrange(xt, 'b (d1 c) w h -> (b d1) c w h', d1=data_time_window//2, c=1) 
            with torch.no_grad():
                xt = torch.sqrt(baralpha[t-1])*x0 + torch.sqrt(1-baralpha[t-1])*epsilon#1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*)
                #xt += sigma[t] * torch.randn((bs*data_time_window//2,1,w,h), device=data.device)
            
        x0[x0>1] = 1
        x0[x0<0] = 0
        xtlist.append(x0)
    # xtlist is a list of tensors, indexed by time
    # each element has dimension batch x 1 x w x h
    # data has shape bs x timelength (double prediction window) x w x h
    prediction = torch.stack([xtlist[i][:,0] for i in range(len(xtlist))])
 
    prediction = prediction.permute(1, 0, 2, 3)
    #sourcext = rearrange(prediction[:,:-1], 'b t w h -> (b t) 1 w h') 
    predict_std = torch.std(prediction, dim=0)
    
    SAVEPLOT = True
    if SAVEPLOT:
        plt.clf()
        plt.close('all')
        x = np.arange(64)
        y = np.arange(32)
        X, Y = np.meshgrid(x, y)

        num_tstamp = data_time_window//2
        delta = data_time_window//2 // num_tstamp
        fig, axes = plt.subplots(2 , num_tstamp, figsize=(num_tstamp* 4, 4))

        # Iterate over the tensors and plot them as greyscale/colorful images
        
        for j in range(num_tstamp):
            image = data[i,data_time_window//2 + j*delta].detach().cpu().numpy()
            axes[0,j].imshow(image, cmap='afmhot')
            axes[0,j].axis('off')

            image = predict_std[j].detach().cpu().numpy()
            axes[1,j].imshow(image, cmap='afmhot')
            axes[1,j].axis('off')   
        plt.savefig('samples/double_physics/diffusion_std.png',bbox_inches='tight')
    return 0,0   


def sampling_with_physics(unet,beta,alpha,baralpha,timelist,epoch,rdata,dinvmat,ds,args, SAVEPLOT=True, CALCMETRICS=False, ADDCONSISTENCYSCORE=False, flownet=None):
    device=args['device']
    lookback = ds.lookback
    data_time_window = ds.data_time_window
    
    data, heatsource, physics_context = ds.slice_data(rdata,device)
    # we only predict the second half
    physics_context = rearrange(physics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 
    bs, dt, w, h = data.shape        
     
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    time_span = 100
    # (b dt) 1 w h
    xt = torch.randn((bs*data_time_window//2,1,w,h), device=beta.device)
    init_context = torch.cat([data[i:i+1,:2].repeat(data_time_window//2, 1, 1, 1) for i in range(bs)])
    with torch.no_grad():
        trajectory = [xt]
        print('Sampling in epoch %s ...'%epoch)
        for tid in tqdm(range(Time-1,-1,-1)):
            t = timelist[tid:(tid+1)]
            if not args['physics_guided']:
                physics_context *= 0
            
            inputts = torch.cat((xt, physics_context, init_context,) ,dim=1)
            #xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputts,t))
            #xt += sigma[t] * torch.randn((bs*data_time_window//2,1,w,h), device=beta.device)
            
            epsilon = unet(inputts,t)
            x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            xt = torch.sqrt(baralpha[t-1])*x0 + torch.sqrt(1-baralpha[t-1])*epsilon#1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*)

            if t%time_span == 0:
                trajectory.append(x0)
        
        trajectory.append(x0)
    result_dict = []
    if CALCMETRICS:
        xtreshape =  rearrange(x0, '(b t) c w h -> b (t c) w h ', c=1, b=bs) 
        metrics, _ = mse_psnr_ssim(data[:,data_time_window//2:].reshape(-1,1,*xtreshape.shape[-2:]),xtreshape.reshape(-1,1,*xtreshape.shape[-2:]))
        result_dict = metrics
    
    if ADDCONSISTENCYSCORE:
        heatsource = rdata[:,data_time_window:2*data_time_window].to(device) 
                
        # physics_context has shape batch x ( length x w x h )
        physics_context_1 = heatsource_to_pde(heatsource, dinvmat, lookback)
        #physics_context = rearrange(physics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 

        sub_physics_context = physics_context_1#torch.cat([physics_context[ii*data_time_window//2+j:ii*data_time_window//2+(j+1),:,:,:] for ii in range(bs)], dim=0)
        sub_physics_mask = genmask(sub_physics_context, 10,20).detach()
        
        predicted_flow = unet_inference(flownet, physics_context_1)
        # predicted flow has a shape of batch x 4 x w x h x 2
        predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> (b t) w h d1', d1=2) 
        sourcext = rearrange(xtreshape[:,:-1], 'b t w h -> (b t) 1 w h') 
        targetxt = rearrange(xtreshape[:,1:], 'b t w h -> (b t) 1 w h') 
        sub_physics_mask = rearrange(sub_physics_mask[:,data_time_window//2:-1], 'b t w h -> (b t) 1 w h')
        #print(sourcext.shape)
        #print(targetxt.shape)
        #print(predicted_flow.shape)
        of = Optical_Flow()
        consistency_loss = of_loss_small(sourcext, targetxt, sub_physics_mask, predicted_flow.detach(), of).detach()/(1e-10+torch.sum(torch.sum((sub_physics_mask*torch.abs(sourcext))**2)))
        result_dict.append(consistency_loss)

    if SAVEPLOT:
        # Create a figure with N subplots
        plt.clf()
        plt.close('all')
        num_tstamp = data_time_window//2
        delta = data_time_window//2 // num_tstamp
        fig, axes = plt.subplots(3 * bs, num_tstamp, figsize=(num_tstamp* 4, 1.5*bs * 4))

        # Iterate over the tensors and plot them as greyscale/colorful images
        for i in range(bs):
            for j in range(num_tstamp):
                image = data[i,data_time_window//2 + j*delta].detach().cpu().numpy()
                axes[3*i,j].imshow(image, cmap='afmhot')
                axes[3*i,j].axis('off')

                image = physics_context[i*data_time_window//2 + j*delta,0].detach().cpu().numpy()
                axes[3*i+1,j].imshow(image, cmap='afmhot')
                axes[3*i+1,j].axis('off')  
        
                image = x0[i*data_time_window//2 + j*delta,0].detach().cpu().numpy()
                axes[3*i+2,j].imshow(image, cmap='afmhot')
                axes[3*i+2,j].axis('off')   
        plt.savefig('samples/double_physics/single_epoch_%s.png'%epoch,bbox_inches='tight')
    return result_dict
 

def eval(unet,ds,args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    trainlen = int(len(ds)*0.8)
    data_time_window = args['data_time_window']
    testdl = DataLoader(ds[trainlen:], batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())
    
    flownet = Unet(channels=10, dim=64, out_dim=8).to(device)
    flownet.load_state_dict(torch.load('playground/intermediate/unet_of.pt'))
    flownet.eval()
    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    print(len(beta_list))
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)

    dinvmat = args['dinvmat'].detach()
    lookback = 5

    evallist = []
    for tdata in tqdm(testdl):        
        #result_dict = sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, -1, tdata, dinvmat, ds, args, True, True, True, flownet)
        #result_dict = sequential_sample(unet, flownet, beta_list,alpha_list,bar_alpha_list,timelist, -1, tdata, dinvmat, ds, args)
        result_dict = sample_std(unet, flownet, beta_list,alpha_list,bar_alpha_list,timelist, -1, tdata, dinvmat, ds, args)

        result_dict = torch.tensor(result_dict)
        #print(result_dict)
        evallist.append(result_dict)
        return
    evallist = torch.stack(evallist)
    print('-'*10)
    print(torch.mean(evallist,dim=0))
    print(torch.std(evallist,dim=0))

def train(unet,ds,args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    trainlen = int(len(ds)*0.8)
    data_time_window = args['data_time_window']
    dl = DataLoader(ds[:trainlen], batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(ds[trainlen:], batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    opt = Adam(unet.parameters(), lr = args['train_lr'], weight_decay=5e-6)

    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    print(len(beta_list))
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)

    dinvmat = args['dinvmat'].detach()
    for epoch in range(args['num_epochs']):
        totloss = 0
        totlen = 0
        for rdata in tqdm(dl):
            # data has the dimension batch x time x channel x weight x height
            #data = rdata.to(device)
            
            data, heatsource, physics_context = ds.slice_data(rdata,device)
            bs = len(data)
            t_list = torch.randint(low=0, high=Time, size=(len(physics_context),),device=device)
            noise = torch.randn((len(physics_context),1,data.shape[-2],data.shape[-1]), device=device)

            
            # physics-guided method                
            inputdata = torch.sqrt(bar_alpha_list[t_list])[:,None,None,None]*rearrange(data[:,data_time_window//2:], 'b t  w h -> (b t) 1 w h') + torch.sqrt(1-bar_alpha_list[t_list])[:,None,None,None]*noise
            initial_context = torch.cat([data[i:i+1,:2].repeat(data_time_window//2, 1, 1, 1) for i in range(bs)])

            if not args['physics_guided']:
                # do not input physics information
                physics_context *= 0
            inputdata = torch.cat((inputdata, physics_context, initial_context,) ,dim=1)
            
            try:
                predictions = unet(inputdata, t_list)  
            except:
                print(data.shape)
                #print(rdata[0])                
                print('la fin')
                assert False   
                         
            loss = torch.sum((noise - predictions)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totloss += loss.item()
            totlen += bs
        print("[%s/%s], loss is %.4f"%(epoch, args['num_epochs'], totloss/totlen))
        if epoch % 10 == 0 or epoch < 10:
            for tdata in testdl:
                sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, dinvmat, args)
                break
            torch.save(unet.state_dict(), 'models/model%s.pth'%('_with_physics' if args['physics_guided'] else ''))
    return 0


def thermal_example():
    args = {
        'num_epochs': 51,
        'length':5,
        'train_lr': 1e-4,
        'train_batch_size': 8,
        'T':1000,        
        'beta_start':1e-4,
        'beta_end':0.02,
        'physics_guided':0,
        'device':'cuda',
    }
    args['data_time_window'] = 10
    dataset = NISTDataset(args,lookback=5, data_time_window=args['data_time_window'])
    
    #analyze_kappa(dataset,args)
    unet = Unet(channels=4, dim=64, out_dim=1).to(args['device'])
    #unet = Unet(channels=args['length'], dim=64, out_dim=args["length"]).to(args['device'])
    
    #train(unet,dataset,args)

    
    unet.load_state_dict(torch.load('models/model%s.pth'%('_with_physics' if args['physics_guided'] else '')))
    unet.eval()
    eval(unet, dataset, args)


if __name__ == "__main__":
    seed = 0
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    thermal_example()
    #CIFAR10_example()
