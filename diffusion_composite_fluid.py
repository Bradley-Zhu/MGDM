

from random import random
import matplotlib.pyplot as plt

from collections import namedtuple
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, utils
from torchvision.io import read_video
from torchvision.transforms import v2
from einops import rearrange, reduce, repeat

from tqdm.auto import tqdm

from architectures import Unet
from utils import uxy_to_color
from evaluations import mse_psnr_ssim

# readin data
def read_b_data(path='fluid_data_gen/intermediate_dataset/01_'):
    #with open(path+'rotlist_l.npy', 'rb') as f:
    #    rotlist_l = torch.tensor(np.load(f))#.unsqueeze(1)
    #with open(path+'rotlist_m.npy', 'rb') as f:
    #    rotlist_m = torch.tensor(np.load(f))
    with open(path+'rotlist_h.npy', 'rb') as f:
        rotlist_h = torch.tensor(np.load(f),dtype=torch.float32)
    with open(path+'blist_l.npy', 'rb') as f:
        blist_l = torch.tensor(np.load(f),dtype=torch.float32)#.unsqueeze(1)
    with open(path+'blist_m.npy', 'rb') as f:
        blist_m = torch.tensor(np.load(f),dtype=torch.float32)
    with open(path+'blist_h.npy', 'rb') as f:
        blist_h = torch.tensor(np.load(f),dtype=torch.float32)
    return blist_l, blist_m, blist_h, rotlist_h

# dataset classes
class FluidDataset(Dataset):
    def __init__(
        self,
        folder,
        from_raw=True
    ):
        super().__init__()
        intermediate_path = 'taichigen/intermediate/dataset.pt'
        if from_raw:
            self.gendata_from_raw(folder,intermediate_path)
        else:
            self.load_data(intermediate_path)
        
        self.preprocess()
        print('Done preprocessing')

    def slice_data(self, rdata, device):
        initial_context = rdata[:,:self.gaps[0]].to(device)
        physics_context_1 = rdata[:,self.gaps[0]:self.gaps[1]].to(device)
        physics_context_2 = rdata[:,self.gaps[1]:self.gaps[2]].to(device)
        data = rdata[:,self.gaps[2]:].to(device)
        return initial_context, physics_context_1, physics_context_2, data
    def preprocess(self):        
        self.cs1_list = self.transform_cs1_context(torch.stack(self.cs1_list, dim=0))
        self.cs2_list = self.transform_cs2_context(torch.stack(self.cs2_list, dim=0))
        self.init_context_list = torch.stack(self.init_context_list, dim=0)
        self.data_list = torch.stack(self.data_list, dim=0)
        with torch.no_grad():
            maxnorm = 10
            for dt in [self.cs1_list, self.cs2_list, self.data_list]:
                #maxnorm = torch.max(torch.abs(dt))
                dt = dt/maxnorm
                #print(maxnorm)
            self.init_context_list[:,1:] = self.init_context_list[:,1:]/maxnorm
        self.full_data_list = torch.cat((self.init_context_list, self.cs1_list, self.cs2_list, self.data_list), dim=1)
        self.gaps = [self.init_context_list.shape[1],self.init_context_list.shape[1]+self.cs1_list.shape[1],self.init_context_list.shape[1]+self.cs1_list.shape[1]+self.cs2_list.shape[1]]
        #assert False
    def transform_cs1_context(self, input_tensor):
        return F.interpolate(input_tensor, scale_factor=( 4, 4), mode='bilinear', align_corners=False)
    
    def transform_cs2_context(self, input_tensor):
        return F.interpolate(input_tensor, scale_factor=( 2, 2), mode='bilinear', align_corners=False)
    
    def load_data(self, savename):
        state_dict = torch.load(savename)
        self.init_context_list = state_dict['init_context_list']
        self.cs1_list = state_dict['cs1_list']
        self.cs2_list = state_dict['cs2_list']
        self.data_list = state_dict['data_list']
        print('dataset loaded with %s samples'%len(self.cs1_list))
        
    def gendata_from_raw(self, folder, savepath):
        high_res_file_list = [filename for filename in os.listdir(folder) if '_res_128_' in filename]
        middle_res_file_list = [filename[:-12]+'64_data.npy' for filename in high_res_file_list]
        low_res_file_list = [filename[:-12]+'32_data.npy' for filename in high_res_file_list]
        self.init_context_list = []
        self.cs1_list = []
        self.cs2_list = []
        self.data_list = []
        print("constructing datasets with %s samples"%len(low_res_file_list))
        for i in tqdm(range(len(low_res_file_list))):
            with open(folder+low_res_file_list[i], 'rb') as f:
                lowresi = np.load(f)
            with open(folder+middle_res_file_list[i], 'rb') as f:
                middleresi = np.load(f)
            with open(folder+high_res_file_list[i], 'rb') as f:
                highresi = np.load(f)
            # above are all  w x h x 20 tensors
            # 20 = [vx, vy, p, rot, wall] x 4   
            init_context = np.stack((highresi[:,:,4], highresi[:,:,5], highresi[:,:,6]),axis=0)
            cs1 = np.stack((lowresi[:,:,10], lowresi[:,:,11]), axis=0)
            cs2 = np.stack((middleresi[:,:,10], middleresi[:,:,11]), axis=0)
            datai = np.stack((highresi[:,:,10], highresi[:,:,11]), axis=0)
            self.init_context_list.append(torch.tensor(init_context))
            self.cs1_list.append(torch.tensor(cs1))
            self.cs2_list.append(torch.tensor(cs2))
            self.data_list.append(torch.tensor(datai))
        torch.save({
            'init_context_list':self.init_context_list,
            'cs1_list':self.cs1_list,
            'cs2_list':self.cs2_list,
            'data_list':self.data_list,
                    }, savepath)
        #self.data = [dt for dt in self.data]
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        #print('-'*10)
        #print(len(self.init_context_list[index]))
        #print(len(self.cs1_list[index]))
        #print('-'*10)
        return self.full_data_list[index]
        #return self.init_context_list[index], self.transform_cs1_context(self.cs1_list[index]), self.transform_cs2_context(self.cs2_list[index]), self.data_list[index]

class SmokeDataset(Dataset):
    def __init__(
        self,
        suffix=['01','001','0001','00001'],
    ):
        super().__init__()
        intermediate_path = 'fluid_data_gen/intermediate_dataset/'
        self.blist_l, self.blist_m, self.blist_h, self.rotlist_h = [], [], [], []
        for pat in suffix:
            blist_l, blist_m, blist_h, rotlist_h = read_b_data(intermediate_path+pat+'_')
            self.blist_l.append(blist_l) 
            self.blist_m.append(blist_m) 
            self.blist_h.append(blist_h)
            self.rotlist_h.append(rotlist_h)

        self.preprocess()
        print('Done preprocessing, dataset contains %s samples'%(len(self.full_data_list)))

    def slice_data(self, rdata, device):
        initial_context = rdata[:,:self.gaps[0]].to(device)
        physics_context_1 = rdata[:,self.gaps[0]:self.gaps[1]].to(device)
        physics_context_2 = rdata[:,self.gaps[1]:self.gaps[2]].to(device)
        data = rdata[:,self.gaps[2]:].to(device)
        return initial_context, physics_context_1, physics_context_2, data
    
    def preprocess(self):        
        self.cs1_list = self.transform_cs1_context(torch.cat(self.blist_l, dim=0)[:,-2:])
        self.cs2_list = self.transform_cs2_context(torch.cat(self.blist_m, dim=0)[:,-1:])
        self.init_context_list = torch.cat(self.blist_h, dim=0)[:,0:1]
        self.init_context_list = torch.cat((self.init_context_list,torch.cat(self.rotlist_h, dim=0).unsqueeze(1)),dim=1)
        self.data_list = torch.cat(self.blist_h, dim=0)[:,1:]
        with torch.no_grad():
            #maxnorm = [16.,1.8,2.2,2.1]
            maxnorm = [1.,1.,1.,1.]
            for dtid, dt in enumerate([self.cs1_list, self.cs2_list, self.data_list, self.init_context_list]):
                #maxnorm = torch.max(torch.abs(dt))
                dt = dt/maxnorm[dtid]
                #print(dt.shape)
                #print(torch.max(torch.abs(dt)))
        self.full_data_list = torch.cat((self.init_context_list, self.cs1_list, self.cs2_list, self.data_list), dim=1)
        self.gaps = [self.init_context_list.shape[1],self.init_context_list.shape[1]+self.cs1_list.shape[1],self.init_context_list.shape[1]+self.cs1_list.shape[1]+self.cs2_list.shape[1]]
        #assert False

    def transform_cs1_context(self, input_tensor):
        return F.interpolate(input_tensor, scale_factor=( 4, 4), mode='bilinear', align_corners=False)
    
    def transform_cs2_context(self, input_tensor):
        return F.interpolate(input_tensor, scale_factor=( 2, 2), mode='bilinear', align_corners=False)
    
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        #print('-'*10)
        #print(len(self.init_context_list[index]))
        #print(len(self.cs1_list[index]))
        #print('-'*10)
        return self.full_data_list[index]
        #return self.init_context_list[index], self.transform_cs1_context(self.cs1_list[index]), self.transform_cs2_context(self.cs2_list[index]), self.data_list[index]

def plot_vortex(trajectory, wall, physics_context, cs2, gt_data,savename):
    # Create a figure with N subplots
    plt.clf()
    plt.close('all')
    num_tstamp = len(trajectory)

    pbs = 4
    fig, axes = plt.subplots(pbs, num_tstamp+3, figsize=((num_tstamp+3)* 4, pbs * 2.1))

    # Iterate over the tensors and plot them as greyscale/colorful images
    for i in range(pbs):
        #print(physics_context[i,0])
        image = uxy_to_color(physics_context[i,0], physics_context[i,1], wall[i])
        image = np.transpose(image, axes=(1, 0, 2))
        axes[i,0].imshow(image)
        axes[i,0].axis('off')

        image = uxy_to_color(cs2[i,0], cs2[i,1], wall[i])
        image = np.transpose(image, axes=(1, 0, 2))
        axes[i,1].imshow(image, cmap='gray')
        axes[i,1].axis('off')  

        image = uxy_to_color(gt_data[i,0], gt_data[i,1], wall[i])
        image = np.transpose(image, axes=(1, 0, 2))
        axes[i,2].imshow(image, cmap='gray')
        axes[i,2].axis('off')  
        for j in range(num_tstamp):
            data = trajectory[len(trajectory)-1-j]
            image = uxy_to_color(data[i,0], data[i,1], wall[i])
            image = np.transpose(image, axes=(1, 0, 2))
            axes[i,3+j].imshow(image, cmap='gray')
            axes[i,3+j].axis('off')   
    plt.savefig(savename,bbox_inches='tight')

def plot_bouyancy(trajectory, physics_context, cs2, gt_data,savename):
    # Create a figure with N subplots
    plt.clf()
    plt.close('all')
    num_tstamp = len(trajectory)

    pbs = 4
    fig, axes = plt.subplots(pbs, num_tstamp+3, figsize=((num_tstamp+3)* 4, pbs * 4))

    # Iterate over the tensors and plot them as greyscale/colorful images
    for i in range(pbs):
        #print(physics_context[i,0])
        image = physics_context[i,-1].detach().cpu().numpy()
        #image = np.transpose(image, axes=(1, 0, 2))
        axes[i,0].imshow(-image, cmap='coolwarm')
        axes[i,0].axis('off')

        image = cs2[i,-1].detach().cpu().numpy()
        axes[i,1].imshow(-image, cmap='coolwarm')
        axes[i,1].axis('off')  

        image = gt_data[i,-1].detach().cpu().numpy()
        axes[i,2].imshow(-image, cmap='coolwarm')
        axes[i,2].axis('off')  
        for j in range(num_tstamp):
            data = trajectory[len(trajectory)-1-j]
            image = data[i,0].detach().cpu().numpy()
            axes[i,3+j].imshow(-image, cmap='coolwarm')
            axes[i,3+j].axis('off')   
    plt.savefig(savename,bbox_inches='tight')

def sampling_with_physics(unet,beta,alpha,baralpha,timelist,epoch,rdata,ds,args):
    device=args['device']
    init_context, physics_context, cs2, gt_data = ds.slice_data(rdata, device)
    bs, dt, w, h = rdata.shape        
    
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    time_span = 100
    # (b dt) 1 w h
    xt = torch.randn((bs,gt_data.shape[1],w,h), device=beta.device)
    with torch.no_grad():
        trajectory = [xt]
        print('Sampling in epoch %s ...'%epoch)
        for tid in tqdm(range(Time-1,-1,-1)):
            t = timelist[tid:(tid+1)]
            if not args['physics_guided']:
                physics_context *= 0
            #print(xt.shape)
            #print(physics_context.shape)
            #print(init_context.shape)
            inputts = torch.cat((xt, physics_context, init_context,) ,dim=1)
            xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputts,t))
            xt += sigma[t] * torch.randn((bs,1,w,h), device=beta.device)
            if t%time_span == 0:
                trajectory.append(xt)
        trajectory.append(xt)
    savename = 'samples/fluid_vortex_%s_epoch_%s.png'%(args['plot_vortex'],epoch)
    if args['plot_vortex']:
        plot_vortex(trajectory, init_context[:,0], physics_context, cs2, gt_data,savename)
    else:
        plot_bouyancy(trajectory, physics_context, cs2, gt_data,savename)



def expensive_physics_loss(cs2, x, wall=None):
    if wall is None:
        pooled = F.avg_pool2d((cs2-x), kernel_size=(2, 2), stride=(2, 2))
    else:
        pooled = F.avg_pool2d((cs2-x)*wall[:,None,:,:], kernel_size=(4, 4), stride=(4, 4))
    return torch.sum(pooled**2)

def fourier_low_frequency_loss(cs2, x, wall=None):
    xk = torch.fft.fft2(x, dim=(-2, -1))
    csk = torch.fft.fft2(cs2, dim=(-2, -1))
    _,_,n1,n2 = xk.shape
    n1 = n1//4
    n2 = n2//4
    return 0.0001*torch.norm(xk[:,:,:n1,:n2]-csk[:,:,:n1,:n2])**2

def full_sample(unet,beta,alpha,baralpha,timelist,epoch,rdata,ds,args,SAVEPLOT = False):
    device=args['device']
    init_context, physics_context, cs2, gt_data = ds.slice_data(rdata, device)

    bs, dt, w, h = gt_data.shape   

    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    time_span = 100
    # (b dt) 1 w h
    xt = torch.randn(gt_data.shape, device=beta.device)
    trajectory = []
    #return mse_psnr_ssim(gt_data, cs2)

    for tid in range(Time-1,-1,-1):
        t = timelist[tid:(tid+1)]
        
        if not args['physics_guided']:
            physics_context *= 0
        inputts = torch.cat((xt, physics_context, init_context,) ,dim=1)
        with torch.no_grad():
            #xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputts,t))
            #xt += sigma[t] * torch.randn((bs,1,w,h), device=beta.device)
            #continue
            epsilon = unet(inputts,t)

            x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            strength = 0.01
        for zz in range(1):
            # apply expensive physics loss
            tsxt = torch.zeros(xt.shape, device = xt.device, requires_grad = True)
            with  torch.no_grad():
                tsxt.data = tsxt.data + x0.data
            
            #loss = expensive_physics_loss(cs2.detach(), tsxt, init_context[:,0].detach())#init_context[:,0].detach())
            loss = expensive_physics_loss(cs2.detach(), tsxt)
            #loss = fourier_low_frequency_loss(cs2.detach(), tsxt)
            loss.backward()
            grad_x = tsxt.grad * strength
            with torch.no_grad():
                x0 = x0 + (-grad_x)#1/torch.sqrt(alpha[t])*(- (1-alpha[t])*grad_x)
                        #xt = rearrange(xt, '(b d1) c w h -> b (d1 c) w h ', d1=data_time_window//2, c=1) 
                        #xt[:,1:] = (1 - beta2) * xt[:,1:] + beta2*predicted_from_of
                        #xt = rearrange(xt, 'b (d1 c) w h -> (b d1) c w h', d1=data_time_window//2, c=1) 
                #x0 = x0 + (-grad_x)*(1-alpha[t])
        with torch.no_grad():            
            xt = torch.sqrt(baralpha[t-1])*x0 + torch.sqrt(1-baralpha[t-1])*epsilon #1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*)
            #xt += st * torch.randn(xt.shape, device=beta.device)
        
        if SAVEPLOT and t % 200 == 0:
            trajectory.append(x0)
    result_dict, result_dict_std = mse_psnr_ssim(gt_data, x0)
    #result_dict = [ri.item() for ri in result_dict]

    
    if SAVEPLOT:
        savename = 'samples/double_physics_fluid/fluid_vortex_epoch_%s.png'%epoch
        if args['plot_vortex']:
            plot_vortex(trajectory+[x0], init_context[:,0], physics_context, cs2, gt_data, savename)
        else:
            #plot_vortex(trajectory+[x0], init_context[:,0], physics_context, cs2, gt_data, savename)
            plot_bouyancy(trajectory+[x0], physics_context, cs2, gt_data,savename)

    return result_dict, result_dict_std   

def train(unet,ds,args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    trainlen = int(len(ds)*0.9)
    dl = DataLoader(ds[:trainlen], batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(ds[trainlen:], batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())
    opt = Adam(unet.parameters(), lr = args['train_lr'], weight_decay=5e-6)

    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    print(len(beta_list))
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)

    for epoch in range(args['num_epochs']):
        totloss = 0
        totlen = 0
        for rdata in tqdm(dl):
            # data has the dimension batch x time x channel x weight x height
            #data = rdata.to(device)
            initial_context, physics_context, _, data = ds.slice_data(rdata, device)
                     
            t_list = torch.randint(low=0, high=Time, size=(len(physics_context),),device=device)
            noise = torch.randn((len(physics_context),data.shape[1],data.shape[-2],data.shape[-1]), device=device)
            
            # physics-guided method                
            inputdata = torch.sqrt(bar_alpha_list[t_list])[:,None,None,None]*data + torch.sqrt(1-bar_alpha_list[t_list])[:,None,None,None]*noise
     
            if not args['physics_guided']:
                # do not input physics information
                physics_context *= 0
            
            inputdata = torch.cat((inputdata, physics_context, initial_context,) ,dim=1)
            
            
            try:
                predictions = unet(inputdata, t_list)  
            except:
                print(data.shape)
                print(physics_context.shape)
                print(initial_context.shape)
                print(inputdata.shape)
                #print(rdata[0])                
                print('la fin')
                assert False   
                         
            loss = torch.sum((noise - predictions)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totloss += loss.item()
            totlen += len(data)
        print("[%s/%s], loss is %.4f"%(epoch, args['num_epochs'], totloss/totlen))
        if  epoch % 10 == 0 or epoch < 10:
            for tdata in testdl:
                sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, ds,args)
                break
            torch.save(unet.state_dict(), 'models/model_fluid_smoke%s.pth'%('' if args['physics_guided'] else '_nophysique'))
    return 0

def eval(unet,ds,args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    trainlen = int(len(ds)*0.9)
    dl = DataLoader(ds[:trainlen], batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(ds[trainlen:], batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())

    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)
    epoch = -1
    evallist = []
    for tdata in tqdm(testdl):
        #res_dict = ddim_sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, ds,args)
        res_dict, _ = full_sample(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, ds,args,SAVEPLOT=True)
        #print(res_dict)
        evallist.append(res_dict)
        #return
    evallist = torch.tensor(evallist)
    print(torch.mean(evallist, dim=0))
    print(torch.std(evallist, dim=0))
    
def taichi_fluid_example():
    args = {
        'num_epochs': 101,
        'train_lr': 1e-4,
        'train_batch_size': 16,
        'plot_vortex':1,
        'T':1000,        
        'beta_start':1e-4,
        'beta_end':0.02,
        
        'physics_guided':1,
        'device':'cuda',
    }
    video_path = r"taichigen/data/"
    dataset = FluidDataset(folder=video_path,from_raw=False)
    
    #analyze_kappa(dataset,args)
    unet = Unet(channels=7, dim=64, out_dim=2).to(args['device'])
    #unet = Unet(channels=args['length'], dim=64, out_dim=args["length"]).to(args['device'])
    
    #train(unet,dataset,args)
    unet.load_state_dict(torch.load('models/model_fluid_taichi.pth'))
    unet.eval()
    eval(unet,dataset,args)
    #sample_video('data/LBMAM_raw' ,args, unet)

def fluid_example():
    args = {
        'num_epochs': 101,
        'train_lr': 1e-4,
        'train_batch_size': 16,
        'plot_vortex':0,
        'T':1000,
        
        'beta_start':1e-4,
        'beta_end':0.02,
        
        'physics_guided':0,
        'device':'cuda',
    }

    dataset = SmokeDataset()
    unet = Unet(channels=5, dim=64, out_dim=1).to(args['device'])
    #train(unet,dataset,args)

    unet.load_state_dict(torch.load('models/model_fluid_smoke_nophysique.pth'))
    unet.eval()
    eval(unet,dataset,args)

 
if __name__ == "__main__":
    seed = 0
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fluid_example()
    #CIFAR10_example()

