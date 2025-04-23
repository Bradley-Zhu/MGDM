

from random import random
import matplotlib.pyplot as plt

from collections import namedtuple
from multiprocessing import cpu_count

import numpy as np
import torch
from torch import nn
import os
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torchvision
from torchvision import transforms as T, utils
from torchvision.io import read_video
from torchvision.transforms import v2
from einops import rearrange, reduce, repeat
#from diffusers.models import AutoencoderKL

from tqdm.auto import tqdm

from architectures import Unet,Vaenet

from utils import uxy_to_color
from evaluations import mse_psnr_ssim
from ditmodels import MyDiT_S_8, MyDiT_S_2, MyDiT_B_4, MyDiT_B_2
import wandb
import time
# readin raw data
def read_r_data(path='fluid_data_gen/new_dataset'):
    
    #cs1_list, cs2_list, init_context_list, data_list, temps_list
    names = ['cs1_list', 'cs2_list','history_list','data_list','temps_list']
    full_data_list = [[] for i in range(len(names))]
    for filename in os.listdir(path):
        if filename.endswith('.npy') and 'cs1_list' in filename:
            for dtid,dtname in enumerate(names):
                newfilename = filename.replace("cs1_list", dtname)
                with open(os.path.join(path,newfilename), 'rb') as f:
                    slice_data = torch.tensor(np.load(f),dtype=torch.float32)
                    full_data_list[dtid].append(slice_data)
    return (torch.cat(full_data_list[i], dim=0) for i in range(len(names)))

def normalize_tensor(tensor):
    # Compute channel-wise minimum and maximum values
    channel_min = tensor.amin(dim=(0, 2, 3), keepdim=True)  # Min across N, H, W for each channel
    channel_max = tensor.amax(dim=(0, 2, 3), keepdim=True)  # Max across N, H, W for each channel
    print(channel_min,channel_max)
    return
    # Normalize to [0, 1]
    normalized_tensor = (tensor - channel_min) / (channel_max - channel_min + 1e-8)

    # Scale to [-1, 1]
    scaled_tensor = normalized_tensor * 2 - 1

    return scaled_tensor

def gen_train_test_dataset(intermediate_path='fluid_data_gen/new_dataset/',suffix=['01','001','0001','00001'],preprocess=0,train_ratio=0.9,args={}):
    
    
    # reshape data into desired shape
    cs1_list, cs2_list, init_context_list, data_list, temps_list = read_r_data(intermediate_path)
    

    '''
    cs2_list = torch.cat(blist_m, dim=0)[:,-1:]
    init_context_list = torch.cat(blist_h, dim=0)[:,0:1]
    init_context_list = torch.cat((init_context_list,torch.cat(rotlist_h, dim=0).unsqueeze(1)),dim=1)
    data_list = torch.cat(blist_h, dim=0)[:,1:]
    '''
    # normalization
    if True:
        with torch.no_grad():
            # currently there is no normalization
            #maxnorm = [8.,2.5,3.2,3.6]
            cs1_list_norm = torch.tensor([2.3,8.4])
            cs2_list_norm = torch.tensor([2.6])
            data_list_norm = torch.tensor([3.2])
            init_context_list_norm = torch.tensor([2.9,3.6,2.3,1.3])

            cs1_list =  cs1_list/cs1_list_norm[None,:,None,None]
            cs2_list =  cs2_list/cs2_list_norm[None,:,None,None]
            data_list =  data_list/data_list_norm[None,:,None,None]
            init_context_list =  init_context_list/init_context_list_norm[None,:,None,None]

        
            #input()
    
    if preprocess:
        bs = 32
        init_context_list_new = []
        data_list_new = []
        for i in range(len(data_list)//bs):
            pass
    trainlen = int(len(data_list)*train_ratio)
    train_set = SmokeDataset(init_context_list[:trainlen], cs1_list[:trainlen], cs2_list[:trainlen], data_list[:trainlen], temps_list[:trainlen])
    test_set = SmokeDataset(init_context_list[trainlen:], cs1_list[trainlen:], cs2_list[trainlen:], data_list[trainlen:], temps_list[trainlen:])
    print('Done preprocessing, dataset contains %s samples, among which %s are training, %s are testing data'%(len(data_list), len(train_set), len(test_set)))
    return train_set, test_set

# dataset classes
class SmokeDataset(Dataset):
    def __init__(
        self,
        init_context_list,cs1_list,cs2_list,data_list,temps_list
    ):
        super().__init__()
        self.init_context_list = init_context_list
        self.cs1_list = cs1_list
        self.cs2_list = cs2_list
        self.data_list = data_list
        self.temps_list = temps_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # always the list in the following order
        return self.init_context_list[index], self.cs1_list[index], self.cs2_list[index], self.data_list[index], self.temps_list[index]


def plot_bouyancy(trajectory, physics_context, cs2, gt_data,savename):
    # Create a figure with N subplots
    plt.clf()
    plt.close('all')
    num_tstamp = len(trajectory)

    pbs = min(4,len(physics_context))
    fig, axes = plt.subplots(pbs, num_tstamp+3, figsize=((num_tstamp+3)* 4, pbs * 4))

    # Iterate over the tensors and plot them as greyscale/colorful images
    for i in range(pbs):
        #print(physics_context[i,0])
        image = physics_context[i,0].detach().cpu().numpy()
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


def sampling_with_physics(unet,beta,alpha,baralpha,timelist,epoch,rdata,args):
    device=args['device']
    with torch.no_grad():
        initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
        initial_context = args['encode_initial_context'](initial_context)
        physics_context = args['encode_physics_context'](c1_context)
        data = args['encode_data'](odata)    
    bs, dt, w, h = data.shape        
    
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    time_span = 200
    # (b dt) 1 w h
    xt = torch.randn(data.shape, device=beta.device)
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
            inputts = torch.cat((xt, physics_context, initial_context,) ,dim=1)
            #inputdata = torch.cat((inputdata, physics_context, initial_context,) ,dim=1)
            #print(t.shape)
            #print(phs_time.shape)
            #input()
            xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputts,t.repeat(len(phs_time)),phs_time[:,0]))
            xt += sigma[t] * torch.randn(data.shape, device=beta.device)
            if t%time_span == 0:
                trajectory.append(args['decode_data'](xt))
        trajectory.append(args['decode_data'](xt))
    savename = 'samples/fluid_vortex_%s_epoch_%s.png'%(args['plot_vortex'],epoch)
    
    plot_bouyancy(trajectory, physics_context, c2_context, odata, savename)



def expensive_physics_loss(cs2, x, wall=None):
    if wall is None:
        pooled = cs2-F.avg_pool2d((x), kernel_size=(2, 2), stride=(2, 2))
    else:
        pooled = cs2-F.avg_pool2d((x)*wall[:,None,:,:], kernel_size=(4, 4), stride=(4, 4))
    return torch.sum(pooled**2)

def fourier_low_frequency_loss(cs2, x, wall=None):
    xk = torch.fft.fft2(x, dim=(-2, -1))
    csk = torch.fft.fft2(cs2, dim=(-2, -1))
    _,_,n1,n2 = xk.shape
    n1 = n1//4
    n2 = n2//4
    return 0.0001*torch.norm(xk[:,:,:n1,:n2]-csk[:,:,:n1,:n2])**2

def full_sample(unet,beta,alpha,baralpha,timelist,epoch,rdata,args,SAVEPLOT = False):
    device=args['device']
    
    with torch.no_grad():
        initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
        initial_context = args['encode_initial_context'](initial_context)
        physics_context = args['encode_physics_context'](c1_context)
        data = args['encode_data'](odata)    
    bs, dt, w, h = data.shape  
   

    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    time_span = 200
    # (b dt) 1 w h
    xt = torch.randn(data.shape, device=beta.device)
    trajectory = []
    #return mse_psnr_ssim(gt_data, cs2)

    for tid in range(Time-1,-1,-1):
        t = timelist[tid:(tid+1)]        
        if not args['physics_guided']:
            physics_context *= 0
            inputts = torch.cat((xt, initial_context,) ,dim=1)

        else:
            inputts = torch.cat((xt, physics_context, initial_context,) ,dim=1)
        #print(xt.shape) #1
        #print(physics_context.shape) #2
        #print(initial_context.shape) #4
        #print(inputts.shape)
        #input()
        with torch.no_grad():
            #xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputts,t))
            #xt += sigma[t] * torch.randn((bs,1,w,h), device=beta.device)
            #continue
            epsilon = unet(inputts,t.repeat(len(phs_time)),phs_time[:,0])

            x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            strength = 0.01
        for zz in range(0):
            # apply expensive physics loss
            tsxt = torch.zeros(xt.shape, device = xt.device, requires_grad = True)
            with  torch.no_grad():
                tsxt.data = tsxt.data + x0.data
          
            #loss = expensive_physics_loss(cs2.detach(), tsxt, init_context[:,0].detach())#init_context[:,0].detach())
            loss = expensive_physics_loss(c2_context.detach(), tsxt)
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
        
        if SAVEPLOT and  t % 200 == 0:
            with torch.no_grad():
                trajectory.append(args['decode_data'](x0))
    #result_dict, result_dict_std = mse_psnr_ssim(gt_data, x0)
    #result_dict = [ri.item() for ri in result_dict]
    
    trajectory = trajectory+[args['decode_data'](x0)]
    if SAVEPLOT:
        savename = 'samples/double_physics_fluid/fluid_vortex_epoch_%s.png'%epoch
        
        plot_bouyancy(trajectory, c1_context, c2_context, odata,savename)
    
    reconsdata = trajectory[-1]
    with torch.no_grad():
        if args['ldm']:
            reconsdata = args['avg_pool_2'](reconsdata)
            bs = 2
            nbs = len(reconsdata)
            reconsgt = [args['avg_pool_2'](args['decode_data'](data[bid*bs:(bid+1)*bs])) for bid in range(nbs//bs)]
            reconsgt = torch.cat(reconsgt,dim=0)
        else:
            reconsgt = args['decode_data'](data)
    result_dict, result_dict_std = mse_psnr_ssim(reconsgt, reconsdata)
    return trajectory[-1], result_dict
    #return result_dict, result_dict_std   


def train(unet, trainds, testds, args):
    device=args['device']
    dl = DataLoader(trainds, batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    #assert len(trainds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'

    testdl = DataLoader(testds, batch_size = args['train_batch_size']//4, shuffle = False, pin_memory = True,  num_workers = cpu_count())
    opt = AdamW(unet.parameters(), lr = args['train_lr'], weight_decay=5e-6)

    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    print(len(beta_list))
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)

    for epoch in range(args['num_epochs']):
        totloss = 0
        totlen = 0
        time_start = time.time()
        for rdata in tqdm(dl):
            # data has the dimension batch x time x channel x weight x height
            #data = rdata.to(device)
            #initial_context, physics_context, _, data = ds.slice_data(rdata, device)
            with torch.no_grad():                
                oinitial_context, ophysics_context, _, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
                initial_context = args['encode_initial_context'](oinitial_context)
                physics_context = args['encode_physics_context'](ophysics_context)
                data = args['encode_data'](odata)
                
                for idt, dtf in enumerate([initial_context,physics_context,data]):
                    maxnorm = torch.max(torch.abs(dtf))
                    if maxnorm> 1.1:
                        print(maxnorm)


            t_list = torch.randint(low=0, high=Time, size=(len(physics_context),),device=device)
            noise = torch.randn((data.shape), device=device)
            
            # physics-guided method                
            inputdata = torch.sqrt(bar_alpha_list[t_list])[:,None,None,None]*data + torch.sqrt(1-bar_alpha_list[t_list])[:,None,None,None]*noise
     
            if not args['physics_guided']:
                # do not input physics information
                physics_context *= 0
            if args['physics_guided']:
                inputdata = torch.cat((inputdata, physics_context, initial_context,) ,dim=1)
            else:
                inputdata = torch.cat((inputdata, initial_context,) ,dim=1)
            try:
                predictions = unet(inputdata, t_list, phs_time[:,0])  
            except:
                print(data.shape)
                print(physics_context.shape)
                print(initial_context.shape)
                print(inputdata.shape)
                print(t_list.shape)
                print(phs_time.shape)
                #print(rdata[0])                
                print('Above shows error messages')
                assert False   
                         
            loss = torch.sum((noise - predictions)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totloss += loss.item()
            totlen += len(data)

            
        print("[%s/%s], loss is %.4f"%(epoch, args['num_epochs'], totloss/totlen))
        time_end = time.time()
        msg = {"train loss": totloss/totlen, 
                            "epoch": epoch,
                   "time per rpoch": time_end - time_start
                           }
        
        if epoch % 10 == 0 or epoch < 10:
            for tdata in testdl:
                #sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, args)
                image_array,_ = full_sample(unet,beta_list,alpha_list,bar_alpha_list,timelist,epoch,tdata,args,SAVEPLOT = True)
                break
            torch.save(unet.state_dict(), 'models/model_fluid_smoke_%s_%s_ldm_%s.pth'%('' if args['physics_guided'] else '_nophysique',args['net'],args['ldm']))
        else:
            image_array == []   
        if args['use_wandb']:
            
            for ii in range(len(image_array)):
                msg['sample_%s'%ii] = wandb.Image(image_array[ii,0], caption="Sample %s"%ii)

            # log metrics to wandb
            wandb.log(msg)
    return 0

def eval(unet,testds,args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    testdl = DataLoader(testds, batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())

    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)
    epoch = -1
    evallist = []
    for tdata in tqdm(testdl):
   
        _, res_dict = full_sample(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata,args,SAVEPLOT=True)

        evallist.append(res_dict)
        #break
        #return
    evallist = torch.tensor(evallist)
    print(torch.mean(evallist, dim=0))
    print(torch.std(evallist, dim=0))

def channel_augment(batch):
    with torch.no_grad():
        gap = 1 - torch.abs(batch)
        gap[gap<0] = 0
        res = batch.repeat(1,3,1,1)
        noise = torch.rand(batch.shape,device=batch.device)
        res[:,0:1] += noise*gap
        res[:,1:2] -= noise*gap
    return res

def fluid_example():
    args = {
        'num_epochs': 201,
        'train_lr': 1e-4,
        'train_batch_size': 16,
        'plot_vortex':0,
        'T':1000,
        
        'beta_start':1e-4,
        'beta_end':0.02,
        
        'physics_guided':1,
        'device':'cuda',
        'encode_initial_context': lambda x: x,
        'encode_physics_context': lambda x: x,
        'encode_data': lambda x: x,
        'decode_data': lambda x: x,
        'use_wandb':0,
        'net':'unet',
        'ldm':0,
    }
    # prepare the dataset
    print("loading data")
    train_dataset, test_dataset = gen_train_test_dataset()
    print("data loaded")
    input()
    if False:
        args['num_epochs_vae'] = 100
        #vae = AutoencoderKL(in_channels=1, out_channels=1).to(args['device'])
        vae = Vaenet().to(args['device'])
        trainvae(vae, train_dataset, test_dataset, args)
        return
    
    args['encode_initial_context'] = lambda initialcondition: F.avg_pool2d(initialcondition,kernel_size=2, stride=2)
    args['encode_data'] = lambda data: F.avg_pool2d(data, kernel_size=2, stride=2)
    args['encode_physics_context'] = lambda phy: F.interpolate(phy,size=(128,128), mode='bilinear', align_corners=False)
    args['avg_pool_2'] = lambda data: F.avg_pool2d(data, kernel_size=2, stride=2)
    if args['ldm']:
        vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-mse").to(args['device'])
        #args['encoder_function'] = lambda image: vae.encode(image).latent_dist.sample()
        # remove the encoding on ux, since we can only encode 3 channels
        # if we use vae
        normconst = 75.
        args['encode_initial_context'] = lambda initialcondition: vae.encode(initialcondition[:,1:]).latent_dist.mean/normconst
        args['encode_data'] = lambda data: vae.encode(channel_augment(data)).latent_dist.mean/normconst
        args['decode_data'] = lambda latent: torch.mean(vae.decode(latent*normconst).sample, dim=1).unsqueeze(1)
        
        def physics_encode(phy):        
            output_tensor = torch.cat([phy, phy[:,:1]], dim=1)
            output_tensor = F.interpolate(output_tensor,size=(256,256), mode='bilinear', align_corners=False)
            return vae.encode(output_tensor).latent_dist.sample()/normconst
        args['encode_physics_context'] = physics_encode
        in_channels = 12
        out_channels = 4
        input_size=32
    else:
        #in_channels = 10
        in_channels = 7
        out_channels = 1
        input_size=128
        if not args['physics_guided']:
            in_channels = 5
   
    if args['net'].lower() == 'dit_s_2':
        dit = MyDiT_S_2(input_size=input_size,in_channels=in_channels, out_channels=out_channels).to(args['device'])
        args['train_batch_size'] = 8
    elif args['net'].lower() == 'dit_b_4':
        dit = MyDiT_B_4(input_size=input_size,in_channels=in_channels, out_channels=out_channels).to(args['device'])
    else:
        dit = Unet(channels=in_channels, dim=32, out_dim=out_channels).to(args['device'])
    #dit.load_state_dict(torch.load('models/model_fluid_smoke.pth'))
    #return
    if args['use_wandb']:
        try:
            # monitor the progress through wandb
            
            wandb.init(
            # set the wandb project where this run will be logged
            project="Dit training on fluid",
            # track hyperparameters and run metadata
            config=args
            )
        except:
            print("wandb failed to initialize or connect")
        
    #train(dit,train_dataset,test_dataset,args)
    #return
    if args['use_wandb']:
        try:
            wandb.finish()
        except:
            pass
    #return
    dit.load_state_dict(torch.load('models/model_fluid_smoke_%s_%s_ldm_%s.pth'%('' if args['physics_guided'] else '_nophysique',args['net'],args['ldm'])))
    #dit.load_state_dict(torch.load('models/model_fluid_smoke__unet_200.pth'))
    dit.eval()
    #args['encode_initial_context'] = lambda initialcondition: vae.encode(initialcondition[:,1:]).latent_dist.mean/normconst
    print('model loaded')
    args['train_batch_size'] = 16
    
    eval(dit,test_dataset,args)

 
if __name__ == "__main__":
    seed = 0
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    fluid_example()

