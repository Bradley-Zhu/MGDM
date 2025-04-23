import random
import numpy as np
import torch
import torch.optim as optim
import time
import torch.nn.functional as F

from torch.nn.parameter import Parameter
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW
import matplotlib.pyplot as plt
from multiprocessing import cpu_count

from einops import rearrange, reduce, repeat
from tqdm import tqdm

#import heat_fit_conductance as hf
from .heat_fit_conductance import *
from .optical_flow import Optical_Flow

import sys
sys.path.append("..")
from architectures import Unet


def unet_inference(unet,x,phs_t=None):
    n,c,h,w = x.shape
    if phs_t is None:
        t_list = torch.zeros(n,device=x.device)
    else:
        t_list = phs_t
    return unet(x, t_list, t_list)


def gen_of_gt():
    device = 'cuda'
    regen = False
    fulldata, dinvmat = find_conductance(False, True)#hf.
    # full data is a list
    # each element is another list [10 x 32 x 64, 10 x 32 x 64]
    # first one is the actual data
    # second one is the heat source
    
    # train the optical flow
    of = Optical_Flow()
    start = 5
    print('start calculating optical flow using ad-hoc methods, this can take some time')
    for i in tqdm(range(len(fulldata))):
        data_batch_i = fulldata[i][0]
        data_needed = data_batch_i[start:]
        data_needed = data_needed.to(device).unsqueeze(1)
        # this information is never used in prediction
        flow_i = of.get_of(data_needed[:-1],data_needed[1:], verbose=0)
        fulldata[i].append(flow_i.detach().cpu())
    print('done calculating optical flow using ad-hoc methods')
    torch.save({'augmenteddata':fulldata},'intermediate/alldata_withof.pt')
    return

def middle_result_check(solution, flow):
    b,n,w,h = solution.shape
    meaningful_flow = flow.reshape(b,-1,2,w,h)
    
    x = np.arange(64)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)

    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(2, n//2, figsize=(4 * n//2, 2 * 2))
    for i in range(n//2):
        axes[0,i].imshow(solution[0][i+n//2].detach().cpu().numpy())
        axes[0,i].axis('off')

        axes[1,i].imshow(solution[0][i+n//2].detach().cpu().numpy())
        axes[1,i].axis('off')
        if i < n//2-1:
            axes[1,i].quiver(X, Y, 
                             meaningful_flow[0,i,0].detach().cpu().numpy(), 
                             meaningful_flow[0,i,1].detach().cpu().numpy(), 
                             scale=10)
        
    plt.savefig('simulations/flow_check.png',bbox_inches='tight')

def heatsource_to_pde(heatsource, dinvmat, lookback=5):
    physics_context = []
    b,dt,w,h = heatsource.shape
    for i in range(0,dt):                
        for j in range(0,lookback):
            if i - j < 0:
                break
            if j == 0:
                predict = dinvmat[3]* torch.sigmoid(dinvmat[2])**j * greensfunction(heatsource[:,i-j], dinvmat[0], dinvmat[1], 1 + j)#hf.
            else:
                predict += dinvmat[3]* torch.sigmoid(dinvmat[2])**j * greensfunction(heatsource[:,i-j], dinvmat[0], dinvmat[1], 1 + j)#hf.
                    #print(predict.shape)
                    #return
        # predict has shape batch x w x h
        physics_context.append(predict)
    # physics_context has shape batch x ( length x w x h )
    physics_context = torch.stack(physics_context,dim=1)#.unsqueeze(1)
    return physics_context

def train():
    statedict = torch.load('intermediate/alldata_withof.pt')

    fulldata = statedict['augmenteddata']
    _, dinvmat = find_conductance(False, False)#hf.

    print(len(fulldata))
    print(len(fulldata[0]))
    print(fulldata[0][2].shape)

    device = 'cuda'
    unet = Unet(channels=10, dim=64, out_dim=8).to(device)
    n,w,h = fulldata[0][0].shape    
    
    combineddata = [torch.cat((dt[1],dt[2].permute(0, 3, 1, 2).reshape(-1,w,h)), dim=0) for dt in fulldata]
    trainlen = int(len(combineddata)*0.9)
    random.shuffle(combineddata)
    traindl = DataLoader(combineddata[:trainlen], batch_size = 8, shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(combineddata[trainlen:], batch_size = 8, shuffle = True, pin_memory = True, num_workers = cpu_count())
    epochs = 15
    opt = AdamW(unet.parameters(), lr = 0.0001, weight_decay=5e-6)

    for epid in range(epochs):
        totloss = 0
        totsample = 0

        testloss = 0
        testsample = 0
        for data in traindl:

            heatsource = data[:,:n].to(device)
            physics_context = heatsource_to_pde(heatsource, dinvmat.detach())
            dtoutput = data[:,n:].to(device)            
            prediction = unet_inference(unet, physics_context)
            loss = torch.mean(torch.sum((dtoutput-prediction)**2))
            
            opt.zero_grad()
            loss.backward()
            opt.step()

            totloss += loss.item()*len(data)
            totsample += len(data)
        for data in testdl:
            with torch.no_grad():
                heatsource = data[:,:n].to(device)
                physics_context = heatsource_to_pde(heatsource, dinvmat.detach())
                dtoutput = data[:,n:].to(device)            
                prediction = unet_inference(unet, physics_context)
                loss = torch.mean(torch.sum((dtoutput-prediction)**2))
                testloss += loss.item()*len(data)
                testsample += len(data)
        middle_result_check(physics_context, prediction)
        torch.save(unet.state_dict(), 'intermediate/unet_of.pt')
        print('[%s/%s]: optical flow avg loss is %.4f, test loss is %.4f'%(epid, epochs, totloss/totsample, testloss/testsample))
            

if __name__ == "__main__":
    gen_of_gt()
    train()
    
