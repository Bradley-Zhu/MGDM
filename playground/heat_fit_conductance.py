import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import torch.optim as optim
import time
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import sys
 
# setting path
sys.path.append('../')
from slice_video import slice_all_and_save

def update(tcur,hs,khor,kver,loss,cutoff):
    with torch.no_grad():
        right2left = tcur[:,1:] - tcur[:,:-1]
        left2right =  tcur[:,:-1] - tcur[:,1:]
        heatin = torch.zeros(tcur.shape, device=tcur.device)
        heatin[:,:-1] = heatin[:,:-1] + khor*(right2left)
        heatin[:,1:] = heatin[:,1:] + khor*(left2right)

        up2down = tcur[1:,:] - tcur[:-1,:]
        down2up = -up2down
        heatin[1:,:] = heatin[1:,:] + kver*down2up
        heatin[:-1,:] = heatin[:-1,:] + kver*up2down
        heatin[tcur<cutoff] *= 0.1
        tnew = tcur * loss + heatin + hs
        return tnew

def plotall(tlist,hslist):
    plt.clf()
    plt.close('all')
    N = 20
    fig, axes = plt.subplots(2, N, figsize=(N * 4, 2 * 4))
    delta = len(hslist)//N
    for i in range(N):
        #print(i*delta)
        axes[0,i].imshow(tlist[i*delta])
        axes[0,i].axis('off')

        axes[1,i].imshow(hslist[i*delta])
        axes[1,i].axis('off')
    plt.savefig('simulations/simulations1.png',bbox_inches="tight")


def simulation1():
    nx = 100
    ny = 100
    temp = torch.zeros(nx,ny)
    tlist = []
    hslist = []
    kver = 0.1
    khor = 0.4
    episode = 1000
    speed = 0.05
    starty = 50
    startx = 2
    peaktemp = 1
    loss = 0.99
    cutoff = 0.4
    
    for ti in range(episode):
        hsmat = torch.zeros(nx,ny)
        hsx = startx + speed * ti
        hsxl = math.floor(hsx)
        hsxu = math.ceil(hsx)
        if hsxl-hsxu == 0:
            hsmat[(starty-0):(starty+1),hsxl] += peaktemp
        else:
            hsmat[(starty-0):(starty+1),hsxl] += (hsxu - hsx) * peaktemp 
            hsmat[(starty-0):(starty+1),hsxu] += (hsx - hsxl) * peaktemp 
        if ti%100 == 0:
            tlist.append(temp)
            hslist.append(hsmat)
            
        temp = update(temp,hsmat,khor,kver,loss,cutoff)
    plotall(tlist,hslist)

def preprocess(one_batch):
    tlist = one_batch[:,0]
    n, w, h = tlist.size()

    # Reshape the tensor to a (n, w * h) tensor
    reshaped_ts = tlist.view(n, -1)

    # Get the indices of the maximum element along the 1D tensor
    max_indices = torch.argmax(reshaped_ts, dim=1)
    yindices = max_indices // h
    xindices = max_indices % h

    heatsource = tlist*0
    heatsource[torch.arange(len(yindices)), yindices, xindices] = tlist[torch.arange(len(yindices)), yindices, xindices]
    heatsource[heatsource < 0.8] = 0
    heatsource[heatsource > 0.8] = 0.88
    #plotall(tlist,heatsource)

    return [tlist, heatsource]

def gaussian_kernel(s1, s2, raw_width, raw_height, device):
    width = raw_width // 4
    height = raw_height // 4
    
    x = torch.arange(2*width+1,device=device).view(1, -1)
    y = torch.arange(2*height+1,device=device).view(-1, 1)
    center_x = width 
    center_y = height 
    kernel = torch.exp(-((x - center_x)**2 * s1 + (y - center_y)**2 * s2))
    return kernel / kernel.sum()

def greensfunction(input_tensor, s1,s2,t):
    if len(input_tensor.shape) == 2:
        H, W = input_tensor.shape
    else:
        bs, H, W = input_tensor.shape
    kernel = gaussian_kernel(s1/t, s2/t, W, H,input_tensor.device).unsqueeze(0).unsqueeze(0)
    # Apply 2D convolution with the Gaussian kernel
    #output = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding=(W // 2, H // 2))[0,0]
    if len(input_tensor.shape) == 2:
        output = F.conv2d(input_tensor.unsqueeze(0).unsqueeze(0), kernel, padding='same')[0,0]
    else:
        output = F.conv2d(input_tensor.view(bs,1,H,W), kernel, padding='same')[:,0]
    return output

def ts_repeat(ts, repeat):
    # Expand the dimensions of ts to (b, t, c, w, h)
    expanded_ts = ts.unsqueeze(1).expand(-1, repeat, -1, -1, -1)

    # Reshape the expanded tensor to (b*t, c, w, h)
    return expanded_ts.reshape(-1, ts.size(1), ts.size(2), ts.size(3))

def find_conductance(called_from_parent=False, REGEN = False):
    if REGEN:
        if called_from_parent:
            video_path = "data/LBMAM_raw"
        else:
            video_path = "../data/LBMAM_raw"
        alldata = slice_all_and_save(video_path,10,[],onlyeven=True)
        
        statedict = {'alldata':alldata}
        if called_from_parent:
            torch.save(statedict,'playground/intermediate/alldata.pt')
        else:
            torch.save(statedict,'intermediate/alldata.pt')
    else:
        if called_from_parent:
            statedict = torch.load('playground/intermediate/alldata.pt')
        else:
            statedict = torch.load('intermediate/alldata.pt')
        alldata = statedict['alldata']
        print('data loaded with shape:')
        print(len(alldata), alldata[0].shape)
    #return alldata
    norms = []
    normthreshold = 200
    fulldata = []
    for i,bt in enumerate(alldata):
        if torch.sum(bt).item() > normthreshold:
            fulldata.append(preprocess(alldata[i]))
    print('After filtering at threshold %s, %s batches are created!'%(normthreshold, len(fulldata)))
    ylist = []
    xlist = []
    trainds = int(len(fulldata)*0.8)
    device = 'cuda'
    lookback = 5
    RETRAIN = False
    if RETRAIN:
        dinvmat = Parameter(torch.ones(4,device=device)*0.5)
        opt = optim.AdamW([dinvmat],lr=0.01,weight_decay=1e-4)
        epochs = 10
        for n in range(epochs):
            totloss = 0
            totbatch = 0
            for data in fulldata[:trainds]:
                tlist, heatsource, horheatin, verheatin = data[0].to(device), data[1].to(device), data[2], data[3]
                
                dt = len(tlist)
                
                
                for i in range(lookback,dt):
                    predict = greensfunction(0*tlist[i-lookback], dinvmat[0], dinvmat[1], 1+lookback)
                    for j in range(0,lookback):
                        predict += dinvmat[3]* torch.sigmoid(dinvmat[2])**j * greensfunction(heatsource[i-j], dinvmat[0], dinvmat[1], 1 + j)
                    
                loss = torch.mean((predict - tlist[i])**2)

                opt.zero_grad()
                loss.backward()
                opt.step()

                totloss += loss
                totbatch += 1
            print("epoch %s, loss : %.6f, dinv1: %.4f, dinv2: %.4f, dinv3: %.4f , dinv4: %.4f"%(n,totloss/totbatch, dinvmat[0].item(), dinvmat[1].item(), torch.sigmoid(dinvmat[2]).item(), dinvmat[3].item()))  
        dinvmat = dinvmat.detach().cpu()
    else:
        dinvmat = torch.tensor([0.1622, 0.2494, 0.9865, 10.5185])  
    return fulldata, dinvmat

def gen_test_data(called_from_parent=False, REGEN = False):
    if REGEN:
        if called_from_parent:
            video_path = "data/LBMAM_raw"
        else:
            video_path = "../data/LBMAM_raw"
        alldata = slice_all_and_save(video_path,10,[],onlyeven=True)
        
        statedict = {'alldata':alldata}
        if called_from_parent:
            torch.save(statedict,'playground/intermediate/alldata.pt')
        else:
            torch.save(statedict,'intermediate/alldata.pt')
    else:
        if called_from_parent:
            statedict = torch.load('playground/intermediate/alldata.pt')
        else:
            statedict = torch.load('intermediate/alldata.pt')
        alldata = statedict['alldata']
        print('data loaded with shape:')
        print(len(alldata), alldata[0].shape)
    #return alldata
    norms = []
    normthreshold = 200
    fulldata = []
    for i,bt in enumerate(alldata):
        if torch.sum(bt).item() > normthreshold:
            fulldata.append(preprocess(alldata[i]))
    print('After filtering at threshold %s, %s batches are created!'%(normthreshold, len(fulldata)))
    ylist = []
    xlist = []
    trainds = int(len(fulldata)*0.8)
    device = 'cuda'
    lookback = 5
    dinvmat =  torch.tensor([0.1622, 0.2494, 0.9865, 10.5185])
    return fulldata, dinvmat

def regression():
    fulldata, dinvmat = find_conductance()
    predicted = []
    trainds = int(len(fulldata)*0.8)
    lookback = 5
    for data in fulldata[trainds:]:
        tlist, heatsource, horheatin, verheatin = data[0], data[1], data[2], data[3]
                
        dt = len(tlist)
        for i in range(lookback):
            predicted.append(tlist[i])     
        for i in range(lookback,dt):
            predict = greensfunction(tlist[i-lookback]*0, dinvmat[0], dinvmat[1], 1+lookback)
                    #print(predict.shape)
                    #assert False
            for j in range(0,lookback):
                predict += dinvmat[3]* torch.sigmoid(dinvmat[2])**j * greensfunction(heatsource[i-j], dinvmat[0], dinvmat[1], 1 + j)
            predicted.append(predict)
        break
    plotall(tlist[:-1],predicted)
    #print(theta)
    #plt.hist(np.array(norms))
    #plt.savefig('norms.png',bbox_inches='tight')  
    #preprocess(alldata[201])

if __name__=="__main__":
    #simulation1()
    regression()