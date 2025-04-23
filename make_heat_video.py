import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import torch.nn.functional as F
from architectures import Unet,Vaenet
from einops import rearrange
import playground.heat_fit_conductance as hf
from playground.optical_flow_fit import unet_inference
from playground.optical_flow import Optical_Flow

import torch
#from ditmodels import MyDiT_S_8, MyDiT_S_2, MyDiT_B_4, MyDiT_B_2
#from diffusion_fluid_tase import gen_train_test_dataset, full_sample
from diffusion_heat_tase import NISTDataset, genmask, heatsource_to_pde
import cv2
import os

def of_loss_small(xt_original, xt_new, physics_mask, v, of):
    xt = xt_original #rearrange(xt_folded, '(b d1) c w h -> b (d1 c) w h ', d1=time_window, c=1) 
    bs = len(xt)    

    predicted_xt = of.forward(xt_original,v) 
    
    #predicted_xt = rearrange(predicted_xt, '(b t) 1 w h -> b t w h ', t=time_window-1) 
    #physics_mask = physics < 0.001
    
    return torch.sum((physics_mask*torch.abs(xt_new-predicted_xt*0.9))**2)



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
    return heatsource
    #return [tlist, heatsource]

def phs2flow(rphysics):
    
    # first extimate the speed of laser movement
    tlist = rphysics[0]
    total_time = len(tlist)
    device = rphysics.device
    n, w, h = tlist.size()

    # Reshape the tensor to a (n, w * h) tensor
    reshaped_ts = tlist.view(n, -1)

    # Get the indices of the maximum element along the 1D tensor
    max_indices = torch.argmax(reshaped_ts, dim=1)
    yindices = max_indices // h
    xindices = max_indices % h

    dy = yindices[1:] - yindices[:-1]
    dx = xindices[1:] - xindices[:-1]

    signvx = torch.sum(dx) / (1e-10+torch.abs(torch.sum(dx)))
    vlvec = torch.sqrt(dy**2+dx**2)
    vlmin = 2
    vl = max(vlmin,torch.median(vlvec))
    vs = vl+0.01
    
    yloc,xloc = torch.meshgrid(torch.arange(32), torch.arange(64))
    yloc = yloc.to(device)
    xloc = xloc.to(device)

    flowxy = []

    prevxcenter = xindices[0]
    prevycenter = yindices[0]
    previ = 0
    for i in range(total_time-5):        

        xcenter = xindices[5+i]
        ycenter = yindices[5+i]
        print("time %s"%i)
        print(xcenter.item(),ycenter.item())
        if xcenter.item() == 0 and ycenter.item() == 0:
            print('out of bound')
            prev10x = xindices[max(0,i+5-25):min(i+5+10,total_time)]
            dx = prev10x[1:] - prev10x[:-1]
            signvx = torch.sum(dx) / (1e-10+torch.abs(torch.sum(dx)))
            
            xcenter = prevxcenter + signvx*vl*(5+i - previ)
            ycenter = prevycenter 
        else:
            print('in bound')
            previ = 5+i
            prevxcenter = xcenter
            prevycenter = ycenter
        
        
        deltay = yloc - ycenter
        
        deltax = xloc - xcenter
        #deltay = 63-deltay
        delta = torch.sqrt((deltax**2+deltay**2)*(vs**2)-(deltay**2)*(vl**2))
       
        deltat1 = (deltax*vl + delta)/(vl**2-vs**2)
        deltat2 = (deltax*vl - delta)/(vl**2-vs**2)
        deltat = deltat1*(deltat1>0) + deltat2 * (deltat2>0)
        costheta = (deltax - vl*deltat)/(vs*deltat)
        sintheta = (deltay)/(vs*deltat)
        # flowxy is a list
        # each element has dimension 32 x 64 x 2
        flowxy.append(torch.stack((vs*costheta,vs*sintheta),dim=-1))
        '''
        print(i)
        print(flowxy[-1])
        print(flowxy[-1].shape)
        print(len(flowxy))
        print((xcenter,ycenter))

        print('-'*10)
        '''
    #input()
    return torch.stack(flowxy,dim=0).unsqueeze(0)
    
def sequential_sample(unet,flownet,beta,alpha,baralpha,timelist,epoch,rdata,dinvmat,args):
    device=args['device']
    #lookback = ds.lookback
    data_time_window = 10
    
    # now data has shape [bs, timelength>10, w, h]
    #data, heatsource, rphysics_context = ds.slice_data(rdata,device)
    data = rdata
    heatsource = preprocess(rdata)
    data = data[:,0].unsqueeze(0)
    rphysics_context = heatsource_to_pde(heatsource.unsqueeze(0),dinvmat)
    # rdata has shape 500 x 1 x 32 x 64
    # data has shape 1 x 500 x 32 x 64
    # rphysics has shape 1 x 500 x 32 x 64
    # heat source has shape 500 x 32 x 64

    # calculate the predicted flow
    all_flows = []

    # 10 physics_context -> 4 flow vectors at t=5,6,7,8 (not 9)
    n_eval = (data.shape[1] - 5)//4
    for i in range(n_eval):
        predicted_flow = unet_inference(flownet,rphysics_context[:,i*4:(i*4+10)])
        # predicted flow has a shape of batch x 4 x w x h x 2
        predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> b t w h d1', d1=2) 
        all_flows.append(predicted_flow)
    all_flows = torch.cat(all_flows, dim=1)
    
    
    #print(all_flows.shape)
    #input()
    # all flows has shape [1, 492, 32, 64, 2]
    of = Optical_Flow()

    # we only predict the second half
    #physics_context = rearrange(rphysics_context[:,data_time_window//2:], 'b t w h -> (b t) 1 w h') 
    
    bs, dt, w, h = data.shape   
    
    sigma = torch.sqrt(beta)
    Time = len(alpha)
    
    xtlist = []
    # (b dt) 1 w h
    #predict_length = dt - 5
    offset = 20
    predict_length = 80
    all_flows = phs2flow(rphysics_context[:,offset:(offset+predict_length+10)])
    all_flows = torch.nan_to_num(all_flows)
    for i in range(all_flows.shape[1]):
        fli = all_flows[0,i]
        print(torch.mean(fli[:,:,0]**2+fli[:,:,1]**2))
    input()
    '''
    plt.clf()
    plt.close('all')
    nrow = 6
    ncol = 6
    fig, axes = plt.subplots(2*nrow, ncol, figsize=( 3*ncol, 3*nrow),dpi=300)
    x = np.arange(64)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)
    for i in range(nrow):
        for j in range(ncol):

            # Iterate over the tensors and plot them as greyscale/colorful images
            
            image = all_flows[0,i*nrow+j,:,:,0].detach().cpu().numpy()
            axes[2*i,j].imshow(image, cmap='afmhot')
            axes[2*i,j].axis('off')
            axes[2*i+1,j].quiver(X, Y, 
                                0.1*all_flows[0,i*nrow+j,:,:,0].detach().cpu().numpy(), 
                                0.1*all_flows[0,i*nrow+j,:,:,1].detach().cpu().numpy(), 
                                scale=10) 

            image = all_flows[0,i*nrow+j,:,:,1].detach().cpu().numpy()
            axes[2*i+1,j].imshow(image)
            axes[2*i+1,j].axis('off')
    plt.savefig('samples/aa3.png',bbox_inches='tight')
    print('done')
    input()
    '''
    x = np.arange(64)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)

    for j in tqdm(range(predict_length)):
        xt = torch.randn((bs,1,w,h), device=beta.device)
        if j % 4  == 0:
            # update init_context only evey t=4 steps
            if j  == 0:
                init_context = data[:,offset:(offset+2)]
            else:
                init_context = torch.cat(xtlist[(j-3):(j-1)],dim=1).detach()
        
        #print('Sampling in epoch %s ...'%epoch)
        strength = 0.005
        # the sub_physics_context is rearranged, so we need to transform it back
        sub_physics_context = rphysics_context[:,(offset+j+5):(offset+j+6)]
        #sub_physics_context = rearrange(rphysics_context[:,(j+5):(j+6)],'b t w h -> (b t) 1 w h')
        #sub_physics_context = torch.cat([physics_context[ii*data_time_window//2+j:ii*data_time_window//2+(j+1),:,:,:] for ii in range(bs)], dim=0)
        sub_physics_mask = genmask(sub_physics_context, 10,20).detach()
        
        for tid in range(Time-1,0,-1):
            t = timelist[tid:(tid+1)]
            if not args['physics_guided']:
                physics_context *= 0
            inputts = torch.cat((xt, sub_physics_context, init_context,) ,dim=1)
            with torch.no_grad():
                epsilon = unet(inputts,t,None)
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
                    
                    loss = of_loss_small(xtlist[-1], tsxt, sub_physics_mask, 
                                         #all_flows[:,offset+j-1].detach(), 
                                         all_flows[:,j-1].detach(),
                                         of)
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
    
        plt.clf()
        plt.close('all')
        
        fig, axes = plt.subplots(2, 2, figsize=( 3*2, 1.5*2),dpi=200)

        # Iterate over the tensors and plot them as greyscale/colorful images
            
        image = xtlist[-1][0,0].detach().cpu().numpy()
        axes[0,0].imshow(image, cmap='afmhot')
        axes[0,0].axis('off')
        axes[0,0].title.set_text('Generated sample')

        image = data[0,5+offset+j].detach().cpu().numpy()
        axes[1,0].imshow(image,cmap='afmhot')
        axes[1,0].axis('off') 
        axes[1,0].title.set_text('Ground truth')

        image = rphysics_context[0,5+offset+j].detach().cpu().numpy()
        axes[0,1].imshow(image)
        axes[0,1].axis('off') 
        axes[0,1].title.set_text('2D heat equation')
        if j > 0:
            axes[1,1].quiver(X, Y, 
                                0.1*all_flows[0,j-1,:,:,0].detach().cpu().numpy(), 
                                -0.1*all_flows[0,j-1,:,:,1].detach().cpu().numpy(), 
                                scale=10) 
        axes[1,1].imshow(image)
        axes[1,1].axis('off') 
        axes[1,1].title.set_text('Flow of spatter')

        plt.savefig('videosamples/heatframes/epoch_%s.png'%j,bbox_inches='tight')
    
    
def gen_frames(folder = 'videosamples/frames'):
    bs = 1
    #ds = NISTDataset()
    
    
    ct = 0
    vmin = -0.1#3
    vmax = 0.3
    args = {
        'T':1000,
        'beta_start':1e-4,
        'beta_end':0.02,
        'physics_guided':1,
        'device':'cuda',
        'encode_initial_context': lambda x: x,
        'encode_physics_context': lambda x: x,
        'encode_data': lambda x: x,
        'decode_data': lambda x: x,
        'device':'cuda',
    }
    args['encode_initial_context'] = lambda initialcondition: F.avg_pool2d(initialcondition,kernel_size=2, stride=2)
    args['encode_data'] = lambda data: F.avg_pool2d(data, kernel_size=2, stride=2)
    args['encode_physics_context'] = lambda phy: F.interpolate(phy,size=(128,128), mode='bilinear', align_corners=False)
    device = args['device']
    
    
    dit = Unet(channels=4, dim=64, out_dim=1).to(args['device']).to(args['device'])
    dit.load_state_dict(torch.load('models/model%s.pth'%('_with_physics' if args['physics_guided'] else '')))
    dit.eval()
    

    flownet = Unet(channels=10, dim=64, out_dim=8).to(device)
    flownet.load_state_dict(torch.load('playground/intermediate/unet_of.pt'))
    flownet.eval()
    # prepare the dataset
    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)
    rid = 0
    dinvmat =  torch.tensor([0.1622, 0.2494, 0.9865, 10.5185])

    if True:
        rdata = torch.load('data/nist_test_seq')
        rdata = rdata[0]
                                            #(unet,flownet,beta,alpha,baralpha,timelist,epoch,rdata,dinvmat,ds,args)
        sequential_sample(dit, flownet, beta_list,alpha_list,bar_alpha_list,timelist, -3, rdata,dinvmat,args)
        return
        with torch.no_grad():
            #print(odata.max())
            #print(odata.min())
            for nid in range(N):
                plt.clf()
                plt.close()

                fig, axes = plt.subplots(1, 3, figsize=(4* 4, 1 * 4))
                
                image = -odata[ct%bs,0].detach().cpu().numpy()
                axes[0].imshow(image, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[0].axis("off")
                
                image = -physics_context[ct%bs,-2].detach().cpu().numpy()
                axes[1].imshow(image, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[1].axis("off")
                
                image = -images[ct%bs,0].detach().cpu().numpy()
                axes[2].imshow(image, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[2].axis("off")
            
                
                # Save the frame to a buffer
                plt.tight_layout()
                plt.savefig(folder+"/frame_%s.png"%ct, bbox_inches='tight', pad_inches=0, dpi=100)
                ct += 1
            rid += 1
        #res_dict = ddim_sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata, ds,args)
        #sampling_with_physics(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata,args)
        #_, res_dict = full_sample(unet, beta_list,alpha_list,bar_alpha_list,timelist, epoch, tdata,args,SAVEPLOT=True)

def make_video(folder = 'videosamples/heatframes'):
    output_file = "videosamples/heatvideo01.avi"
    fps = 8  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = 256 *2
    frame_height = 256
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    all_files = os.listdir(folder)
    #print(all_files)
    #input()
    for filename in all_files:
        fullname = os.path.join(folder,filename)

        # Read the saved frame and convert it to BGR for OpenCV
        frame_bgr = cv2.imread(fullname)
        
        _,w,_ = frame_bgr.shape
        #frame_bgr = frame_bgr[:,:w//2]

        # Resize the frame to match the video dimensions
        frame_bgr = cv2.resize(frame_bgr, (frame_width, frame_height))

        # Write the frame to the video
        video_writer.write(frame_bgr)

    # Release the video writer
    video_writer.release()

if __name__ == "__main__":
    folder = 'videosamples/heatframes'
    #gen_frames(folder=folder)
    make_video(folder=folder)