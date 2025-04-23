import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.data import Dataset, DataLoader
from multiprocessing import cpu_count
import torch.nn.functional as F
from architectures import Unet,Vaenet
from einops import rearrange

import torch
from diffusion_fluid import gen_train_test_dataset,expensive_physics_loss,plot_bouyancy,mse_psnr_ssim
import cv2
import os

def sample_expensive_physics_loss(cs2, x, wall=None):
    if wall is None:
        pooled = cs2-F.avg_pool2d((x), kernel_size=(2, 2), stride=(2, 2))
    else:
        pooled = cs2-F.avg_pool2d((x)*wall[:,None,:,:], kernel_size=(4, 4), stride=(4, 4))
    return torch.sum(pooled**2,dim=[1,2,3])


def full_sample(unet,beta,alpha,baralpha,timelist,epoch,rdata,args,SAVEPLOT = False):
    device=args['device']
    
    with torch.no_grad():
        initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
        initial_context = args['encode_initial_context'](initial_context)
        physics_context = args['encode_physics_context'](c1_context)
        data = args['encode_data'](odata)  

        rpt = 2
        
        initial_context = initial_context.repeat(rpt,1,1,1)
        physics_context = physics_context.repeat(rpt,1,1,1)
        data = data.repeat(rpt,1,1,1)
        phs_time = phs_time.repeat(rpt,1)
        c2_context = c2_context.repeat(rpt,1,1,1)
        
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
            epsilon = unet(inputts,t.repeat(len(phs_time)) ,phs_time[:,0])
            
            x0 = 1/torch.sqrt(baralpha[t])*(xt - torch.sqrt(1-baralpha[t])*epsilon)
            strength = 0.01
        for zz in range(1):
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
   
    trajectory = trajectory+[args['decode_data'](x0)]
    if SAVEPLOT:
        savename = 'samples/double_physics_fluid/fluid_vortex_epoch_%s.png'%epoch
        
        plot_bouyancy(trajectory, c1_context, c2_context, odata,savename)
    
    reconsdata = trajectory[-1]
   
    sloss = sample_expensive_physics_loss(c2_context.detach(),reconsdata)
    sloss = rearrange(sloss, '(r b)  -> r b', r=rpt) 
    minlossid = torch.argmin(sloss,dim=0)
    minlossid = bs//rpt * minlossid + torch.arange(bs//rpt,device=sloss.device)
    # [rpt 0, rpt 0, rpt 1, ... rpt 0]
    reconsdata = reconsdata[minlossid]
 
    with torch.no_grad():
        if args['ldm']:
            reconsdata = args['avg_pool_2'](reconsdata)
            bs = 2
            nbs = len(reconsdata)
            reconsgt = [args['avg_pool_2'](args['decode_data'](data[bid*bs:(bid+1)*bs])) for bid in range(nbs//bs)]
            reconsgt = torch.cat(reconsgt,dim=0)
        else:
            reconsgt = args['decode_data'](data)
    return reconsdata, None

def gen_frames(folder = 'videosamples/frames'):
    bs = 16
    trainds,testds = gen_train_test_dataset(intermediate_path='fluid_data_gen/test_dataset',train_ratio=1)
    testdl = DataLoader(trainds, batch_size = bs, shuffle = False, pin_memory = True, num_workers = cpu_count())
    

    
    ct = 0
    vmin = -0.1#3
    vmax = 0.3
    args = {
        'T':1000,
        'beta_start':1e-4,
        'beta_end':0.02,
        'physics_guided':1,
        'ldm':0,
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
    in_channels = 7
    out_channels = 1
    input_size=128
    dit = Unet(channels=in_channels, dim=32, out_dim=out_channels).to(args['device'])

    #dit = MyDiT_S_2(input_size=input_size,in_channels=in_channels, out_channels=out_channels).to(args['device'])
    
    dit.load_state_dict(torch.load('models/model_fluid_smoke__unet_200.pth'))
    dit.eval()
    
    # prepare the dataset
    Time = args['T']
    beta_list = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=device)
    timelist = torch.arange(Time,device=device)
    alpha_list = 1 - beta_list
    bar_alpha_list = torch.cumprod(alpha_list,dim=0)

    for rdata in tqdm(testdl):
        with torch.no_grad():                
            ophysics_context, odata, phs_time = rdata[1].to(device), rdata[3].to(device), rdata[4].to(device)
            N = len(odata)
        images, res_dict = full_sample(dit, beta_list,alpha_list,bar_alpha_list,timelist, -3, rdata,args,SAVEPLOT=False)
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
                axes[0].set_title('Ground truth',fontsize=20)

                
                image = -ophysics_context[ct%bs,-2].detach().cpu().numpy()
                axes[1].imshow(image, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[1].axis("off")
                axes[1].set_title('Simulation',fontsize=20)
                
                image = -images[ct%bs,0].detach().cpu().numpy()
                axes[2].imshow(image, cmap='coolwarm', vmin=vmin, vmax=vmax)
                axes[2].axis("off")
                axes[2].set_title('Generated sample',fontsize=20)
            
                
                # Save the frame to a buffer
                plt.tight_layout()
                plt.savefig(folder+"/frame_%s.png"%ct, bbox_inches='tight', pad_inches=0, dpi=100)
                ct += 1
       
def make_video(folder = 'videosamples/frames'):
    output_file = "videosamples/fluid01.avi"
    fps = 8  # Frames per second
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    frame_width = 256 * 4
    frame_height = 256
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (frame_width, frame_height))

    all_files = os.listdir(folder)
    #print(all_files)
    #input()
    for filename in all_files:
        fullname = os.path.join(folder,filename)

        # Read the saved frame and convert it to BGR for OpenCV
        frame_bgr = cv2.imread(fullname)

        # Resize the frame to match the video dimensions
        frame_bgr = cv2.resize(frame_bgr, (frame_width, frame_height))

        # Write the frame to the video
        video_writer.write(frame_bgr)

    # Release the video writer
    video_writer.release()

if __name__ == "__main__":
    folder = 'videosamples/frames'
    gen_frames(folder=folder)
    make_video(folder=folder)