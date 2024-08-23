import random
import matplotlib.pyplot as plt
import numpy as np
import scipy
import torch
from os import listdir
from os.path import isfile, join
import torchvision
import torchvision.transforms as T
from torchvision.transforms import v2
from tqdm.auto import tqdm

import cv2
#size=(64,128)
size=(32,64)
transform = torchvision.transforms.Compose([
                               T.Resize(size=size),
                               torchvision.transforms.Normalize(
                                 (0), (1180))
                                ])

def sclice_one(name,length):
    data = scipy.io.loadmat(name)
    temp = data['Layer']['RadiantTemp'][0][0]
    all_clips = []
    for i in range(temp.shape[2]):
        if i%length == 0:
            new_clip = []
        new_clip.append(torch.tensor(temp[:,:,i]+0.,dtype=torch.float))
        if i%length == length-1:
            new_clip = torch.stack(new_clip)
            if new_clip.sum()<1e6:
                continue
            all_clips.append(transform(new_clip.unsqueeze(1)))
    #print(max([torch.max(mi).item() for mi in all_clips]))
    #assert False
    return all_clips

def slice_all_and_save(folder, length, size, onlyeven = False):
    matfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and '.mat' in f]
    if onlyeven:
        matfiles = [f for f in matfiles if int(f[-5]) in [1,3,5,7,9]]
    add_data = []
    id = 0
    for file in matfiles:
        sliced_clips = sclice_one(join(folder, file),length)
        add_data = add_data + sliced_clips
        if id >= len(matfiles) - 5:
            break
        id += 1
    print('-'*20)
    print('%s clips are created'%len(add_data))
    print('one sample has the shape of:')
    print(add_data[0].shape)
    #torch.save({'cliced_clips':add_data}, 'data/LBMAM_sliced/data.pkl')
    return add_data


def sampling(unet,context,gblurer,beta,alpha,baralpha,timelist,epoch,args, num_sample=10,save=True):
    
    sigma = torch.sqrt(beta)
    bshape = (num_sample,1, context.shape[3], context.shape[4])
    Time = len(alpha)
    
    time_span = 200
    xt = torch.randn(bshape, device=beta.device)

    dissipate_frame = gblurer(args['conduction']*context)
    heat_source = context[:,1:] - dissipate_frame[:,:-1]
    #inputdata = torch.cat((inputdata, heat_source[:,:-1,0,:,:], data[:,-2,:,:,:], dissipate_last_frame[:,-2,:,:,:]),dim=1)
            
    with torch.no_grad():
        trajectory = [xt]
        
        #for tid in tqdm(range(Time-1,-1,-1)):
        for tid in range(Time-1,-1,-1):
            t = timelist[tid:(tid+1)]
            inputdata = torch.cat(( dissipate_frame[:,-2]+xt, heat_source[:,:-1,0,:,:], context[:,-2,:,:,:], dissipate_frame[:,-2,:,:,:]),dim=1)
            xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputdata,t))
            xt += sigma[t] * torch.randn(bshape, device=beta.device)
            if t%time_span == 0:
                trajectory.append(xt)
        trajectory.append(xt)
    outputframe = dissipate_frame[:,-2]+xt
    if save:
        print('Sampling in epoch %s ...'%epoch)
        plt.clf()
        plt.close('all')
        fig, axes = plt.subplots(num_sample, context.shape[1]+1 + len(trajectory), figsize=(4*num_sample,4*(context.shape[1]+1 + len(trajectory))))

        for j in range(num_sample):
            for i in range(args['length']):
                image = context[j,i,0,:,:].detach().cpu().numpy()
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')

            image = outputframe[j,0,:,:].detach().cpu().numpy()
            axes[j,args['length']].imshow(image, cmap='gray')
            axes[j,args['length']].axis('off')
            for id in range(len(trajectory)):
                i = id + args['length']+1
                # grey-scale image
                image = trajectory[len(trajectory)-1-id][j,0,:,:].detach().cpu().numpy()
                # Plot the image in subplot i
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')
        plt.savefig('samples/epoch_%s.png'%epoch,bbox_inches='tight')
    return outputframe

def sampling_nophysics(unet,context,gblurer,beta,alpha,baralpha,timelist,epoch,args, num_sample=10,save=True):
    
    sigma = torch.sqrt(beta)
    bshape = (num_sample,1, context.shape[3], context.shape[4])
    Time = len(alpha)
    
    time_span = 200
    xt = torch.randn(bshape, device=beta.device)

    #inputdata = torch.cat((inputdata, heat_source[:,:-1,0,:,:], data[:,-2,:,:,:], dissipate_last_frame[:,-2,:,:,:]),dim=1)
            
    with torch.no_grad():
        trajectory = [xt]
        
        #for tid in tqdm(range(Time-1,-1,-1)):
        for tid in range(Time-1,-1,-1):
            t = timelist[tid:(tid+1)]
            inputdata = torch.cat((xt, context[:,:-1,0]),dim=1)
            xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputdata,t))
            xt += sigma[t] * torch.randn(bshape, device=beta.device)
            if t%time_span == 0:
                trajectory.append(xt)
        trajectory.append(xt)
    outputframe = xt
    if save:
        print('Sampling in epoch %s ...'%epoch)
        plt.clf()
        plt.close('all')
        fig, axes = plt.subplots(num_sample, context.shape[1]+1 + len(trajectory), figsize=(4*num_sample,4*(context.shape[1]+1 + len(trajectory))))

        for j in range(num_sample):
            for i in range(args['length']):
                image = context[j,i,0,:,:].detach().cpu().numpy()
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')

            image = outputframe[j,0,:,:].detach().cpu().numpy()
            axes[j,args['length']].imshow(image, cmap='gray')
            axes[j,args['length']].axis('off')
            for id in range(len(trajectory)):
                i = id + args['length']+1
                
                # grey-scale image
                image = trajectory[len(trajectory)-1-id][j,0,:,:].detach().cpu().numpy()
                # Plot the image in subplot i
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')
        plt.savefig('samples/epoch_%s.png'%epoch,bbox_inches='tight')
    return outputframe

def conditional_joint_sampling_img(unet,context,rinvsq,beta,alpha,baralpha,timelist,epoch, num_sample=10,args={},save=True):
    sigma = torch.sqrt(beta)
    bshape = (num_sample,context.shape[1], context.shape[3], context.shape[4])
    Time = len(alpha)
    #print('-'*20)
    #print(context.shape)
    #print(bshape)
    
    time_span = 200
    xt = torch.randn(bshape, device=beta.device)

    #inputdata = torch.cat((inputdata, heat_source[:,:-1,0,:,:], data[:,-2,:,:,:], dissipate_last_frame[:,-2,:,:,:]),dim=1)
            
    with torch.no_grad():
        trajectory = [xt[:,-1,:,:]]
        
        #for tid in tqdm(range(Time-1,-1,-1)):
        for tid in range(Time-1,-1,-1):
            t = timelist[tid:(tid+1)]
            inputdata = xt #torch.cat((xt, context[:,:-1,0]),dim=1)
            xt = 1/torch.sqrt(alpha[t])*(xt - (1-alpha[t])/(torch.sqrt(1-baralpha[t]))*unet(inputdata,t))
            #print('-'*20)
            #print(xt.shape)
            #print(context.shape)
            #assert False
            xt += sigma[t] * torch.randn(bshape, device=beta.device)
            xt[:,:-1] = xt[:,:-1] + rinvsq[t]*(context[:num_sample,:-1,0]-xt[:,:-1])

            
            if t%time_span == 0:
                trajectory.append(xt[:,-1,:,:])
        trajectory.append(xt[:,-1,:,:])
    
    outputframe = xt[:,-1,:,:]
    if save:
        print('Sampling in epoch %s ...'%epoch)
        plt.clf()
        plt.close('all')
        fig, axes = plt.subplots(num_sample, context.shape[1]+1 + len(trajectory), figsize=(4*num_sample,4*(context.shape[1]+1 + len(trajectory))))

        for j in range(num_sample):
            for i in range(args['length']):
                image = context[j,i,0,:,:].detach().cpu().numpy()
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')

            image = outputframe[j,:,:].detach().cpu().numpy()
            axes[j,args['length']].imshow(image, cmap='gray')
            axes[j,args['length']].axis('off')
            for id in range(len(trajectory)):
                i = id + args['length']+1
                
                # grey-scale image
                
                image = trajectory[len(trajectory)-1-id][j,:,:].detach().cpu().numpy()
                
                # Plot the image in subplot i
                axes[j,i].imshow(image, cmap='gray')
                axes[j,i].axis('off')
        plt.savefig('samples/conditional/epoch_%s.png'%epoch,bbox_inches='tight')
    return outputframe.unsqueeze(1)

def sample_video(folder ,args, unet):
    matfiles = [f for f in listdir(folder) if isfile(join(folder, f)) and '.mat' in f]
    add_data = []
    
    #name = random.choice(matfiles)
    name = matfiles[-2]
    data = scipy.io.loadmat(join(folder, name))
    temp = data['Layer']['RadiantTemp'][0][0]
    
    #startid = random.choice(list(range(temp.shape[2]-10*length)))
    startid = 25*42+7
    new_clip = []

    tot_frames = 100
    n_sample = 10
    for i in range(startid, startid+tot_frames):     
        new_clip.append(torch.tensor(temp[:,:,i]+0.,dtype=torch.float, device=args['device']))
    
    # context has dimension 1 x 5 x 1 x size
    ground_truth = transform(torch.stack(new_clip).unsqueeze(1)).unsqueeze(0)
    print(ground_truth.shape)
    return
    context = ground_truth[:,:args['length']].repeat(n_sample,1,1,1,1)
    
    gblurer = v2.GaussianBlur(kernel_size=(7,7), sigma=(2.,2.))
  
    Time = args['T']
    beta = torch.linspace(args['beta_start'], args['beta_end'], steps=Time,device=args['device'])
    timelist = torch.arange(Time,device=args['device'])
    rinvsq = torch.linspace(0.3, 1e-4, steps=Time,device=args['device'])
    #print(beta_list)
    #print(timelist)
    #assert False
    print(len(beta))
    alpha = 1 - beta
    baralpha = torch.cumprod(alpha,dim=0)
    for i in tqdm(range(tot_frames-5)):
        context_ending = context[:,-args['length']:]
        extendedcontext = torch.cat((context_ending, torch.zeros(context_ending.shape, device=context_ending.device)), dim=1)
            
        if args['physics_guided']:
            
            out = sampling(unet,extendedcontext[:,1:1+args['length']],gblurer,beta,alpha,baralpha,timelist,0,args, save=False, num_sample=n_sample)

        elif args['guided']:
            
            out = conditional_joint_sampling_img(unet,extendedcontext[:,1:1+args['length']],rinvsq,beta,alpha,baralpha,timelist,epoch=0,args=args, save=False, num_sample=n_sample)
                                                #(unet,context,rinvsq,beta,alpha,baralpha,timelist,epoch, num_sample=10,args={},save=True)
        else:
            out =  sampling_nophysics(unet,extendedcontext[:,1:1+args['length']],gblurer,beta,alpha,baralpha,timelist,0,args, save=False, num_sample=n_sample)
        out[out<0] = 0
        #print('-=-'*10)
        #print(context.shape)
        #print(out.unsqueeze(1).unsqueeze(2).shape)
        context = torch.cat((context,out.unsqueeze(1)),dim=1)
    save_path = 'video_samples/sample%s.mp4'%args['physics_guided'] 
    just_show_img(ground_truth[0,:,0],context[:,:,0],'video_samples/sample%s.png'%args['physics_guided'])
    create_video(context[0,:,0], save_path)


  
def just_show_img(gt_img_ts, img_ts,save_path):
    N = len(gt_img_ts)
    print('number of images')
    print(N)
    n_sample = len(img_ts)
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(n_sample+1,N, figsize=(4*N,1.8*(1+n_sample)))

    for j in range(N):
        image = gt_img_ts[j].detach().cpu().numpy()
        axes[0,j].imshow(image, cmap='gray')
        axes[0,j].axis('off')
        for k in range(n_sample):
            image = img_ts[k,j].detach().cpu().numpy()
            axes[1+k,j].imshow(image, cmap='gray')
            axes[1+k,j].axis('off')
    plt.savefig(save_path, bbox_inches='tight')

def create_video(frames_tensor, save_path):
    # Get the dimensions of the tensor
    n, w, h = frames_tensor.shape

    # Create a VideoWriter object to save the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(save_path, fourcc, fps=2, frameSize=(h, w))

    # Convert the tensor to numpy array
    frames_array = frames_tensor.cpu().numpy()
    frames_array[frames_array>1] = 1
    frames_array[frames_array<0] = 0
    # Iterate over each frame and write it to the video
    for i in range(n):
        frame = np.uint8(frames_array[i]*255)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # Convert grayscale to BGR if necessary
        video.write(frame)

    # Release the VideoWriter object
    video.release()

def create_video_for_presentation():
    folder = "data/LBMAM_raw"
    args = {
        'num_epochs': 51,
        'length':5,
        'train_lr': 1e-4,
        'train_batch_size': 8,
        'T':1000,        
        'beta_start':1e-4,
        'beta_end':0.02,
        'physics_guided':1,
        'device':'cuda',
    }
    args['data_time_window'] = 10
    from architectures import Unet    
   
    unet = Unet(channels=4, dim=64, out_dim=1).to(args['device'])
   
    
    unet.load_state_dict(torch.load('models/model%s.pth'%('_with_physics' if args['physics_guided'] else '')))
    unet.eval()
    sample_video(folder, args, unet)

if __name__=="__main__":
    #slice_all_and_save('data/LBMAM_raw',5)
    create_video_for_presentation()
       
