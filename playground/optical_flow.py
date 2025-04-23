import numpy as np
import torch
import torch.optim as optim
import time
import torch.nn.functional as F

from torch.nn.parameter import Parameter
import matplotlib.pyplot as plt

import numpy as np
from torchvision.models.optical_flow import raft_large
from einops import rearrange, reduce, repeat


from .heat_fit_conductance import find_conductance

# everything is written by ChatGPT
# seems to work
class Optical_Flow():

    def __init__(self, device='cuda'):
        #self.model = raft_large(pretrained=True, progress=False).to(device)
        #self.model = self.model.eval()
        self.device = device

    def get_of(self, img1_batch_raw, img2_batch_raw, verbose=0):
        return parametric_of(img1_batch_raw, img2_batch_raw, img1_batch_raw.device, verbose=verbose)
        if img1_batch_raw.shape[1] == 1:
            n, _, w, h = img2_batch_raw.shape
            
            img1_batch = img1_batch_raw.expand(n, 3, w, h)
            img2_batch = img2_batch_raw.expand(n, 3, w, h)
        else:
            img1_batch = img1_batch_raw
            img2_batch = img2_batch_raw
        mindim = min(img1_batch.shape[2:])
        if mindim < 128:
            expanddim = int(128/mindim)
            n,c, oldw, oldh = img1_batch.shape
            neww, newh = oldw*expanddim, oldh*expanddim
            new_size = (neww,newh)
            img1_batch = F.interpolate(img1_batch, size=new_size, mode='bilinear', antialias=True, align_corners=False)
            img2_batch = F.interpolate(img2_batch, size=new_size, mode='bilinear', antialias=True, align_corners=False)

        list_of_flows =  raft_of(img1_batch, img2_batch, self.model, self.device)
        flows = list_of_flows[-1]
        if mindim < 128:
            new_size = (oldw,oldh)
            flows = F.interpolate(flows, size=new_size, mode='bilinear', antialias=True, align_corners=False)
        # flows has dimension batch x 2 x w x h
        return flows.permute(0, 2, 3, 1)

    def forward(self, img1_batch, flow):
        return optical_flow_warp(img1_batch, flow)

def parametric_of(img1_batch,img2_batch,device,verbose=0):
    n,c,h,w = img1_batch.shape
    
    centers = torch.randn(n,2,device=device)*0.00001
    #with torch.no_grad():
    #    centers[:,0] += 1/2
    #    centers[:,1] += 1/2
    centers = Parameter(centers) 

    addflow = Parameter(0.0001*torch.randn(n,h,w,2,device=device))
    velocity = Parameter(0.5*torch.abs(torch.rand(n,device=device)))
    opt = optim.AdamW([velocity,addflow],lr=0.01,weight_decay=1e-4)
    epochs = 150

    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w), indexing='ij')
    grid_x = grid_x.repeat(n, 1, 1).to(device)
    grid_y = grid_y.repeat(n, 1, 1).to(device)
    
    reshaped_ts = img1_batch.view(n, -1)

    # Get the indices of the maximum element along the 1D tensor
    max_indices = torch.argmax(reshaped_ts, dim=1)
    yindices = max_indices // w
    xindices = max_indices % w
    #print(grid_x.shape)
   
    for n in range(epochs):
        totloss = 0
        totbatch = 0
        # construct flow from 3 parameters
        real_centers_x = xindices + centers[:,0]*0
        real_centers_y = yindices + centers[:,1]*0

        dx = grid_x - real_centers_x[:,None,None]
        dy = grid_y - real_centers_y[:,None,None]
        norm = torch.sqrt(dx**2 + dy**2)+0.01
       
        flowx = dx/norm*velocity[:,None,None]
        flowy = -dy/norm*velocity[:,None,None]
        #flow = rearrange(flow, 'b d h w -> b h w d') 
        flow = torch.stack((flowx,flowy),dim=3) + addflow
        
        predict = optical_flow_warp(img1_batch, flow)
        loss = torch.mean(torch.abs(predict - img2_batch)**2)
        

        opt.zero_grad()
        loss.backward()
        opt.step()
        
        with torch.no_grad():
            centers[centers<0] = 0
            centers[centers>1] = 1
            velocity[velocity>10] = 10
            velocity[velocity<0] = 0
            addflow[addflow>0.5] = 0.5
            addflow[addflow<-0.5] = -0.5
            
            #if torch.max(torch.abs(velocity)) > 10:
            #    velocity = velocity*10 / torch.max(torch.abs(velocity))
        
        totloss += loss
        totbatch += 1
        if verbose > 10:
            print("epoch %s, loss : %.6f, center x: %.3f, center y: %.3f, v: %.3f"%(n,totloss/totbatch,centers[0,0].item()*w, centers[0,1].item()*h,velocity[0].item()))  
        #print(centers)
    return flow


def raft_of(img1_batch,img2_batch,model,device):
    list_of_flows = model(img1_batch.to(device), img2_batch.to(device))
    return list_of_flows

def optical_flow_warp(image, flow):
    # image can be batch x channel x w x h
    # flow should be in the dimension batch x w x h x 2
    # Create grid of indices
    batch_size, num_channels, height, width = image.size()
    grid_y, grid_x = torch.meshgrid(torch.arange(0, height), torch.arange(0, width), indexing='ij')
    grid = torch.stack((grid_x.float(), grid_y.float()), dim=-1).unsqueeze(0)  # Add batch dimension
    grid = grid.repeat(batch_size, 1, 1, 1).to(image.device)
    #print(grid)
    # Add flow to grid
    new_grid = grid - flow  # Permute flow to match grid dimensions
    
    # Normalize grid to [-1, 1] range
    normalized_grid = 2 * new_grid / torch.tensor([width - 1, height - 1], device=image.device).view(1, 1, 1, -1) - 1

    # Perform differentiable image warping
    warped_image = F.grid_sample(image, normalized_grid, align_corners=True)

    return warped_image


def cv2_optical_flow(img1, img2):
    import cv2
    flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow


def lucas_kanade_optical_flow(image1, image2, window_size=5):
    # very useless
    kernel_size = window_size
    half_kernel = window_size // 2

    # Define spatial gradient operator
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float)

    # Compute image gradients
    grad_x = F.conv2d(image1, sobel_x.unsqueeze(0).unsqueeze(0), padding=1)
    grad_y = F.conv2d(image1, sobel_y.unsqueeze(0).unsqueeze(0), padding=1)
    _,_,nx,ny = image1.shape
    grad_t = image2 - image1
    A = torch.cat((torch.diag(grad_x.view(nx*ny)), torch.diag(grad_y.view(nx*ny))), dim=1)
    b = - grad_t.view(nx*ny)

    sol = A.T@torch.inverse(A@A.T)@b
    u = sol[:nx*ny]
    v = sol[nx*ny:]
    return u.view(1,1,nx,ny),v.view(1,1,nx,ny)
  

# Example usage
def lucas_example():
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 1 * 4))
    #image1 = torch.randn(1, 1, 100, 100) * 0 # Input image 1
    #image2 = torch.randn(1, 1, 100, 100) * 0 # Input image 2
    nx = 100
    ny = 100
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    image1 = torch.tensor(np.exp(-((X+0.2)**2+(Y)**2)/(2*1**2))).unsqueeze(0).unsqueeze(0).float()
    image2 = torch.tensor(np.exp(-((X-0.2)**2+(Y)**2)/(2*1**2))).unsqueeze(0).unsqueeze(0).float() 
    optical_flow_u, optical_flow_v = lucas_kanade_optical_flow(image1, image2)
    
    print(optical_flow_v.shape)
    imglist = [image1, image2,optical_flow_u, optical_flow_v]
    for i in range(4):
        img = imglist[i][0][0].detach().numpy()
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.savefig('figures/ofsample.png',bbox_inches='tight')

# Example usage
def cv2_example():
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(1, 5, figsize=(5 * 4, 1 * 4))
    #image1 = torch.randn(1, 1, 100, 100) * 0 # Input image 1
    #image2 = torch.randn(1, 1, 100, 100) * 0 # Input image 2
    nx = 100
    ny = 100
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    image1 = np.exp(-((X+1)**2+(Y)**2)/(2*0.1**2))
    image2 = np.exp(-((X-1)**2+(Y)**2)/(2*0.1**2))
    timestart = time.time()
    flow = cv2_optical_flow(image1/np.sum(image1), image2/np.sum(image2))
    timeend = time.time()
    print('%s seconds for optical flow'%(timeend - timestart))
    flow = flow/np.max(flow) * 1
    #return
    image1ts = torch.tensor(image1).unsqueeze(0).unsqueeze(0).float()
    flowts = torch.tensor(flow).unsqueeze(0).float()
    
   
    # Remap the original image based on the calculated optical flow
    warped_image = optical_flow_warp(image1ts, flowts)
    print(warped_image.shape)
    
    imglist = [image1, image1 ,image2, warped_image[0][0].detach().numpy(), flow[:,:,1]]
    for i in range(5):
        img = imglist[i]
        axes[i].imshow(img,extent=(-3, 3, -3, 3),)
        axes[i].axis('off')
    axes[1].quiver(X, Y, flow[:,:,0], flow[:,:,1], color='black')
    print(np.linalg.norm(imglist[2]), np.linalg.norm(imglist[0]-imglist[2]), np.linalg.norm(imglist[3]-imglist[2]))
    plt.savefig('figures/ofsample.png',bbox_inches='tight')

def raft_example():
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(1, 4, figsize=(4 * 8, 1 * 8))
    device = 'cuda'
    model = raft_large(pretrained=True, progress=False).to(device)
    model = model.eval()
    nx = 128
    ny = 128
    x = np.linspace(-3, 3, nx)
    y = np.linspace(-3, 3, ny)
    X, Y = np.meshgrid(x, y)
    
    image1 = torch.cat([torch.tensor(np.exp(-((X+1)**2+(Y)**2)/(2*1**2))).unsqueeze(0).unsqueeze(0).float() for i in range(3)], dim=1)
    image2 = torch.cat([torch.tensor(np.exp(-((X-1)**2+(Y)**2)/(2*1**2))).unsqueeze(0).unsqueeze(0).float() for i in range(3)], dim=1)
    optflow = raft_of(image1, image2, model, device)
    
    warped_image = optical_flow_warp(image1, torch.stack((optflow[0][0,0],optflow[0][0,1]),dim=2).cpu())
    
    imglist = [image1[0][0], image2[0][0], warped_image[0][0], optflow[0][0][1]]
    for i in range(4):
        img = imglist[i].cpu().detach().numpy()
        axes[i].imshow(img,extent=(-3, 3, -3, 3))
        axes[i].axis('off')
    axes[0].quiver(X, Y, optflow[0][0,0].cpu().detach().numpy(), optflow[0][0,1].cpu().detach().numpy(), color='black',scale=5000)

    plt.savefig('figures/ofsample.png',bbox_inches='tight')

def raft_example_2():
    fulldata, dinv = find_conductance(True,False)#hf.
    print(fulldata[0][0].shape)
    data = fulldata[148][0]
    tlist = [29,30,31,32,33,34,35,36]
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(2, len(tlist), figsize=(len(tlist) * 16, 2 * 8))
    device = "cuda"
    of = Optical_Flow(device=device)
    
    pictures = data[tlist].to(device).view(len(tlist),1,64,128)
    pictures[:,:,18:30,:]*=0
    flow = of.get_of(pictures[:-1], pictures[1:])
    # flows has dimension batch x w x h x 2
    #print(flow.shape)
    prediction = of.forward(pictures[:-1], flow)
    x = np.arange(128)
    y = np.arange(64)
    X, Y = np.meshgrid(x, y)
    
    
    for i in range(len(tlist)):
        axes[0,i].imshow(data[tlist[i]])
        axes[0,i].axis('off')
        axes[0,i].imshow(data[tlist[i]])
        if i < len(tlist)-1:
            axes[0,i].quiver(X, Y, 
                             flow[i,:,:,0].detach().cpu().numpy(), 
                             flow[i,:,:,1].detach().cpu().numpy(), 
                             scale=500)
        if i == 0:
            img = data[tlist[i]]
        else:
            img = prediction[i-1,0].detach().cpu().numpy()
        axes[1,i].imshow(img)
        axes[1,i].axis('off')
    plt.savefig('figures/heatsample.png',bbox_inches='tight')
    
def new_example():
    fulldata, dinv = find_conductance(False,False)#hf.
    print(fulldata[0][0].shape)
    data = fulldata[148][0]
    tlist = [29,30,31,32,33,34,35,36]
    
    device = "cuda"
    #of = Optical_Flow(device=device)
    
    pictures = data[tlist].to(device).view(len(tlist),1,32,64)
    pictures[:,:,18:30,:]*=0
    flow = parametric_of(pictures[:-1], pictures[1:], device)
    plt.clf()
    plt.close('all')
    fig, axes = plt.subplots(2, len(tlist), figsize=(len(tlist) * 16, 2 * 8))
    x = np.arange(64)
    y = np.arange(32)
    X, Y = np.meshgrid(x, y)
    prediction = optical_flow_warp(pictures[:-1],flow)
    for i in range(len(tlist)):
        axes[0,i].imshow(data[tlist[i]])
        axes[0,i].axis('off')
        axes[0,i].imshow(data[tlist[i]])
        if i < len(tlist)-1:
            axes[0,i].quiver(X, Y, 
                             flow[i,:,:,0].detach().cpu().numpy(), 
                             flow[i,:,:,1].detach().cpu().numpy(), 
                             scale=100)
        if i == 0:
            img = data[tlist[i]]
        else:
            img = prediction[i-1,0].detach().cpu().numpy()
        axes[1,i].imshow(img)
        axes[1,i].axis('off')
    plt.savefig('../figures/heatsample.png',bbox_inches='tight')

if __name__ == "__main__":
    new_example()
    
