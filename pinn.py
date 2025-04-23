import numpy as np
import torch
from diffusion_fluid_tase import gen_train_test_dataset
from diffusion_heat_double import NISTDataset
from architectures import Unet
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from tqdm.auto import tqdm
import torch.nn.functional as F


from utils import uxy_to_color
from evaluations import mse_psnr_ssim
from playground.optical_flow_fit import unet_inference
from multiprocessing import cpu_count
import matplotlib.pyplot as plt
from diffusion_heat_double import Optical_Flow, rearrange, genmask, of_loss_small


def consistency_score(rprediction, rdata,ds,flownet,args):
    device=args['device']
    data_time_window = ds.data_time_window
    bs, _, _, _ = rdata.shape
    of = Optical_Flow()

    data, heatsource, physics_context = ds.slice_data(rdata,device)
    # calculate the predicted flow
    predicted_flow = unet_inference(flownet,physics_context)
    # predicted flow has a shape of batch x 4 x w x h x 2
    #predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> b t w h d1', d1=2) 
    
    xtreshape = rprediction#rearrange(rprediction, '(b t) w h -> b t w h', t=5) 
    predicted_flow = rearrange(predicted_flow, 'b (t d1) w h -> (b t) w h d1', d1=2) 
    sourcext = rearrange(xtreshape[:,:-1], 'b t w h -> (b t) 1 w h') 
    targetxt = rearrange(xtreshape[:,1:], 'b t w h -> (b t) 1 w h') 
    
    sub_physics_context = physics_context
    sub_physics_mask = genmask(sub_physics_context, 10,20).detach()
        
    sub_physics_mask = rearrange(sub_physics_mask[:,data_time_window//2:-1], 'b t w h -> (b t) 1 w h')
    of = Optical_Flow()
    consistency_loss = of_loss_small(sourcext, targetxt, sub_physics_mask, predicted_flow.detach(), of).detach()
    #print(consistency_loss, torch.sum((sub_physics_mask*torch.abs(sourcext))**2))
    return consistency_loss.item()/(1e-10+torch.sum((sub_physics_mask*torch.abs(sourcext))**2))

def test(unet, testdl, ds, args,SAVEFIG=True):
    device=args['device']
    metrics = []
    idx = 0
    for rdata in tqdm(testdl):   
        initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
        inputdata = args['encode_initial_context'](initial_context)
        data = args['encode_data'](odata)      
        #inputdata, data = args['slice_function'](rdata, device)
        
        with torch.no_grad():
            predictions = unet_inference(unet,inputdata,phs_time[:,0])
            if SAVEFIG and idx == 0:
                idx += 1
                plt.clf()
                plt.close('all')
                NIST = False
                if NIST:
                    nbs = 1
                    num_tstamp=5
                    data_time_window = ds.data_time_window
                    fig, axes = plt.subplots(2*nbs, num_tstamp, figsize=(num_tstamp* 4, nbs *2 * 2))
                    for bid in range(nbs):
                        for tid in range(num_tstamp):
                            #(b t) 1 w h 
                            img = rdata[bid,data_time_window//2 + tid].detach().cpu().numpy()
                            axes[bid*nbs,tid].imshow(img)
                            axes[bid*nbs,tid].axis("off")

                            img = predictions[bid*data_time_window//2 + tid].detach().cpu().numpy()
                            axes[bid*nbs+1,tid].imshow(img[0], cmap='afmhot')
                            axes[bid*nbs+1,tid].axis("off")
                    plt.savefig('samples/double_physics/pinn.png',bbox_inches='tight')

                else:
                    pbs = 4
                    fig, axes = plt.subplots(pbs, 2, figsize=(2* 4, pbs * 4))
                    for i in range(pbs):
                        image = data[i,0].detach().cpu().numpy()
                        axes[i,0].imshow(-image, cmap='coolwarm')
                        axes[i,0].axis('off')

                        image = predictions[i,0].detach().cpu().numpy()
                        axes[i,1].imshow(-image, cmap='coolwarm')
                        axes[i,1].axis('off')
                    plt.savefig('samples/double_physics_fluid/pinn.png',bbox_inches='tight')

            #print(predictions[0,0])
            allmetrics, _ = mse_psnr_ssim(data, predictions) 
            if 'flownet' in args:
                rpredictions = rearrange(predictions, '(b t) 1 w h -> b t w h', t=5)
                consistency_score_value = consistency_score(rpredictions,rdata,ds,args['flownet'],args)
                allmetrics.append(consistency_score_value)
            #print(allmetrics)
            #assert False 
            metrics.append(allmetrics)
            if torch.isnan(torch.tensor(allmetrics[0])):
                #print(predictions)
                #print(data)   
                print(torch.sum(torch.isnan(data)))
                print(torch.sum(torch.isnan(rdata)))     
                print(torch.sum(torch.isnan(predictions)))    
                print(allmetrics)
                assert False
    metrics = torch.tensor(metrics)
    return torch.mean(metrics, dim=0), torch.std(metrics, dim=0)


def train(unet, ds,testds, args):
    device=args['device']
    #assert len(ds) >= 100, 'you should have at least 100 images in your folder. at least 10k images recommended'
    trainlen = int(len(ds)*args['trainratio'])
    dl = DataLoader(ds, batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(testds, batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())
    opt = Adam(unet.parameters(), lr = args['train_lr'], weight_decay=5e-6)

    for epoch in range(args['num_epochs']):
        totloss = 0
        totlen = 0+1e-10
        for rdata in tqdm(dl):
            
            # data has the dimension batch x time x channel x weight x height
            #data = rdata.to(device)
            #initial_context, physics_context, _, data = ds.slice_data(rdata, device)            
            #inputdata = initial_context
            #inputdata, data = args['slice_function'](rdata, device)
            initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
            inputdata = args['encode_initial_context'](initial_context)
            #physics_context = args['encode_physics_context'](c1_context)
            data = args['encode_data'](odata)   
            
            #bs, dt, w, h = data.shape  
            try:
                predictions = unet_inference(unet,inputdata,phs_time[:,0])  
            except:
                print(data.shape)
                #print(physics_context.shape)
                #print(initial_context.shape)
                print(inputdata.shape)
                #print(rdata[0])                
                print('la fin')
                assert False                                    
            loss = torch.mean((data - predictions)**2)
            opt.zero_grad()
            loss.backward()
            opt.step()
            totloss += loss.item()
            totlen += len(data)
            #print(loss)
        print("[%s/%s], loss is %.4f"%(epoch, args['num_epochs'], totloss/totlen))
        if  epoch % 10 == 0 or epoch < 10:
            #for tdata in testdl:
            mvar,svar = test(unet, testdl, ds,args)
            print('-'*10)
            print(epoch)
            print(mvar)
            print(svar)
            #    break
            torch.save(unet.state_dict(), 'models/model_pinn_%s.pth'%args['suffix'])
    return 0


def show_some_images(unet,testdataset,args):
    device = args['device']
    rdata = testdataset[[1,5,8,12]] 
    initial_context, c1_context, c2_context, odata, phs_time = rdata[0].to(device), rdata[1].to(device), rdata[2].to(device), rdata[3].to(device), rdata[4].to(device)
                
    inputdata = args['encode_initial_context'](initial_context)
    data = args['encode_data'](odata)      
        
    with torch.no_grad():
        predictions = unet_inference(unet,inputdata,phs_time[:,0])

    plt.clf()
    plt.close('all')
    NIST = False

    pbs = 4
    fig, axes = plt.subplots(pbs, 2, figsize=(2* 4, pbs * 4))
    for i in range(pbs):
        image = data[i,0].detach().cpu().numpy()
        axes[i,0].imshow(-image, cmap='coolwarm')
        axes[i,0].axis('off')

        image = predictions[i,0].detach().cpu().numpy()
        axes[i,1].imshow(-image, cmap='coolwarm')
        axes[i,1].axis('off')
    plt.savefig('samples/double_physics_fluid/pinn_new.png',bbox_inches='tight')


def pinn_fluid():
    args = {
        'num_epochs': 101,
        'train_lr': 1e-6,
        'train_batch_size': 16,       
        'physics_guided':1,
        'device':'cuda',
        'suffix':'fluid',
        'trainratio':0.9,
    }
    dataset, testdataset = gen_train_test_dataset()
    #analyze_kappa(dataset,args)
    unet = Unet(channels=4, dim=64, out_dim=1).to(args['device'])
    def slice_function(rdata, device):
        initial_context, _, _, data = dataset.slice_data(rdata, device)            
        inputdata = initial_context
        return inputdata, data
    args['slice_function'] = slice_function
    args['encode_initial_context'] = lambda initialcondition: F.avg_pool2d(initialcondition,kernel_size=2, stride=2)
    args['encode_data'] = lambda data: F.avg_pool2d(data, kernel_size=2, stride=2)
    args['encode_physics_context'] = lambda phy: F.interpolate(phy,size=(128,128), mode='bilinear', align_corners=False)
    #train(unet, dataset, testdataset,args)
    unet.load_state_dict(torch.load('models/model_pinn_fluid.pth'))
    show_some_images(unet,testdataset,args)


def pinn_nist():
    args = {
        'num_epochs': 101,
        'train_lr': 1e-6,
        'train_batch_size': 16,       
        'device':'cuda',
        'suffix':'nist',
        'trainratio':0.8,
    }
    dataset = NISTDataset()
    #analyze_kappa(dataset,args)
    unet = Unet(channels=2, dim=64, out_dim=1).to(args['device'])
    data_time_window = dataset.data_time_window
    def slice_function(rdata, device):
        data, heatsource, physics_context = dataset.slice_data(rdata, device)            
        #inputdata = data[:,:data_time_window//2]
        #inputdata = inputdata.reshape(-1,1,*inputdata.shape[-2])
        target = data[:,data_time_window//2:].reshape(-1,1,*data.shape[-2:])
        bs = len(data)
        initial_context = torch.cat([data[i:i+1,:2].repeat(data_time_window//2, 1, 1, 1) for i in range(bs)])
        return initial_context, target
    args['slice_function'] = slice_function
    train(unet, dataset, args)

    unet.load_state_dict(torch.load('models/model_pinn_%s.pth'%args['suffix']))
    unet.eval()
    flownet = Unet(channels=10, dim=64, out_dim=8).to(args['device'])
    flownet.load_state_dict(torch.load('playground/intermediate/unet_of.pt'))
    flownet.eval()
    args['flownet'] = flownet

    trainlen = int(len(dataset)*args['trainratio'])
    dl = DataLoader(dataset[:trainlen], batch_size = args['train_batch_size'], shuffle = True, pin_memory = True, num_workers = cpu_count())
    testdl = DataLoader(dataset[trainlen:], batch_size = args['train_batch_size'], shuffle = False, pin_memory = True, num_workers = cpu_count())
    mvar,svar = test(unet, testdl, dataset,args)
    print('-'*10)
    print(mvar)
    print(svar)

if __name__ == "__main__":
    seed = 0
    #random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    pinn_fluid()
    #pinn_nist()
