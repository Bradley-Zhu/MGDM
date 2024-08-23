import torch
import gpytorch
from matplotlib import pyplot as plt
from diffusion_composite_fluid import FluidDataset, SmokeDataset
from diffusion_heat_double import NISTDataset
from tqdm.auto import tqdm
import numpy as np
from utils import uxy_to_color
from evaluations import mse_psnr_ssim
from pinn import Unet, consistency_score,rearrange

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, num_task=2):
        super(MultitaskGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_task
        )
        self.covar_module = gpytorch.kernels.MultitaskKernel(
            gpytorch.kernels.RBFKernel(), num_tasks=num_task, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)

class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([train_y.shape[1]]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([train_y.shape[1]])),
            batch_shape=torch.Size([train_y.shape[1]])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )
    
def fit_a_list_of_gm(train_x_raw, full_train_y, verbose=1, device='cpu'):
    n = len(train_x_raw)
    train_x = train_x_raw.reshape(n,-1)
    ttbatch = full_train_y.shape[2]
    model_list = []
    likelihood_list = []
    for xy in range(full_train_y.shape[1]):
        for bid in tqdm(range(ttbatch)):
            train_y = full_train_y[:,xy,bid]
            # initialize likelihood and model
            n_task = train_y.shape[-1]
            
            #likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_task)
            #model = MultitaskGPModel(train_x, train_y, likelihood, num_task=n_task)
            
            likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=n_task)
            model = BatchIndependentMultitaskGPModel(train_x, train_y, likelihood).to(device)
            
            # Find optimal model hyperparameters
            model.train()
            likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

            training_iter = 50
            for i in range(training_iter):
                # Zero gradients from previous iteration
                optimizer.zero_grad()
                # Output from model
                output = model(train_x)
                # Calc loss and backprop gradients
                loss = -mll(output, train_y)
                loss.backward()
                if verbose>10:
                    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
                        i + 1, training_iter, loss.item(),
                        model.covar_module.base_kernel.lengthscale.item(),
                        model.likelihood.noise.item()
                    ))
                optimizer.step()
            model_list.append(model)
            likelihood_list.append(likelihood)
    return model_list, likelihood_list

def plot_fluid(final_pred, gt_y):
    with torch.no_grad():
        # Initialize plot
        N = 4
        f, ax = plt.subplots(N, 2, figsize=(8, 4*N))
            
        # Plot training data as black stars
        for z in range(N):
            image = final_pred[z,0].detach().cpu().numpy()
            ax[z,0].imshow(-image,cmap='coolwarm')
            ax[z,0].axis("off")
                
            image = gt_y[z,0].detach().cpu().numpy()
            ax[z,1].imshow(-image,cmap='coolwarm')
            ax[z,1].axis("off")
        plt.savefig('samples/double_physics_fluid/gaussian_fluid.png',bbox_inches='tight')

def plot_nist(final_pred, gt_y):
    with torch.no_grad():
        # Initialize plot
        N = 2
        num_time = 5
        f, ax = plt.subplots(N*2, num_time, figsize=(4*num_time, 4*N))
            
        # Plot training data as black stars
        for z in range(N):
            for tid in range(num_time):
                image = final_pred[z*num_time+tid,0].detach().cpu().numpy()
                ax[z*2,tid].imshow(image,cmap='afmhot')
                ax[z*2,tid].axis("off")
                    
                image = gt_y[z*num_time+tid,0].detach().cpu().numpy()
                ax[z*2+1,tid].imshow(image,cmap='afmhot')
                ax[z*2+1,tid].axis("off")
        plt.savefig('samples/nist/gaussian_nist.png',bbox_inches='tight')

def model_eval(model_list,likelihood_list, test_x_raw, test_y_raw, base, PLOT=None):
    n = len(test_x_raw)
    test_x = test_x_raw.reshape(n,-1)
    ttbatch = test_y_raw.shape[2]
    bs = 15
    bnum = n//bs
    gt_y = []
    final_pred = []
    meanvars = []
    for bbid in tqdm(range(bnum)):
        final_predi = [[] for i in range(test_y_raw.shape[1])]
        for xy in  range(test_y_raw.shape[1]):
            for bid in (range(ttbatch)):
                model = model_list[xy*ttbatch+bid]
                likelihood = likelihood_list[xy*ttbatch+bid]
                model.eval()
                likelihood.eval()
                # Test points are regularly spaced along [0,1]
                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(test_x[bbid*bs:(bbid+1)*bs])).mean
                    final_predi[xy].append(observed_pred)
            final_predi[xy] = torch.stack(final_predi[xy], dim=1)
            #print(final_predi[xy].shape)
        final_predi = torch.stack(final_predi, dim=1)
        final_predi = final_predi + base[bbid*bs:(bbid+1)*bs]
        gt_yi = test_y_raw[bbid*bs:(bbid+1)*bs] + base[bbid*bs:(bbid+1)*bs]
        mvi, _ = mse_psnr_ssim(gt_yi, final_predi)
        meanvars.append(mvi)
        final_pred.append(final_predi)
        gt_y.append(gt_yi)
    final_pred = torch.cat(final_pred, dim=0)
    gt_y = torch.cat(gt_y, dim=0)
    totdt = len(final_pred)
    totdt = (totdt//5)*5
    final_pred = final_pred[:totdt]
    gt_y = gt_y[:totdt]
    meanvars = torch.tensor(meanvars)
    if PLOT is not None:
        PLOT(final_pred, gt_y)
    
    print('-'*10)
    print(meanvars)
    print('-'*10)
    return (torch.mean(meanvars, dim=0).tolist(),torch.std(meanvars, dim=0).tolist()), final_pred

def plot_std_nist(model_list,likelihood_list, test_x_raw, test_y_raw, base, PLOT=None):
    n = len(test_x_raw)
    test_x = test_x_raw.reshape(n,-1)
    ttbatch = test_y_raw.shape[2]
    bs = 15
    bnum = n//bs
    
    for bbid in tqdm(range(bnum)):
        final_predi = [[] for i in range(test_y_raw.shape[1])]
        final_stdi = [[] for i in range(test_y_raw.shape[1])]
        for xy in  range(test_y_raw.shape[1]):
            for bid in (range(ttbatch)):
                model = model_list[xy*ttbatch+bid]
                likelihood = likelihood_list[xy*ttbatch+bid]
                model.eval()
                likelihood.eval()
                # Test points are regularly spaced along [0,1]
                # Make predictions by feeding model through likelihood
                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    observed_pred = likelihood(model(test_x[bbid*bs:(bbid+1)*bs]))
                    final_predi[xy].append(observed_pred.mean)
                    final_stdi[xy].append(torch.diag(observed_pred.covariance_matrix).reshape(observed_pred.mean.shape))
                    #print(observed_pred.mean.shape)
                    #print(observed_pred.covariance_matrix.shape)
                    #print(final_stdi[xy][-1])
                    #assert False
            final_predi[xy] = torch.stack(final_predi[xy], dim=1)
            final_stdi[xy] = torch.stack(final_stdi[xy], dim=1)

            #print(final_predi[xy].shape)
        final_predi = torch.stack(final_predi, dim=1)
        final_stdi = torch.stack(final_stdi, dim=1)
        final_predi = final_predi + base[bbid*bs:(bbid+1)*bs]
        gt_yi = test_y_raw[bbid*bs:(bbid+1)*bs] + base[bbid*bs:(bbid+1)*bs]
        break
    num_time = 5
    f, ax = plt.subplots(2, num_time, figsize=(4*num_time, 4))
            
    # Plot training data as black stars
    for tid in range(num_time):
        image = gt_yi[tid,0].detach().cpu().numpy()
        ax[0,tid].imshow(image,cmap='afmhot')
        ax[0,tid].axis("off")
        
        image = np.log(1e-5+final_stdi[tid,0].detach().cpu().numpy())
        print(image)
        ax[1,tid].imshow(image,cmap='afmhot')
        ax[1,tid].axis("off")
        plt.savefig('samples/nist/gaussian_std.png',bbox_inches='tight')

def trainfluid():
    #video_path = r"taichigen/data/"
    #dataset = FluidDataset(folder=video_path,from_raw=False)
    dataset = SmokeDataset()
    traindssuperset = dataset[:int(0.9*len(dataset))]
    trainds = torch.stack([ds for ds in traindssuperset[:100]],dim=0)
    device = "cuda"
    
    init_context, physics_context, cs2, gt_data = dataset.slice_data(trainds, device)
    train_x = physics_context[:,-2:]
    train_y = gt_data - physics_context[:,-1:]
    
    model_list, likelihood_list = fit_a_list_of_gm(train_x, train_y, verbose=1,device=device)
    testdssuperset = dataset[int(0.9*len(dataset)):]
    testds = torch.stack([ds for ds in testdssuperset[:1000]],dim=0)

    init_context, physics_context, cs2, gt_data = dataset.slice_data(testds, device)
    test_x = physics_context[:,-2:]
    test_y = gt_data - physics_context[:,-1:]
    #test_x = testds[:,1:3]
    #test_y = testds[:,dataset.gaps[2]:] - testds[:,dataset.gaps[1]:dataset.gaps[2]]
    
    (metrics_mean, metrics_std), _ = model_eval(model_list,likelihood_list, test_x, test_y, physics_context[:,-1:], testds[:,0], PLOT=plot_fluid)
    print(metrics_mean)
    print(metrics_std)

def trainnist():
    #video_path = r"taichigen/data/"
    #dataset = FluidDataset(folder=video_path,from_raw=False)
    dataset = NISTDataset()
    traindssuperset = dataset[:int(0.8*len(dataset))]
    trainds = torch.stack([ds for ds in traindssuperset[:100]],dim=0)
    device = "cuda"
    data, heatsource, physics_context = dataset.slice_data(trainds, device)
    data_window = dataset.data_time_window
    
    train_x = physics_context[:,data_window//2:].reshape(-1,1,*physics_context.shape[-2:])
    train_y = data[:,data_window//2:].reshape(-1,1,*physics_context.shape[-2:]) - train_x
    model_list, likelihood_list = fit_a_list_of_gm(train_x, train_y, verbose=1,device=device)
    testdssuperset = dataset[int(0.8*len(dataset)):]
    testds = torch.stack([ds for ds in testdssuperset[:]],dim=0)

    gt_data, heatsource, physics_context = dataset.slice_data(testds, device)
    test_x = physics_context[:,data_window//2:].reshape(-1,1,*physics_context.shape[-2:])
    test_y = gt_data[:,data_window//2:].reshape(-1,1,*physics_context.shape[-2:]) - test_x
    #test_x = testds[:,1:3]
    #test_y = testds[:,dataset.gaps[2]:] - testds[:,dataset.gaps[1]:dataset.gaps[2]]
    plot_std_nist(model_list,likelihood_list, test_x, test_y, test_x, PLOT=plot_nist)
    return
    (metrics_mean, metrics_std),predictions = model_eval(model_list,likelihood_list, test_x, test_y, test_x, PLOT=plot_nist)
    flownet = Unet(channels=10, dim=64, out_dim=8).to(device)
    flownet.load_state_dict(torch.load('playground/intermediate/unet_of.pt'))
    flownet.eval()
    rpredictions = rearrange(predictions, '(b t) 1 w h -> b t w h', t=5)
    cs_list = []
    dt = 10
    tot = len(testds)//dt
    for i in range(tot):
        csi = consistency_score(rpredictions[i*dt:(i+1)*dt],testds[i*dt:(i+1)*dt],dataset,flownet,{'device':device})
        cs_list.append(csi.item())
    #print(cs_list)
    metrics_mean.append(torch.mean(torch.tensor(cs_list)).item())
    metrics_std.append(torch.std(torch.tensor(cs_list)).item())

    
    print(metrics_mean)
    print(metrics_std)   
    #print(consistency_score_value)
if __name__ == "__main__":
    #trainfluid()
    trainnist()