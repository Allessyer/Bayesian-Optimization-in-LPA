import torch
from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement
from botorch.optim import optimize_acqf

# for plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
rcParams['axes.labelpad'] = 20
import numpy as np
from matplotlib import cm


def create_x_train(r_min, r_max, n_points, dim = 2):
    train_x = []
    bounds = []
    for i in range(dim):
        d = 1
        bound = torch.stack([r_min[i] * torch.ones(d), r_max[i] * torch.ones(d)])
        x = bound[0] + (bound[1] - bound[0]) * torch.rand(n_points, d)
        if len(train_x) == 0:
            bounds = bound
            train_x = x
        else:
            bounds = torch.cat((bounds,bound),1)
            train_x = torch.cat((train_x,x),1)
    return train_x, bounds

def target_function_2D(x_train,objective_function):
    train_x = x_train.clone().detach()
    train_y = []
    
    for x1,x2 in train_x:
        x1 = x1.numpy()
        x2 = x2.numpy()
        x = [x1,x2]
        y = objective_function(x)
        train_y.append(y)
    
    return torch.Tensor(train_y)

def generate_initial_data_2D(r_min, r_max, objective_function,n_points):
    train_x, bounds = create_x_train(r_min, r_max, n_points, dim = 2)
    train_y = target_function_2D(train_x,objective_function)   
    train_y = train_y.unsqueeze(-1)

    return train_x,train_y,bounds

def init_GP(x_train, y_train, maximize):
    # 1. Создаем гауссовский процесс
    gaussian_process = SingleTaskGP(x_train,y_train) 
    mll = ExactMarginalLogLikelihood(gaussian_process.likelihood,gaussian_process) 
    # 2. Обучаем гауссовский процесс
    fit_gpytorch_model(mll) 

    # 3. Создаем acquisition function
    if maximize:
        best_y_train = y_train.max()
    else:
        best_y_train = y_train.min()
    
    # EI
    EI = ExpectedImprovement(
        model=gaussian_process,
        best_f=best_y_train,
        maximize=maximize
        )
    
    acquisition_function = EI
    return gaussian_process,acquisition_function

def get_next_points(x_train, y_train, bounds, maximize, n_points=1):
    gaussian_process,acquisition_function = init_GP(x_train, y_train, maximize)

    # 4. Оптимизируем acquisition function и получаем нового кандидата
    candidates,_ = optimize_acqf(
    acq_function=acquisition_function,
    bounds=bounds,
    q=n_points,
    num_restarts=200,
    raw_samples=512,
    )
    return candidates,gaussian_process,acquisition_function

def plot_result(x_train,y_train,bounds,gaussian_process,acquisition_function,step):
    
    # Plot existing data: x_train, y_train
    x_train = x_train.cpu().numpy()
    y_train = y_train.cpu().numpy()

    font = {'family': 'serif',
    'color':  '#484441',   
    'weight': 'normal',
    'size': 16,
    }

    fig = plt.figure(figsize=(20,10))
    ax = fig.gca(projection='3d')

    ax.scatter(x_train[:5,0],x_train[:5,1], y_train[:5], color = 'gray', s=50,label="initial data")
    ax.scatter(x_train[5:-1,0],x_train[5:-1,1], y_train[-1], color = 'black', s=30, label='Observed data')
    ax.scatter(x_train[-1,0],x_train[-1,1], y_train[-1], color = 'red', s=30, label='last candidate')

    ax.set_xlabel('$X_1$', fontdict=font)
    ax.set_ylabel('$X_2$',fontdict=font)
    ax.set_zlabel(r'$Y$', fontdict=font)
    ax.tick_params(labelcolor='blue', labelsize=13, width=10)

    ax.legend()
    ax.view_init(elev=25., azim=50)
    
    # -plot result of bo step
    r_min = bounds[0]
    r_max = bounds[1]

    x_data, _ = create_x_train(r_min, r_max, n_points=500, dim = 2)
    # --plot gaussian process mean 
    posterior = gaussian_process.posterior(x_data)
    mean = posterior.mean.squeeze(-1).cpu().detach().numpy()
    x1 = x_data[:,0].cpu().detach().numpy()
    x2 = x_data[:,1].cpu().detach().numpy()
    
    surf = ax.plot_trisurf(x1, x2, mean, cmap=cm.jet, linewidth=0.1, alpha=0.4)
    cbar = fig.colorbar(surf, shrink=0.5, aspect=5)
    cbar.set_label('Mean', rotation=270)
    

    # --plot acquisition function

    x_grid = x_data.unsqueeze(-2)
    with torch.no_grad():
        acqu = acquisition_function(x_grid).cpu().detach().numpy()
    
    
    surf2 = ax.tricontourf(x_data[:,0], x_data[:,1], acqu,zdir='z', cmap = cm.viridis, offset=-0.5)
    ax.set_zlim(-0.5, mean.max()+5)
    cbar2 = fig.colorbar(surf2, shrink=0.5, aspect=5)
    cbar2.set_label('Acquisition function', rotation=270)
    
    # plt.colorbar();
    plt.show()
    fig.savefig(f"bo_result_{step}.png")
    
def Bayesian_Optimization(x_train, y_train, bounds, objective_function, maximize, n_steps):
    
    for step in range(n_steps):
        print(f'Nr. of optimization step: {step}')
        x_pred,gaussian_process,acquisition_function = get_next_points(x_train, y_train, bounds,maximize, n_points=1)
        y_pred = target_function_2D(x_pred,objective_function).unsqueeze(-1)
        
        print(f'New candidates are: {x_pred}')
        print(f'New value: {y_pred}')
        
        x_train = torch.cat([x_train, x_pred])
        y_train = torch.cat([y_train, y_pred])
        
        if maximize:
            arg_max = y_train.argmax().item()
            best_y = y_train.max().item()
            best_x = x_train[arg_max]
        else:
            arg_min = y_train.argmin().item()
            best_y = y_train.min().item()
            best_x = x_train[arg_min]
            
        print(f'Best x_train: {best_x}')
        print(f'Best y_train: {best_y}')
        
        plot_result(x_train,y_train,bounds,gaussian_process,acquisition_function,step)
        
    return x_train,y_train