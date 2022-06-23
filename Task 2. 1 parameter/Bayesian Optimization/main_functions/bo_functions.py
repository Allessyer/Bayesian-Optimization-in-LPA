from my_functions.assel_functions import analyse_simulation
from my_functions.assel_functions import delete_dir
from my_functions.run_simulation import run_simulation

from timeit import default_timer as timer

import torch
import numpy as np

import os
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (7,4*2)

from botorch.models import SingleTaskGP
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition import ExpectedImprovement, PosteriorMean, ProbabilityOfImprovement
from botorch.optim import optimize_acqf


def target_function(x_train,objective_function):
    """
    Calculates for each point x_i in x_train objective function value
    and returns 

    Parameters
    ----------
    x_train: 1D torch.Tensor
            set of points where we calculate our objective function as initial points
    function: function
            objective function
    
    Returns
    -------
    torch.Tensor of values of objective function in each point x_i

    """
    train_x = x_train.clone().detach()
    train_y = []
    for x in train_x:
        x = x[0].numpy()
        y = objective_function(x)
        train_y.append(y)
    
    return torch.Tensor(train_y)

def generate_initial_data(r_min, r_max, objective_function,n_points=10,d=1):
    
    """
    Generates initial dataset train_x, train_y

    Parameters
    ----------
    r_min: torch.float
            lower bound of x_train
            
    r_max: torch.float
            upper bound of x_train
            
    objective_function: function
            objective function
    
    n_points: torch.int
            number of points in dataset
    
    d: torch.int
        dimension
    
    Returns
    -------
        train_x: 1D torch.Tensor 
        train_y: 1D torch.Tensor 
        bounds: bounds

    """
    
    bounds = torch.stack([r_min * torch.ones(d), r_max * torch.ones(d)])
    train_x = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_points, d)
    train_y = target_function(train_x,objective_function)   
    train_y = train_y.unsqueeze(-1)

    return train_x,train_y,bounds


def init_GP(x_train, y_train, maximize):
    
    """
    Initialize Gaussian Process Regression
    
    Parameters
    ----------
    x_train: 1D torch.Tensor
            training set of the varied parameter
            
    y_train: 1D torch.Tensor
            objective function values
    
    Returns
    -------
        gaussian_process
        acquisition_function
    
    """
    
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
    
    PM = PosteriorMean(model = gaussian_process)
    
    PI = ProbabilityOfImprovement(
        model=gaussian_process,
        best_f=best_y_train,
        maximize=maximize
        )
    
    acquisition_function = EI
    return gaussian_process,acquisition_function

def get_next_points(x_train, y_train, bounds, maximize, n_points=1):
    
    """
    Calculates next candidate 
    
    Parameters
    ----------
    x_train: 1D torch.Tensor
            training set of the varied parameter
            
    y_train: 1D torch.Tensor
            objective function values
    
    bounds: bounds of x_train value
    
    n_points: torch.int
            number of generated candidates
    
    Returns
    -------
        candidates
        gaussian_process
        acquisition_function
    
    """
    
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

def plot_result_1D(x_train,y_train,bounds,gaussian_process,acquisition_function,x_pred,y_pred,step):
    
    font = {'family': 'serif',
        'color':  '#484441',   
        'weight': 'normal',
        'size': 16,
        }
    
    fig, (ax1, ax2) = plt.subplots(2,figsize=(10,5*2))
    fig.suptitle(f'BO step = {step}',fontdict=font,color='black',fontsize=20)
    
    # Plot mean of predicted Gaussian Processes
    r_min = bounds[0].item()
    r_max = bounds[1].item()
    x_data = torch.linspace(r_min,r_max,100)

    posterior = gaussian_process.posterior(x_data)
    # Get upper and lower confidence bounds (2 standard deviations from the mean)
    lower, upper = posterior.mvn.confidence_region()
    # Plot posterior means as blue line
    ax1.plot(x_data.cpu().detach().numpy(), posterior.mean.cpu().detach().numpy(), color='b',label='Mean')
    # Shade between the lower and upper confidence bounds
    ax1.fill_between(x_data.cpu().detach().numpy(), lower.cpu().detach().numpy(), upper.cpu().detach().numpy(), alpha=0.3)    

    # Plot existing data: x_train, y_train
    ax1.scatter(x_train,y_train,color='black',label='Observed Data')

    # Plot candidate: x_pred, y_pred
    ax1.scatter(x_pred,y_pred,color='red',label='Candidate')
    
    # plot acquisition function
    x_grid = x_data.reshape(-1, 1, 1).to(x_train)
    
    with torch.no_grad():
        acqu = acquisition_function(x_grid).cpu().numpy()
        
    
    x_grid = x_grid.reshape([x_grid.shape[0],])
    ax2.plot(x_grid, acqu, color = '#2d6b22', label='acquisition')
    
    ax1.axvline(x_pred, linestyle = '--',color='#8ab446')
    ax2.axvline(x_pred, linestyle = '--',color='#8ab446')
    ax1.set_xlabel('x',fontdict=font)
    ax1.set_ylabel('f(x)',fontdict=font)
    ax1.legend()
    ax1.grid()
    ax1.tick_params(labelcolor='#474745', labelsize=10, width=3)

    ax2.set_xlabel('x',fontdict=font)
    ax2.set_ylabel('Acquisition function',fontdict=font)
    ax2.legend()
    ax2.grid()
    ax2.tick_params(labelcolor='#474745', labelsize=10, width=3)
    
    
    plt.show()
    fig.savefig(f"bo_result_{step}.png")
    
    
def Bayesian_Optimization(r_min, r_max,x_train,y_train,bounds,objective_function,n_steps,maximize):
    for step in range(n_steps):
        print(f'Nr. of optimization step: {step}')
        x_pred,gaussian_process,acquisition_function = get_next_points(x_train, y_train, bounds,maximize, n_points=1)
        
        y_pred = target_function(x_pred,objective_function).unsqueeze(-1)
        print(f'New candidate is: {x_pred}')
        print(f'New target value is: {y_pred}')

        # Add new candidates to x_train
        x_train = torch.cat([x_train, x_pred])
        y_train = torch.cat([y_train, y_pred])

        # Find new best result
        best_y_train = y_train.min().item()
        print(f'Best point performs this way: {best_y_train}')

        plot_result_1D(x_train,y_train,bounds,gaussian_process,acquisition_function,x_pred,y_pred,step)
        
    return x_train,y_train


def plot_convergence(X, y, n_steps, maximize=False):
    """
    Plot convergence history: distance between consecutive x's and value of
    the best selected sample
    Parameters
    ----------
    X : torch.tensor, shape=(n_samples, dim)
        History of evaluated input values
    y : torch.tensor, shape=(n_samples,)
        History of evaluated objective values
    Returns
    -------
    """
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    X = X[-n_steps:]
    dist = torch.norm(X[1:] - X[:-1], dim=-1).cpu().numpy()
    if maximize:
        cum_best = np.maximum.accumulate(y[-n_steps:].cpu().numpy())
    else:
        cum_best = np.minimum.accumulate(y[-n_steps:].cpu().numpy())

    axes[0].plot(dist, '.-', c='r',)
    axes[0].set_xlabel('Iteration', fontsize=14)
    axes[0].set_ylabel(r"$d(x_i - x_{i - 1})$", fontsize=14)
    axes[0].set_title("Distance between consecutive x's", fontsize=14)
    axes[0].grid(True)


    axes[1].plot(cum_best, '.-')
    axes[1].set_xlabel('Iteration', fontsize=14)
    axes[1].set_ylabel('Best y', fontsize=14)
    axes[1].set_title('Value of the best selected sample', fontsize=14)
    axes[1].grid(True)

    fig.tight_layout()
    fig.savefig('convergence.png')