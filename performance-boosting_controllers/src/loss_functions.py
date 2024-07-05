import torch
import torch.nn.functional as F
import numpy as np




def f_loss_u(t, u):
    loss_u = (u ** 2).sum()
    return loss_u


def f_lower_bound(x, bound):
    threshold = 80
    beta = 1
 
    delta = bound - x
    """
    loss_bound = beta*torch.log(1+torch.exp((delta))).sum()

        if delta >= threshold:
            loss_bound = beta*delta.sum()
    """

    loss_bound = 100*torch.relu(delta).sum()
    return loss_bound

def f_upper_bound(x, bound):
    threshold = 80
    beta = 1

    delta = x -bound

    """


    loss_bound = 10*torch.log10(1+torch.exp((delta))).sum()

    if delta >= threshold:
        loss_bound = beta*delta.sum()
    """

    loss_bound = 100*torch.relu(delta).sum()
    return loss_bound

def f_activation(u, u_min):
    loss_low = f_upper_bound(u,0)
    loss_high = f_lower_bound(u,u_min)
    loss = torch.minimum(loss_low,loss_high).sum()
    return loss


