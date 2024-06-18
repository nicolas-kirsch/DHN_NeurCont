import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
import numpy as np
from torch import nn

dtype = torch.float
device = torch.device("cpu")

# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'dataset_sysID_3tanks.mat')

from Models import Controller,Plant,PsiX,REN
import math

#### Size of elements ####

t_end = 20
n = torch.tensor(1)  # input dimensions

p = torch.tensor(1)  # output dimensions

n_xi = np.array(10)

l = np.array(1)

#### Initialize data ####

#Controller state
xi = torch.randn(n_xi, device=device, dtype=dtype)

#Disturbance
d = torch.from_numpy(np.array([1,2,3,4,5]+[0]*15)).float().to(device)

#Controller output
u = torch.zeros(t_end,dtype=dtype,device=device)

#Plant state
x = torch.zeros(t_end,dtype=dtype,device=device)
x[0] = 20


#### Initialize plant and controller ####

dhn = Plant(m=100,cp=4186,cop = 2)
cont = Controller(Plant,n,p,n_xi,l)

#### Forward pass ####

for t in range(1,t_end):
    x[t] = dhn.forward(t,x,u)+d[t]
    u[t],xi= cont.forward(t,x,u,xi)

print(x)
