from Models import NetworkedRENs
import numpy as np
import math
import matplotlib.pyplot as plt
import scipy
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
from os.path import dirname, join as pjoin
import torch
from torch import nn

dtype = torch.float
device = torch.device("cpu")

plt.close('all')
# Import Data
folderpath = os.getcwd()
filepath = pjoin(folderpath, 'dataset_sysID_3tanks.mat')
data = scipy.io.loadmat(filepath)
print(type(data))
dExp, yExp, dExp_val, yExp_val, Ts = data['dExp'], data['yExp'], \
    data['dExp_val'], data['yExp_val'], data['Ts'].item()
nExp = yExp.size

t = np.arange(0, np.size(dExp[0, 0], 1) * Ts, Ts)

# plt.plot(t, yExp[0,-1])
# plt.show()

seed = 16
torch.manual_seed(seed)

n = torch.tensor([2, 1, 1])  # input dimensions

p = torch.tensor([1, 1, 1])  # output dimensions

n_xi = np.array([10, 10, 10])
# nel paper n1, numero di stati
l = np.array([10, 10, 10])  # nel paper q, dimension of the square matrix D11 -- number of _non-linear layers_ of the RE

M = torch.tensor([[0, 0, 1], [0, 0, 0], [1, 0, 0], [0, 1, 0]])
Mud = torch.eye(4)
Mey = torch.eye(3)
M = M.float()
N = 3

RENsys = NetworkedRENs(N, M, Mud, Mey, n, p, n_xi, l, top=True)

# Define the system

# Define Loss function
MSE = nn.MSELoss()

# Define Optimization method
learning_rate = 1.0e-1
optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
optimizer.zero_grad()

t_end = yExp[0, 0].shape[1] - 1
t_end = 150

epochs = 70
LOSS = np.zeros(epochs)
loss = 0

for epoch in range(epochs):
    if epoch == epochs - epochs / 2:
        learning_rate = 1.0e-2
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    if epoch == epochs - epochs / 6:
        learning_rate = 1.0e-3
        optimizer = torch.optim.Adam(RENsys.parameters(), lr=learning_rate)
    optimizer.zero_grad()
    loss = 0
    for exp in range(nExp - 1):
        xi = []
        y = torch.from_numpy(yExp[0, exp]).float().to(device)
        y = y.squeeze()
        yRENm = torch.randn(3, t_end + 1, device=device, dtype=dtype)
        yRENm[:, 0] = y[:, 0]
        for j in range(N):
            xi.append(torch.randn(RENsys.r[j].n, device=device, dtype=dtype))
        d = torch.from_numpy(dExp[0, exp]).float().to(device)
        xi = torch.cat(xi)
        for t in range(1, t_end):
            yRENm[:, t], xi, gamma, gammaw, Q, lmip = RENsys(t, d[:, t - 1], xi, checkLMI=True)

        loss = loss + MSE(yRENm[:, 0:yRENm.size(1)], y[:, 0:t_end + 1])
        # ignorare da loss effetto condizione iniziale

    loss = loss / nExp
    loss.backward()
    # loss.backward(retain_graph=True)

    optimizer.step()

    print(f"Epoch: {epoch + 1} \t||\t Loss: {loss}")
    print(f"Gamma: {gamma}")
    print(f"Q: {Q}")
    print(f"Q: {lmip}")
    LOSS[epoch] = loss

plt.figure('30')
plt.plot(LOSS)
plt.title("LOSS")
plt.show()

xi = []
y = torch.from_numpy(yExp[0, 0]).float().to(device)
y = y.squeeze()
yRENm = torch.zeros(3, t_end + 1, device=device, dtype=dtype)
yRENm[:, 0] = y[:, 0]
for j in range(N):
    xi.append(torch.randn(RENsys.r[j].n, device=device, dtype=dtype))
dtrain = torch.from_numpy(dExp[0, 0]).float().to(device)
xi = torch.cat(xi)
for t in range(1, t_end):
    yRENm[:, t], xi, gamma, gammaw, Q = RENsys(t, dtrain[:, t - 1], xi)

t_end = yExp_val[0, 0].shape[1] - 1

xiVal = []
yval = torch.from_numpy(yExp_val[0, 0]).float().to(device)
yval = yval.squeeze()

yRENm_val = torch.zeros(3, t_end + 1, device=device, dtype=dtype)
yRENm_val[:, 0] = yval[:, 0]
for j in range(N):
    xiVal.append(torch.randn(RENsys.r[j].n, device=device, dtype=dtype))
dval = torch.from_numpy(dExp_val[0, 0]).float().to(device)
xiVal = torch.cat(xiVal)
loss_val = 0
for t in range(1, t_end):
    yRENm_val[:, t], xiVal, gammaVal, gammaw, Q = RENsys(t, dval[:, t - 1], xiVal)
loss_val = loss_val + MSE(yRENm_val[:, 0:yRENm_val.size(1)], yval[:, 0:t_end + 1])

plt.figure('1')
plt.plot(yRENm[0, 0:t_end].detach().numpy(), label='REN train')
plt.plot(y[0, 0:t_end].detach().numpy(), label='y train')
plt.title("output 1 train 3 RENs")
plt.legend()
plt.show()

plt.figure('2')
plt.plot(yRENm_val[0, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[0, 0:t_end].detach().numpy(), label='y val')
plt.title("output 1 val 3 RENs")
plt.legend()
plt.show()

plt.figure('3')
plt.plot(yRENm[1, 0:t_end].detach().numpy(), label='REN train')
plt.plot(y[1, 0:t_end].detach().numpy(), label='y train')
plt.title("output 2 train 3 RENs")
plt.legend()
plt.show()

plt.figure('4')
plt.plot(yRENm_val[1, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[1, 0:t_end].detach().numpy(), label='y val')
plt.title("output 2 val 3 RENs")
plt.legend()
plt.show()

plt.figure('5')
plt.plot(yRENm[2, 0:t_end].detach().numpy(), label='REN train')
plt.plot(y[2, 0:t_end].detach().numpy(), label='y train')
plt.title("output 3 train 3 RENs")
plt.legend()
plt.show()

plt.figure('6')
plt.plot(yRENm_val[2, 0:t_end].detach().numpy(), label='REN val')
plt.plot(yval[2, 0:t_end].detach().numpy(), label='y val')
plt.title("output 3 val 3 RENs")
plt.legend()
plt.show()

plt.figure('7')
plt.plot(dtrain[1, 0:t_end].detach().numpy(), label='u')
plt.plot(dval[1, 0:t_end].detach().numpy(), label='u val')
plt.title("input 3 RENs")
plt.legend()
plt.show()

pytorch_total_params_3RENs = sum(p.numel() for p in RENsys.parameters() if p.requires_grad)

print(f"param 3 RENs: {pytorch_total_params_3RENs}")

print(f"Loss Validation 3 RENs: {loss_val}")