import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime

from src.models import Controller,DHN
from src.plots import plot_trajectories, plot_traj_vs_time
from src.loss_functions import f_loss_states, f_loss_u, f_loss_ca, f_loss_obst, f_loss_side, f_upper_bound, f_lower_bound
from src.utils import calculate_collisions, set_params, generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

sys_model = "corridor"
prefix = ''

with_barrier = False

# # # # # # # # Parameters and hyperparameters # # # # # # # #
params = set_params(sys_model)
min_dist, t_end, n_agents, x0, xbar, linear, learning_rate, epochs, Q, \
alpha_u, alpha_x, alpha_obst, n_xi, l, n_traj, std_ini = params


epochs = 500
n_traj = 1
std_ini = 0.5
l,n_xi = 8,8

learning_rate = 1e-4 # *0.5

alpha_u = 0.5
# alpha_barrier = 5  # 250
alpha_x = 1
std_ini_param = 0.005
use_sp = False

epoch_print = 20
n_train = 100
n_test = 1000 - n_train
validation = True
validation_period = 50
n_validation = 100

show_plots = False

t_ext = t_end * 4

x_init = 38

# # # # # # # # Set up logger # # # # # # # #
log_name = sys_model + prefix
now = datetime.now().strftime("%m_%d_%H_%Ms")
filename_log = os.path.join(BASE_DIR, 'log')
if not os.path.exists(filename_log):
    os.makedirs(filename_log)
filename_log = os.path.join(filename_log, log_name+'_log_' + now)

logging.basicConfig(filename=filename_log, format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger(sys_model)
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# # # # # # # # Define models # # # # # # # #
sys = DHN(mass=200, cop =2)
ctl = Controller(sys.f, sys.n, sys.m, n_xi, l, use_sp=use_sp, t_end_sp=t_end, std_ini_param=std_ini_param)
print(ctl.parameters)

# # # # # # # # Define optimizer and parameters # # # # # # # #
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)


# # # # # # # # Training # # # # # # # #
msg = "\n------------ Begin training ------------\n"
msg += "Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate
msg += " -- epochs: %i" % epochs + " -- n_traj: %i" % n_traj + " -- std_ini: %.2f\n" % std_ini
msg += " -- alpha_u: %.4f" % alpha_u + " -- alpha_ca: %i" % alpha_x + " -- alpha_obst: %.1e\n" % alpha_obst
msg += "REN info -- n_xi: %i" % n_xi + " -- l: %i " % l + "use_sp: %r\n" % use_sp
msg += "--------- --------- ---------  ---------"
logger.info(msg)
best_valid_loss = 1e9
best_params = None
best_params_sp = None
loss_log = []
for epoch in range(epochs):
    
    optimizer.zero_grad()
    loss_x_l, loss_u, loss_x_h, loss_obst, loss_side, loss_barrier = 0, 0, 0, 0, 0, 0

    w_in = torch.zeros(t_end, sys.n)
    for i in range(len(w_in)):
        w_in[i, :] = 0

    u = torch.zeros(sys.m)
    x = torch.zeros(sys.n)
    x[0] = x_init


    xi = torch.zeros(ctl.psi_u.n_xi)
    omega = (x, u)
    x_log = []
    u_log = []
    for t in range(t_end):
        x_prev = x
        x, _ = sys(t, x, u, w_in[t, :])

        u, xi, omega = ctl(t, x, xi, omega)
        loss_u = loss_u + alpha_u * f_loss_u(t, u) / n_traj
        loss_x_l = loss_x_l + alpha_x * f_lower_bound(x,40)

        loss_x_h = loss_x_h + alpha_x * f_upper_bound(x,80)

        x_log.append(x.detach())
        
        u_log.append(u.detach())



    """    print(x_log)
    print("U")
    print(u_log)"""

    loss = loss_x_l + loss_x_h + loss_u
    loss_log.append(loss.detach())


    msg = "Epoch: {:>4d} --- Loss: {} ---||--- Loss x: {}".format(epoch, loss/t_end, loss_x_l)
    msg += " --- Loss u: {:>9.4f} --- Loss x_l: {:>9.4f} --- Loss x_h: {}".format(loss_u,loss_x_l,loss_x_h)
    msg += " --- Loss side: {:>9.2f}--- Loss barrier: {:>9.2f}".format(loss_side, loss_barrier)
    loss.backward()
    optimizer.step()
    ctl.psi_u.set_model_param()
    logger.info(msg)
"""    for j in x_log:
        print(j)
        print(10*torch.log10(1+torch.exp((j-80))).sum())
"""
print("WOWWOWO")

# # # # # # # # Print & plot results # # # # # # # #
x_log = torch.zeros(t_end, sys.n)
u_log = torch.zeros(t_end, sys.m)
w_in = torch.zeros(t_end, sys.n)
w_in = torch.zeros(t_end, sys.n)
for i in range(len(w_in)):
    w_in[i, :] = 0


u = torch.zeros(sys.m)
x = torch.tensor([x_init])
xi = torch.zeros(ctl.psi_u.n_xi)
omega = (x, u)
for t in range(t_end):

    x, _ = sys(t, x, u, w_in[t, :])
    u, xi, omega = ctl(t, x, xi, omega)



    x_log[t] = x.detach()
    u_log[t] = u.detach()

plt.figure()
plt.plot(range(t+1),u_log.numpy())
plt.title("U")

plt.figure()
plt.plot(range(t+1),x_log.numpy())
plt.title("X")

plt.figure()
plt.title("Loss")
plt.plot(range(epochs),loss_log)
plt.show()
# Number of collisions
"""
# Set parameters to the best seen during training
if validation and best_params is not None:
    ctl.psi_u.load_state_dict(best_params)
    ctl.psi_u.eval()
    if use_sp:
        ctl.sp.load_state_dict(best_params_sp)
        ctl.sp.eval()
    ctl.psi_u.set_model_param()


# # # # # # # # Save trained model # # # # # # # #
fname = log_name + '_T' + str(t_end) + '_stdini' + str(std_ini) + '_RS' + str(random_seed)
fname += '.pt'
filename = os.path.join(BASE_DIR, 'trained_models')
if not os.path.exists(filename):
    os.makedirs(filename)
filename = os.path.join(filename, fname)
save_dict = {'psi_u': ctl.psi_u.state_dict(),
             'Q': Q,
             'alpha_u': alpha_u,
             'alpha_ca': alpha_x,
             'alpha_obst': alpha_obst,
             'n_xi': n_xi,
             'l': l,
             'n_traj': n_traj,
             'epochs': epochs,
             'std_ini_param': std_ini_param,
             'use_sp': use_sp,
             'linear': linear
             }
if use_sp:
    save_dict['sp'] = ctl.sp.state_dict()
torch.save(save_dict, filename)
logger.info('[INFO] Saved trained model as: %s' % fname)
"""


