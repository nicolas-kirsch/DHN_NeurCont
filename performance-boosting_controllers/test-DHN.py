import torch
import matplotlib.pyplot as plt
import os
import pickle
import logging
from datetime import datetime
import copy


from src.models import Controller,DHN
from src.loss_functions import f_activation, f_loss_u, f_upper_bound, f_lower_bound
from src.utils import generate_data
from src.utils import WrapLogger


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

random_seed = 3
torch.manual_seed(random_seed)

sys_model = "DHN"
prefix = ''

with_barrier = False

# # # # # # # # Parameters and hyperparameters # # # # # # # #



epochs = 2
std_ini = 0.5
l,n_xi = 10,10

learning_rate = 1e-4 # *0.5

alpha_u = 0.1
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
t_end = 24
t_ext = t_end * 4




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
ctl = Controller(sys.f, sys.n, sys.m, n_xi, l)

data = generate_data(t_end,sys.cp,sys.mass)


# # # # # # # # Define optimizer and parameters # # # # # # # #
optimizer = torch.optim.Adam(ctl.parameters(), lr=learning_rate)


# # # # # # # # Training # # # # # # # #
msg = "\n------------ Begin training ------------\n"
msg += "Problem: " + sys_model + " -- t_end: %i" % t_end + " -- lr: %.2e" % learning_rate
msg += " -- epochs: %i" % epochs 
msg += " -- alpha_u: %.4f" % alpha_u + " -- alpha_x: %i" % alpha_x + " -- " 
msg += "REN info -- n_xi: %i" % n_xi + " -- l: %i " % l
msg += "--------- --------- ---------  ---------"



logger.info(msg)
loss_log = []
for epoch in range(epochs):
    for i in range(data.size()[0]):
        w_in = data[i]

        optimizer.zero_grad()
        loss_x_l, loss_u_min, loss_x_h, loss_u_h,loss_u_l, loss_u_act  = 0, 0, 0, 0,0,0


        u = torch.zeros(sys.m)
        x = torch.zeros(sys.n)


        xi = torch.zeros(ctl.psi_u.n_xi)
        omega = (x, u)
        x_log = []
        u_log = []
        for t in range(t_end):
            x_prev = x
            x, _ = sys(t, x, u, w_in[t, :])

            u, xi, omega = ctl(t, x, xi, omega)
            loss_u_min = loss_u_min + alpha_u * f_loss_u(t, u)
            loss_x_l = loss_x_l + alpha_x * f_lower_bound(x,40)

            loss_x_h = loss_x_h + alpha_x * f_upper_bound(x,80)
            loss_u_h = loss_u_h + f_upper_bound(u,3)
            loss_u_l = loss_u_l + f_lower_bound(u,0)
            #loss_u_act = loss_u_act + f_activation(u,1)
            

            x_log.append(x.detach())
            
            u_log.append(u.detach())


        loss = loss_x_h + loss_x_l + loss_u_min
        
        if i == 0:
            loss_log.append(loss.detach())



        
        loss.backward()
        optimizer.step()
        ctl.psi_u.set_model_param()


    msg = "Epoch: {:>4d} --- Loss: {} ---||".format(epoch, loss)
    msg += " --- Loss u: {:>9.4f} --- Loss x_l: {:>9.4f} --- Loss x_h: {}".format(loss_u_min,loss_x_l,loss_x_h)
    msg += " --- Loss u_l: {:>9.4f}--- Loss u_act: {:>9.4f}".format(loss_u_h,loss_u_act)
    logger.info(msg)

print("WOWWOWO")


# # # # # # # # Print & plot results # # # # # # # #
x_log = torch.zeros(t_end, sys.n)
u_log = torch.zeros(t_end, sys.m)


u = torch.zeros(sys.m)
x = torch.zeros(sys.n)
xi = torch.zeros(ctl.psi_u.n_xi)
omega = (x, u)

 #### TODO: generate validation data
for t in range(t_end):

    x, _ = sys(t, x, u, w_in[t, :])
    u, xi, omega = ctl(t, x, xi, omega)



    x_log[t] = x.detach()
    u_log[t] = u.detach()


print(u_log)

plt.figure()
plt.plot(range(t+1),u_log.numpy())
plt.title("U profile over the horizon")
plt.xlabel("Time (h)")
plt.ylabel("Energy consumption (MJ)")

plt.figure()
plt.plot(range(t+1),x_log.numpy())
plt.title("X profile over the horizon")
plt.xlabel("Time (h)")
plt.ylabel("Temperature (°C)")

plt.figure()
plt.title("Loss evolution over the epochs")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(range(epochs),loss_log)
plt.show()


"""plt.figure()
plt.title("W")
plt.plot(range(t),w_in[1:].detach())
"""

