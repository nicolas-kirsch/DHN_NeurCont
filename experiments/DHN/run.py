
import sys, os, logging, torch,time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)


from config import device
from controllers import PerfBoostController
from arg_parser import argument_parser, print_args
from plants import DHNDataset, DHNSystem
from assistive_functions import WrapLogger
from loss_functions import DHNLoss


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'LTI', 'saved_results')
save_folder = os.path.join(save_path, 'perf_boost_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)

# ----- parse and set experiment arguments -----
args = argument_parser()
# msg = print_args(args)    # TODO
# logger.info(msg)
torch.manual_seed(args.random_seed)

# ------------ 1. Dataset ------------
disturbance = {
    'type':'normal noise',
}

dataset = DHNDataset(
    random_seed=args.random_seed, horizon=args.horizon,
    state_dim=args.state_dim, disturbance=disturbance,
    cp = 4186*10**(-6), mass = 200
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=20)
train_data, test_data = train_data.to(device), test_data.to(device)



# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)

sys = DHNSystem(
    mass=200,cop = 2
).to(device)


ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x_init, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.l,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)




# ------------ 4. Loss ------------
#Size of the minimization 


loss_fn = DHNLoss(
    R=args.alpha_u*100, u_min=dataset.umin, u_max=dataset.umax, x_min=dataset.xmin,x_max=dataset.xmax,
    #alpha_uh = 1, alpha_ul=1,alpha_xh=1,a
    alpha_xl=0.1
)


# ------------ 5. Optimizer ------------
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr, weight_decay = 0.01)
valid_data = train_data      # use the entire train data for validation

### Test forward without demand and input ###
"""zero_cons = torch.zeros(1,24,1)
zero_cons[0,0,0] = 50

with torch.no_grad():
    x_log_test, u_log_test = sys.rollout(
        controller=ctl, data=train_data[0:1,:,0:1]
    )

plt.plot(range(test_data.shape[1]),x_log_test[0])
plt.plot(range(test_data.shape[1]),[40]*(test_data.shape[1]), "--", c = "grey")
plt.plot(range(test_data.shape[1]),[80]*(test_data.shape[1]), "--",c = "grey" )
plt.title("X profile over the horizon")
plt.xlabel("Time (h)")
plt.ylabel("Temperature (°C)")
plt.show()"""


# ------------ 6. Training ------------
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6
t = time.time()
for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()
        # simulate over horizon steps
        x_log, u_log = sys.rollout(controller=ctl, data=train_data_batch)
        # loss of this rollout
        loss, loss_x, loss_u = loss_fn.forward(x_log, u_log)
        # take a step
        loss.backward()
        optimizer.step()

    # print info
    if epoch%args.log_epoch == 0:
        msg = 'Epoch: %i --- train loss: %.2f --- Loss x : %.2f ---  loss u: %.2f'% (epoch, loss, loss_x,loss_u)

        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, u_log_valid = sys.rollout(
                    controller=ctl, data=valid_data
                )
                # loss of the valid data
                loss_valid, loss_x_v, loss_u_v = loss_fn.forward(x_log_valid, u_log_valid)
            msg += ' ---||--- validation loss: %.2f  --- Loss x v: %.2f ---  loss u v: %.2f' % (loss_valid,loss_x_v,loss_u_v)
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                best_valid_loss = loss_valid.item()
                best_params = ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (best so far)'
        duration = time.time() - t
        msg += ' ---||--- time: %.0f s' % (duration)
        logger.info(msg)
        t = time.time()

# set to best seen during training
if args.return_best:
    ctl.set_parameters_as_vector(best_params)


with torch.no_grad():
    x_log_test, u_log_test = sys.rollout(
        controller=ctl, data=valid_data
    )

plt.figure()
for i in range(valid_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),x_log_test[i]+25)
    plt.plot(range(test_data.shape[1]),[40]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[80]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("X profile over the horizon")
    plt.xlabel("Time (h)")
    plt.ylabel("Temperature (°C)")

plt.figure()
for i in range(valid_data.shape[0]): 
    plt.plot(range(test_data.shape[1]),u_log_test[i],label = i )
    plt.plot(range(test_data.shape[1]),[0]*(test_data.shape[1]), "--", c = "grey")
    plt.plot(range(test_data.shape[1]),[4]*(test_data.shape[1]), "--",c = "grey" )
    plt.title("U profile over the horizon")
    plt.xlabel("Time (h)")
    plt.legend()
    plt.ylabel("Energy (MJ)")
plt.show()