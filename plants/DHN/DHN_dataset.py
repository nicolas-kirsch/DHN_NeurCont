import torch
import numpy as np # TODO: move to torch
from plants import CostumDataset
import matplotlib.pyplot as plt


class DHNDataset(CostumDataset):
    def __init__(self, random_seed, horizon, disturbance, state_dim,cp,mass,xmin= None,xmax=None,umin=None,umax=None):
        # experiment and file names
        exp_name = 'DHN'
        file_name = disturbance['type'].replace(" ", "_")+'_data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

        self.cp = cp
        self.mass = mass
        self.horizon = horizon
        self.disturbance = disturbance
        self.state_dim = state_dim

        if umin == None:
            self.umin = torch.tensor([0
                                  ])
        else: 
            self.umin = umin

        if umax == None:
            self.umax = torch.tensor([4
                                  ])
        else: 
            self.umax = umax
            
        if xmin == None:
            self.xmin = torch.tensor([15
                                  ])
        else: 
            self.xmin = xmin
            
        if xmax == None:
            self.xmax = torch.tensor([55
                                  ])
        else: 
            self.xmax = xmax 


        self.random_state = np.random.RandomState(random_seed)  # TODO: remove and use torch

    # ---- data generation ----
    def _generate_data(self, num_samples):
    
        # generate data
        n_data_total = num_samples
        n_states = 1
        n_w = n_states


        # Define the window size for the moving average
        window_size = 3

        # Initial heat demand profile (baseline)
        heat_demand =  [30,20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]
        heat_demand = np.zeros(len(heat_demand))
        # Compute the moving average
        #smoothed_heat_demand = -np.convolve(heat_demand, np.ones(window_size)/window_size, mode='same')*0.06
 
        data_x0 = (40 + 40*torch.rand(n_data_total, n_states))-25
        print("D_x0")
        print(data_x0)
        d = torch.zeros(n_data_total,self.horizon,n_w)  

        for i in range(n_data_total):
            #d[i] = torch.from_numpy(smoothed_heat_demand).reshape(self.horizon,n_w) + torch.randn(self.horizon,n_w)
            d[i] = torch.from_numpy(heat_demand).reshape(self.horizon,n_w) 
            d[i][0] = data_x0[i]*self.cp*self.mass
            


        data = d

        return data



