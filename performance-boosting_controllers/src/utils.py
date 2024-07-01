import torch
import os
import numpy as np
import pickle


# For saving logs when running
class WrapLogger():
    def __init__(self, logger, verbose=True):
        self.can_log = not (logger is None)
        self.logger = logger
        self.verbose = verbose

    def info(self, msg):
        if self.can_log:
            self.logger.info(msg)
        if self.verbose:
            print(msg)

    def close(self):
        if not self.can_log:
            return
        while len(self.logger.handlers):
            h = self.logger.handlers[0]
            h.close()
            self.logger.removeHandler(h)


def generate_data(t_end,cp,mass, n_data_total = 50):

    # generate data
    n_data_total = n_data_total
    n_states = 1
    n_w = n_states


    # Define the window size for the moving average
    window_size = 3

    # Initial heat demand profile (baseline)
    heat_demand =  [30,20, 25, 30, 35, 40, 50, 60, 70, 80, 100, 90, 80, 70, 60, 50, 60, 80, 100, 90, 80, 70, 50, 40]
    # Compute the moving average
    smoothed_heat_demand = np.convolve(heat_demand, np.ones(window_size)/window_size, mode='same')*0.06

    data_x0 = 40 + 40*torch.rand(n_data_total, n_states)

    d = torch.zeros(n_data_total,t_end,n_w)  

    for i in range(n_data_total):
        d[i] = torch.from_numpy(smoothed_heat_demand).reshape(t_end,n_w) + torch.randn(t_end,n_w)
        d[i][0] = -data_x0[i]*cp*mass
        


    data = d

    return data
