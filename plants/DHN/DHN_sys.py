import torch
from assistive_functions import to_tensor
import numpy as np
import torch.nn.functional as F


# ---------- SYSTEM ----------
class DHNSystem(torch.nn.Module):
    def __init__(self,mass,cop,gamma = 0.99,u_init = None):
        
        super().__init__()

        self.mass = mass
        self.cop = cop
        self.cp = 4186*10**(-6)
        A = np.array([[gamma]])
        B = np.array([[self.cop/(self.mass*self.cp)]])
        
        self.A, self.B = to_tensor(A), to_tensor(B)
        self.x_init = to_tensor(np.array([[0]]))

        # Dimensions
        self.state_dim = self.A.shape[0]
        self.in_dim = self.B.shape[1]
        # Check matrices
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, self.in_dim)
        assert self.x_init.shape == (self.state_dim, 1)

        self.u_init = torch.zeros(1, int(self.x_init.shape[1])) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)



    def noiseless_forward(self, t, x: torch.Tensor, u: torch.Tensor):
        x = x.view(-1, 1, self.state_dim)
        u = u.view(-1, 1, self.in_dim)

        f = F.linear(x, self.A) + F.linear(u, self.B)
        return f
    
    def forward(self, t, x, u, w):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """

        return self.noiseless_forward(t, x, u) + w.view(-1, 1, self.state_dim)


    # simulation
    def rollout(self, controller, data: torch.Tensor):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """

  
        controller.reset()
        xs = (data[:, 0:1, :]/(self.mass*self.cp))
        us = controller.forward(xs[:, 0:1, :])
        for t in range(1, data.shape[1]):
            xs = torch.cat(
                (
                    xs,
                    torch.matmul(self.A, xs[:, t-1:t, :]) + torch.matmul(self.B, us[:, t-1:t, :]) + data[:, t:t+1, :]/(self.mass*self.cp)),
                1
            )

            us = torch.cat(
                (us, controller.forward(xs[:, t:t+1, :])),
                1
            )

        controller.reset()
        
        return xs, us