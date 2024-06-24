#!/usr/bin/env python
import torch
import torch.nn as nn
import torch.nn.functional as F


# REN implementation in the acyclic version
# See paper: "Recurrent Equilibrium Networks: Flexible dynamic models with guaranteed stability and robustness"
class PsiU(nn.Module):
    def __init__(self, n, m, n_xi, l, std_ini_param=None):
        super().__init__()
        self.n = n
        self.n_xi = n_xi
        self.l = l
        self.m = m
        if std_ini_param is None:
            std_ini_param = 0.1
        # # # # # # # # # Training parameters # # # # # # # # #
        # Auxiliary matrices:
        std = std_ini_param
        self.X = nn.Parameter((torch.randn(2*n_xi+l, 2*n_xi+l)*std))
        self.Y = nn.Parameter((torch.randn(n_xi, n_xi)*std))
        # NN state dynamics:
        self.B2 = nn.Parameter((torch.randn(n_xi, n)*std))
        # NN output:
        self.C2 = nn.Parameter((torch.randn(m, n_xi)*std))
        self.D21 = nn.Parameter((torch.randn(m, l)*std))
        self.D22 = nn.Parameter((torch.randn(m, n)*std))
        # v signal:
        self.D12 = nn.Parameter((torch.randn(l, n)*std))
        # bias:
        # self.bxi = nn.Parameter(torch.randn(n_xi))
        # self.bv = nn.Parameter(torch.randn(l))
        # self.bu = nn.Parameter(torch.randn(m))
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements
        self.epsilon = 0.001
        self.F = torch.zeros(n_xi, n_xi)
        self.B1 = torch.zeros(n_xi, l)
        self.E = torch.zeros(n_xi, n_xi)
        self.Lambda = torch.ones(l)
        self.C1 = torch.zeros(l, n_xi)
        self.D11 = torch.zeros(l, l)
        self.set_model_param()

    def set_model_param(self):
        n_xi = self.n_xi
        l = self.l
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2*n_xi+l)
        h1, h2, h3 = torch.split(H, [n_xi, l, n_xi], dim=0)
        H11, H12, H13 = torch.split(h1, [n_xi, l, n_xi], dim=1)
        H21, H22, _ = torch.split(h2, [n_xi, l, n_xi], dim=1)
        H31, H32, H33 = torch.split(h3, [n_xi, l, n_xi], dim=1)
        P = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + P + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, t, w, xi):
        vec = torch.zeros(self.l)
        vec[0] = 1
        epsilon = torch.zeros(self.l)
        v = F.linear(xi, self.C1[0,:]) + F.linear(w, self.D12[0,:])  # + self.bv[0]
        epsilon = epsilon + vec * torch.tanh(v/self.Lambda[0])
        for i in range(1, self.l):
            vec = torch.zeros(self.l)
            vec[i] = 1
            v = F.linear(xi, self.C1[i,:]) + F.linear(epsilon, self.D11[i,:]) + F.linear(w, self.D12[i,:])  # self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v/self.Lambda[i])
        E_xi_ = F.linear(xi, self.F) + F.linear(epsilon, self.B1) + F.linear(w, self.B2)  # + self.bxi
        xi_ = F.linear(E_xi_, self.E.inverse())
        u = F.linear(xi, self.C2) + F.linear(epsilon, self.D21) + F.linear(w, self.D22)  # + self.bu
        return 20*u, xi_


class PsiX(nn.Module):
    def __init__(self, f):
        super().__init__()
        n = 4
        m = 2
        self.f = f

    def forward(self, t, omega):
        y, u = omega
        psi_x = self.f(t, y, u)
        omega_ = 0
        return psi_x, omega_


class Input(torch.nn.Module):
    def __init__(self, m, t_end, active=True):
        super().__init__()
        self.t_end = t_end
        self.m = m
        if active:
            std = 0.0
            self.u = torch.nn.Parameter((torch.randn(t_end, m) * std))
        else:
            self.u = torch.zeros(t_end, m)

    def forward(self, t):
        if t < self.t_end:
            return self.u[t, :]
        else:
            return torch.zeros(self.m)


class Controller(nn.Module):
    def __init__(self, f, n, m, n_xi, l, use_sp=False, t_end_sp=None, std_ini_param=None):
        super().__init__()
        self.n = n
        self.m = m
        self.psi_x = PsiX(f)
        self.psi_u = PsiU(self.n, self.m, n_xi, l, std_ini_param=std_ini_param)
        self.output_amplification = 20
        self.use_sp = use_sp
        if use_sp:  # setpoint that enters additively in the reconstruction of omega
            self.sp = Input(n, t_end_sp, active=use_sp)

    def forward(self, t, y_, xi, omega):
        psi_x, _ = self.psi_x(t, omega)
        w_ = y_ - psi_x
        if self.use_sp:
            w_ = w_ + self.sp(t)
        u_, xi_ = self.psi_u(t, w_, xi)
        u_ = u_ * self.output_amplification
        omega_ = (y_, u_)
        return u_, xi_, omega_

class DHN(nn.Module):
    def __init__(self,mass,cop):
        super().__init__()
        self.mass = mass
        self.cp = 4186
        self.cop = cop
        self.n = 1
        self.m = 1

    def f(self, x,u,t): 
        f = x + self.cop/(self.mass*self.cp) * u
        return f

    def forward(self, t, x, u, w):
        x_ = self.f(x, u,t) - w/(self.cp*self.mass)

         # here we can add noise not modelled
        y = x_
        return x_, y

