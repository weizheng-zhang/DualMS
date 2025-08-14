import torch
import torch.nn as nn
import numpy as np

# Random Fourier Features
class InputMapping(nn.Module):
    def __init__(self, d_in, d_out, sigma=2):
        super().__init__()
        self.B = nn.Parameter(torch.randn(d_out // 2, d_in) * sigma,
                              requires_grad=False)

    def forward(self, x):
        
        x = (2*np.pi*x) @ self.B.T
        return torch.cat([torch.sin(x), torch.cos(x)], dim=-1)

class SurfaceModel(nn.Module):

    def __init__(self, rff_sigma=2):
        super().__init__()

        self.rff = InputMapping(3, 2048, sigma=rff_sigma)
        self.f = nn.Sequential(
            nn.Linear(2048, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 256),
            nn.Softplus(),
            nn.Linear(256, 1),
        )        
        

    def forward(self, x, red_points = None, blue_points = None, bdry_points = None, only_f = False):
        
        y = self.rff(x)
        f = self.f(y)
        if only_f:
            return f
        df = torch.autograd.grad(outputs=f,
                                 inputs=x,
                                 grad_outputs=torch.ones_like(f),
                                 create_graph=True,
                                 only_inputs=True)[0]

        if red_points != None:
            red_res = self.f(self.rff(red_points))
        else:
            red_res = None
        if blue_points != None:
            blue_res = self.f(self.rff(blue_points))
        else:
            blue_res = None
        if bdry_points != None:
            bdry_res = self.f(self.rff(bdry_points))
        else:
            bdry_res = None

        return {
            'f': f,
            'df': df,
            'red_res': red_res,
            'blue_res': blue_res,
            'bdry_res': bdry_res
        }