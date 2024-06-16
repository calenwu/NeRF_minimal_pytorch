import torch
import torch.nn as nn
import torch.nn.functional as F

class RealNVP(nn.Module):
    def __init__(self, num_scales=2, in_channels=3, mid_channels=64, num_blocks=8):
        super(RealNVP, self).__init__()

        self.flows = nn.ModuleList()
        self.prior = torch.distributions.Normal(0, 1)

        for _ in range(num_scales):
            self.flows.append(FlowStep(in_channels, mid_channels, num_blocks))
            self.flows.append(FlowStep(in_channels, mid_channels, num_blocks))

    def forward(self, x):
        log_det_jacobian = 0

        for flow in self.flows:
            x, log_det_jacobian = flow(x, log_det_jacobian)

        return x, log_det_jacobian

    def inverse(self, z):
        for flow in reversed(self.flows):
            z = flow.inverse(z)
        return z

    def log_prob(self, x):
        z, log_det_jacobian = self.forward(x)
        log_pz = self.prior.log_prob(z).sum(dim=[1, 2, 3])
        return log_pz + log_det_jacobian

class FlowStep(nn.Module):
    def __init__(self, in_channels, mid_channels, num_blocks):
        super(FlowStep, self).__init__()
        self.s = nn.ModuleList([self._create_network(in_channels, mid_channels, num_blocks) for _ in range(2)])
        self.t = nn.ModuleList([self._create_network(in_channels, mid_channels, num_blocks) for _ in range(2)])
        self.mask = torch.zeros((1, in_channels, 8, 8), dtype=torch.float32)
        self.mask[:, :, ::2, ::2] = 1
        self.mask[:, :, 1::2, 1::2] = 1

    def _create_network(self, in_channels, mid_channels, num_blocks):
        network = [nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_blocks):
            network.append(nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1))
            network.append(nn.ReLU(inplace=True))
        network.append(nn.Conv2d(mid_channels, in_channels, kernel_size=3, padding=1))
        return nn.Sequential(*network)

    def forward(self, x, log_det_jacobian):
        x_a, x_b = x * self.mask, x * (1 - self.mask)
        s_a, t_a = self.s[0](x_a), self.t[0](x_a)
        y_a = x_a
        y_b = x_b * torch.exp(s_a) + t_a
        x = y_a + y_b
        log_det_jacobian += s_a.sum(dim=[1, 2, 3])

        x_a, x_b = x * (1 - self.mask), x * self.mask
        s_b, t_b = self.s[1](x_b), self.t[1](x_b)
        y_a = x_a
        y_b = x_b * torch.exp(s_b) + t_b
        x = y_a + y_b
        log_det_jacobian += s_b.sum(dim=[1, 2, 3])

        return x, log_det_jacobian

    def inverse(self, z):
        z_a, z_b = z * (1 - self.mask), z * self.mask
        s_b, t_b = self.s[1](z_b), self.t[1](z_b)
        y_a = z_a
        y_b = (z_b - t_b) * torch.exp(-s_b)
        z = y_a + y_b

        z_a, z_b = z * self.mask, z * (1 - self.mask)
        s_a, t_a = self.s[0](z_a), self.t[0](z_a)
        y_a = z_a
        y_b = (z_b - t_a) * torch.exp(-s_a)
        z = y_a + y_b

        return z
