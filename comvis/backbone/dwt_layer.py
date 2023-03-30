import torch
import torch.nn as nn

# for haar wavelet 2d transform picked from https://github.com/bes-dev/haar_pytorch
class HaarForward(nn.Module):
    alpha = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ll = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] + x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        lh = self.alpha * (x[:,:,0::2,0::2] + x[:,:,0::2,1::2] - x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hl = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] + x[:,:,1::2,0::2] - x[:,:,1::2,1::2])
        hh = self.alpha * (x[:,:,0::2,0::2] - x[:,:,0::2,1::2] - x[:,:,1::2,0::2] + x[:,:,1::2,1::2])
        return torch.cat([ll,lh,hl,hh], axis=1)


class HaarInverse(nn.Module):
    alpha = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.size(1) % 4 == 0, "The number of channels must be divisible by 4."
        size = [x.shape[0], x.shape[1] // 4, x.shape[2] * 2, x.shape[3] * 2]
        f = lambda i: x[:, size[1] * i : size[1] * (i + 1)]
        out = torch.zeros(size, dtype=x.dtype, device=x.device)
        out[:,:,0::2,0::2] = self.alpha * (f(0) + f(1) + f(2) + f(3))
        out[:,:,0::2,1::2] = self.alpha * (f(0) + f(1) - f(2) - f(3))
        out[:,:,1::2,0::2] = self.alpha * (f(0) - f(1) + f(2) - f(3))
        out[:,:,1::2,1::2] = self.alpha * (f(0) - f(1) - f(2) + f(3))
        return out
