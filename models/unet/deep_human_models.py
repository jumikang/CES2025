from __future__ import print_function
import torchvision
import torch.nn as nn
import torch.utils.data
import torch.optim as optim
from .unet import ATUNet_UV, UNet, ATUNet_Color

class BaseModule(nn.Module):
    def __init__(self, im2d_in=6,
                 return_uv=False,
                 return_disp=False,
                 split_last=True):
        super(BaseModule, self).__init__()
        self.return_uv = return_uv
        self.return_disp = return_disp
        self.split_last = split_last
        if self.return_uv and self.return_disp:
            self.imuv2uvdisp = ATUNet_UV(in_ch=im2d_in, out_ch=6, split_last=self.split_last)
        elif self.return_uv and not self.return_disp:
            self.imuv2uv = UNet(in_ch=6, out_ch=3)
        elif not self.return_uv and self.return_disp:
            self.imuv2disp = UNet(in_ch=6, out_ch=3)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        if self.return_uv and self.return_disp:
            y_uvd, _ = self.imuv2uvdisp(x)
            output = {'uv': y_uvd[:, :3, :, :],
                      'disp': y_uvd[:, 3:, :, :]}
        elif self.return_uv and not self.return_disp:
            y_uv = self.imuv2uv(x)
            output = {'uv': y_uv}
        elif not self.return_uv and self.return_disp:
            y_disp = self.imuv2disp(x)
            output = {'disp': y_disp}
        return output

class BaseColorModule(nn.Module):
    def __init__(self, im2d_in=3,
                 split_last=False):
        super(BaseColorModule, self).__init__()
        self.split_last = split_last
        self.im2im = ATUNet_Color(in_ch=im2d_in, out_ch=3, split_last=self.split_last)

        # weight initialization
        for m in self.modules():
            m = weight_init_basic(m)

    def forward(self, x):
        y, _ = self.im2im(x)
        output = {'color': y}
        return output

class DeepHumanUVNet(nn.Module):
    def __init__(self, opt):
        super(DeepHumanUVNet, self).__init__()
        self.model = BaseModule(return_uv=opt['DATA']['return_uv'],
                                return_disp=opt['DATA']['return_disp'])
        self.automatic_optimization = True

    @torch.no_grad()
    def in_the_wild_step(self, input):
        self.model.eval()
        return self.model(input)

class DeepHumanColorNet(nn.Module):
    def __init__(self, opt):
        super(DeepHumanColorNet, self).__init__()
        self.model = BaseColorModule()

    @torch.no_grad()
    def in_the_wild_step(self, input):
        self.model.eval()
        return self.model(input)

def weight_init_basic(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)

# for test
if __name__ == '__main__':
    # input = Variable(torch.randn(4, 3, 256, 256)).float().cuda()
    input = Variable(torch.randn(4, 2, 5, 5)).float().cuda()
    _, b = torch.Tensor.chunk(input, chunks=2, dim=1)
    print(b.shape)

    print(b)
    print(input)
