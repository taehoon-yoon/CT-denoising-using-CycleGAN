import torch
import torch.nn as nn
from torch.nn import init
import functools
from torchsummary import summary


def get_norm_layer(norm_type='instance'):
    if norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    return norm_layer


def init_weight(m, init_type='normal'):
    name = m.__class__.__name__
    if name.find('Conv') != -1:
        if init_type == 'normal':
            init.normal_(m.weight.data, 0.0, 0.02)
        elif init_type == 'xavier':
            init.xavier_normal_(m.weight.data, gain=0.02)
        elif init_type == 'kaiming':
            init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        if m.bias is not None:
            init.constant_(m.bias.data, 0.0)
    elif name.find('Batch') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UNetSkipConnectionLayer(nn.Module):
    def __init__(self, out_c, inner_c, in_c=None, outter=False, inner=False, use_drop=False,
                 norm_layer=nn.BatchNorm2d, submodule=False):
        super().__init__()
        self.outter = outter

        if in_c is None:
            in_c = out_c

        if type(norm_layer) == functools.partial:
            use_bias = (norm_layer.func == nn.InstanceNorm2d)
        else:
            use_bias = (norm_layer == nn.InstanceNorm2d)

        if self.outter:
            self.outconv = nn.Conv2d(2, out_c, kernel_size=1, stride=1)

        downrelu = nn.LeakyReLU(0.2, True)
        downConv = nn.Conv2d(in_c, inner_c, kernel_size=4, stride=2, padding=1, bias=use_bias)
        uprelu = nn.ReLU(True)
        upNorm = norm_layer(out_c)
        downNorm = norm_layer(inner_c)

        if inner:
            upConv = nn.ConvTranspose2d(inner_c, out_c, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downConv]
            up = [uprelu, upConv, upNorm]
            layers = down + up

        elif outter:
            upConv = nn.ConvTranspose2d(inner_c * 2, out_c, kernel_size=4, stride=2, padding=1)
            down = [downConv]
            up = [uprelu, upConv]
            layers = down + [submodule] + up
        else:
            upConv = nn.ConvTranspose2d(inner_c * 2, out_c, kernel_size=4, stride=2, padding=1, bias=use_bias)
            down = [downrelu, downConv, downNorm]
            up = [uprelu, upConv, upNorm]
            if use_drop:
                layers = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                layers = down + [submodule] + up

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        if self.outter:
            return self.outconv(torch.cat([x, self.model(x)], dim=1))
        else:
            return torch.cat([x, self.model(x)], dim=1)


class UNetGenerator(nn.Module):
    def __init__(self, in_c, out_c, num_downs=8, ngf=64, norm_layer=nn.BatchNorm2d, use_drop=False):
        super().__init__()
        block = UNetSkipConnectionLayer(8 * ngf, 8 * ngf, inner=True, norm_layer=norm_layer)
        for i in range(num_downs - 5):
            block = UNetSkipConnectionLayer(8 * ngf, 8 * ngf, norm_layer=norm_layer, use_drop=use_drop, submodule=block)
        block = UNetSkipConnectionLayer(4 * ngf, 8 * ngf, norm_layer=norm_layer, submodule=block)
        block = UNetSkipConnectionLayer(2 * ngf, 4 * ngf, norm_layer=norm_layer, submodule=block)
        block = UNetSkipConnectionLayer(ngf, 2 * ngf, norm_layer=norm_layer, submodule=block)
        self.model = UNetSkipConnectionLayer(out_c, ngf, in_c=in_c, norm_layer=norm_layer, submodule=block, outter=True)

    def forward(self, x):
        return self.model(x)


class Discriminator(nn.Module):
    def __init__(self, in_c, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super().__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        layers = [
            nn.Conv2d(in_c, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        n_prev = 1
        n_cur = 1
        for i in range(1, n_layers):
            n_cur = min(8, 2 ** i)
            layers += [
                nn.Conv2d(n_prev * ndf, n_cur * ndf, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(n_cur * ndf),
                nn.LeakyReLU(0.2, True)
            ]
            n_prev = n_cur
        n_cur = min(2 ** n_layers, 8)
        layers += [
            nn.Conv2d(n_prev * ndf, n_cur * ndf, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(n_cur * ndf),
            nn.LeakyReLU(0.2, True)
        ]
        layers += [
            nn.Conv2d(n_cur * ndf, 1, kernel_size=4, stride=1, padding=1)
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == '__main__':
    unet = UNetGenerator(1, 1)
    print(summary(unet, input_size=(1, 256, 256), device='cpu'))
    discriminator = Discriminator(1)
    print(summary(discriminator, input_size=(1, 256, 256), device='cpu'))
