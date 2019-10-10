import torch
import fire


class SqueezeNet(torch.nn.Module):
    def __init__(self):
        super(SqueezeNet, self).__init__()
        self.net = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=3, out_channels=96, kernel_size=(7, 7), stride=2),
                torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                torch.nn.ReLU(),
                fire.Fire(in_channels=96, s1x1=16, e1x1=64, e3x3=64),
                fire.Fire(in_channels=128, s1x1=16, e1x1=64, e3x3=64),
                fire.Fire(in_channels=128, s1x1=32, e1x1=128, e3x3=128),
                torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2),
                fire.Fire(in_channels=256, s1x1=32, e1x1=128, e3x3=128),
                fire.Fire(in_channels=256, s1x1=48, e1x1=192, e3x3=192),
                fire.Fire(in_channels=384, s1x1=48, e1x1=192, e3x3=192),
                fire.Fire(in_channels=384, s1x1=64, e1x1=256, e3x3=256),
                torch.nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
                fire.Fire(in_channels=512, s1x1=64, e1x1=256, e3x3=256),
                torch.nn.ReLU(),
                torch.nn.Conv2d(in_channels=512, out_channels=1000, kernel_size=(1, 1)),
                torch.nn.AvgPool2d(kernel_size=(13, 13), stride=1)
                )

    def forward(self, x):
        out = self.net(x)
        return out.view(out.size(0), -1)

if __name__ == '__main__':
    x = torch.randn((10, 3, 224, 224))
    sn = SqueezeNet()
    out = sn(x)
    print (sn)
    print (out.shape)
