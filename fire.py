import torch


class Expand(torch.nn.Module):
    def __init__(self, in_channels, e1_out_channles, e3_out_channles):
        super(Expand, self).__init__()
        self.conv_1x1 = torch.nn.Conv2d(in_channels, e1_out_channles, (1, 1))
        self.conv_3x3 = torch.nn.Conv2d(in_channels, e3_out_channles, (3, 3), padding=1)

    def forward(self, x):
        o1 = self.conv_1x1(x)
        o3 = self.conv_3x3(x3)
        return torch.cat((o1, o3), dim=1)


class Fire(torch.nn.Module):
    """
      Fire module in SqueezeNet
      out_channles = e1x1 + e3x3
      Eg.: input: ?xin_channelsx?x?
           output: ?x(e1x1+e3x3)x?x?
    """
    def __init__(self, in_channels, s1x1, e1x1, e3x3):
        super(Fire, self).__init__()

        # squeeze 
        self.squeeze = torch.nn.Conv2d(in_channels, s1x1, (1, 1))
        self.sq_act = torch.nn.LeakyReLU(0.1)

        # expand
        self.expand = Expand(s1x1, e1x1, e3x3)
        self.ex_act = torch.nn.LeakyReLU(0.1)
        

    def forward(self, x):
        x = self.sq_act(self.squeeze(x))
        x = self.ex_act(self.expand(x))
        return x


def main():
    x = torch.randn((1, 1, 55, 55))
    fire = Fire(1, 16, 64, 64)
    out = fire(x)
    print (fire)
    print (out.shape)

    # print (out)

if __name__ == '__main__':
    main()
