import torch
import torch.nn as nn
import torch.nn.functional as F
from .deform import DeformConv2d

# Dynamic Feature Enhancement Module
class DFEM(nn.Module):
  def __init__(self, nChannels):
    super(DFEM, self).__init__()

    self.conv_x = DeformConv2d(nChannels, nChannels)
    self.conv_y = DeformConv2d(nChannels, nChannels)
    self.conv_o = nn.Conv2d(nChannels*2, nChannels, kernel_size=1)

  def forward(self, x):

    conv_x = self.conv_x(x)
    conv_y = self.conv_y(x)
    xy = torch.cat([conv_x,conv_y],dim=1)
    out = self.conv_o(xy) + x

    return out


def default_conv(in_channels, out_channels, kernel_size):
    return nn.Conv2d(in_channels, out_channels, kernel_size,padding=(kernel_size//2), bias=True)



class PDU(nn.Module):  # physical block
    def __init__(self, channel):
        super(PDU, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ka = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.td = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel // 8, 3),
            nn.ReLU(inplace=True),
            default_conv(channel // 8, channel, 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        a = self.avg_pool(x)
        a = self.ka(a)
        t = self.td(x)
        j = torch.mul((1 - t), a) + torch.mul(t, x)
        return j


class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
                nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y



class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
                nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y


class Dehazeblock(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(Dehazeblock, self).__init__()
        self.conv1=conv(dim, dim, kernel_size)
        self.act1=nn.ReLU(inplace=True)
        self.calayer=CALayer(dim)
        self.pdu=PDU(dim)

    def forward(self, x):
        res=self.act1(self.conv1(x))
        res=res+x
        res=self.calayer(res)
        res=self.pdu(res)
        res += x
        return res


# Adaptive Feature Fusion Module
class AFFM(nn.Module):
    def __init__(self, m=-0.80,channel=None):
        super(AFFM, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()
        self.calayer = CALayer(channel)
        self.pdulayer = PDU(channel)

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)

        fea1=self.pdulayer(self.calayer(fea1))
        fea2=self.pdulayer(self.calayer(fea2))

        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))

        return out


class DehazeNet(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super(DehazeNet, self).__init__()

        # Downsampling
        self.down1 = nn.Sequential( nn.ReflectionPad2d(3),
                                    nn.Conv2d(input_nc, 64, 7),
                                    nn.ReLU(inplace=True))

        self.down2 = nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))

        self.down3 = nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                   nn.ReLU(inplace=True))


        self.Dehazeblock = Dehazeblock(conv=default_conv,dim=256,kernel_size=3)

        self.dfem = DFEM(256)

        # Upsampling
        self.up1 = nn.Sequential(nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True))

        self.up2 = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
                                 nn.ReLU(inplace=True))

        self.up3 = nn.Sequential(nn.ReflectionPad2d(3),
                                 nn.Conv2d(64, output_nc, 7),
                                 nn.Tanh())

        self.AFFM1 = AFFM(m=-1,channel=256)
        self.AFFM2 = AFFM(m=-0.6,channel=128)

    def forward(self, x):

        x_down1 = self.down1(x)         # [bs, 64, 256, 256]
        x_down2 = self.down2(x_down1)   # [bs, 128, 128, 128]
        x_down3 = self.down3(x_down2)   # [bs, 256, 64, 64]

        x1 = self.Dehazeblock(x_down3)  # [bs, 256, 64, 64]
        x2 = self.Dehazeblock(x1)       # [bs, 256, 64, 64]
        x3 = self.Dehazeblock(x2)       # [bs, 256, 64, 64]
        x4 = self.Dehazeblock(x3)       # [bs, 256, 64, 64]
        x5 = self.Dehazeblock(x4)       # [bs, 256, 64, 64]
        x6 = self.Dehazeblock(x5)       # [bs, 256, 64, 64]
        x7 = self.Dehazeblock(x6)       # [bs, 256, 64, 64]
        x8 = self.Dehazeblock(x7)       # [bs, 256, 64, 64]
        x9 = self.Dehazeblock(x8)       # [bs, 256, 64, 64]

        x9_dfem = self.dfem(x9)

        x_out_affm = self.AFFM1(x_down3, x9_dfem)
        x_up1 = self.up1(x_out_affm)    # [bs, 128, 128, 128]
        x_up1_affm = self.AFFM2(x_down2, x_up1)
        x_up2 = self.up2(x_up1_affm)    # [bs, 64, 256, 256]
        out = self.up3(x_up2)           # [bs,  3, 256, 256]

        return out,x9_dfem


class KTTeacher(nn.Module):
    def __init__(self, T_path=None, requires_grad=False):
        super(KTTeacher, self).__init__()

        self.T = DehazeNet().cuda()
        self.T.load_state_dict(torch.load(T_path))

        print("Load teacher model weight successfully")
        self.l1 = nn.L1Loss().cuda()

        if not requires_grad:
            print("Teacher models requires_grad = False")
            for param in self.T.parameters():
                param.requires_grad = requires_grad

    def forward(self,x,y):

        feats_y = self.T(y)
        feats_y = feats_y[1:]

        loss = 0

        for i in range(len(feats_y)):

            d_ap = self.l1(x[i], feats_y[i].detach())
            loss += d_ap

        return loss




if __name__ =="__main__":
    x=torch.ones(1,3,256,256)
    D=DehazeNet()
    res=D(x)
    print(res.shape)
    print(res)
