import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

class conv_block(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(conv_block,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )


    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.up(x)
        return x

class Recurrent_block(nn.Module):
    def __init__(self,ch_out,t=2):
        super(Recurrent_block,self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out,ch_out,kernel_size=3,stride=1,padding=1,bias=True),
		    nn.BatchNorm2d(ch_out),
			nn.ReLU(inplace=True)
        )

    def forward(self,x):
        for i in range(self.t):

            if i==0:
                x1 = self.conv(x)
            
            x1 = self.conv(x+x1)
        return x1
        
class RRCNN_block(nn.Module):
    def __init__(self,ch_in,ch_out,t=2):
        super(RRCNN_block,self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out,t=t),
            Recurrent_block(ch_out,t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in,ch_out,kernel_size=1,stride=1,padding=0)

    def forward(self,x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x+x1


class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3,stride=1,padding=1,bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self,x):
        x = self.conv(x)
        return x

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi


class U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2U_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2U_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)
        

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1



class AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1):
        super(AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

        self.Conv1 = conv_block(ch_in=img_ch,ch_out=64)
        self.Conv2 = conv_block(ch_in=64,ch_out=128)
        self.Conv3 = conv_block(ch_in=128,ch_out=256)
        self.Conv4 = conv_block(ch_in=256,ch_out=512)
        self.Conv5 = conv_block(ch_in=512,ch_out=1024)

        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_conv5 = conv_block(ch_in=1024, ch_out=512)

        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_conv4 = conv_block(ch_in=512, ch_out=256)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_conv3 = conv_block(ch_in=256, ch_out=128)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_conv2 = conv_block(ch_in=128, ch_out=64)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.Conv1(x)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.Conv5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)        
        d5 = self.Up_conv5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


class R2AttU_Net(nn.Module):
    def __init__(self,img_ch=3,output_ch=1,t=2):
        super(R2AttU_Net,self).__init__()
        
        self.Maxpool = nn.MaxPool2d(kernel_size=2,stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=img_ch,ch_out=64,t=t)

        self.RRCNN2 = RRCNN_block(ch_in=64,ch_out=128,t=t)
        
        self.RRCNN3 = RRCNN_block(ch_in=128,ch_out=256,t=t)
        
        self.RRCNN4 = RRCNN_block(ch_in=256,ch_out=512,t=t)
        
        self.RRCNN5 = RRCNN_block(ch_in=512,ch_out=1024,t=t)


        self.Up5 = up_conv(ch_in=1024,ch_out=512)
        self.Att5 = Attention_block(F_g=512,F_l=512,F_int=256)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512,t=t)
        
        self.Up4 = up_conv(ch_in=512,ch_out=256)
        self.Att4 = Attention_block(F_g=256,F_l=256,F_int=128)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256,t=t)
        
        self.Up3 = up_conv(ch_in=256,ch_out=128)
        self.Att3 = Attention_block(F_g=128,F_l=128,F_int=64)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128,t=t)
        
        self.Up2 = up_conv(ch_in=128,ch_out=64)
        self.Att2 = Attention_block(F_g=64,F_l=64,F_int=32)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64,t=t)

        self.Conv_1x1 = nn.Conv2d(64,output_ch,kernel_size=1,stride=1,padding=0)


    def forward(self,x):
        # encoding path
        x1 = self.RRCNN1(x)

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # decoding + concat path
        d5 = self.Up5(x5)
        x4 = self.Att5(g=d5,x=x4)
        d5 = torch.cat((x4,d5),dim=1)
        d5 = self.Up_RRCNN5(d5)
        
        d4 = self.Up4(d5)
        x3 = self.Att4(g=d4,x=x3)
        d4 = torch.cat((x3,d4),dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        x2 = self.Att3(g=d3,x=x2)
        d3 = torch.cat((x2,d3),dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        x1 = self.Att2(g=d2,x=x1)
        d2 = torch.cat((x1,d2),dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        return d1


import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d


class SelfAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.query = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.key = nn.Conv2d(in_ch, in_ch // 8, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.query(x).view(B, -1, H * W)  # B, C/8, N
        k = self.key(x).view(B, -1, H * W)  # B, C/8, N
        v = self.value(x).view(B, -1, H * W)  # B, C, N

        attn = torch.bmm(q.permute(0, 2, 1), k)  # B, N, N
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, N
        out = out.view(B, C, H, W)
        return self.gamma * out + x


class CrossAttention(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.query = nn.Conv2d(ch, ch // 8, 1)
        self.key = nn.Conv2d(ch, ch // 8, 1)
        self.value = nn.Conv2d(ch, ch, 1)

    def forward(self, f1, f2):
        B, C, H, W = f1.shape
        q = self.query(f1).view(B, -1, H * W)  # B, C/8, N
        k = self.key(f2).view(B, -1, H * W)  # B, C/8, N
        v = self.value(f2).view(B, -1, H * W)  # B, C, N

        attn = torch.bmm(q.permute(0, 2, 1), k)  # B, N, N
        attn = F.softmax(attn, dim=-1)
        out = torch.bmm(v, attn.permute(0, 2, 1))  # B, C, N
        out = out.view(B, C, H, W)
        return out


class DBAM(nn.Module):
    def __init__(self, in_ch):
        super().__init__()

        self.conv_branch = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.GroupNorm(1, in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1)
        )

        self.self_attn = SelfAttention(in_ch)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.GroupNorm(1, in_ch)
        self.mlp = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1)
        )
        # Cross Attention
        self.cross_attn = CrossAttention(in_ch)

    def forward(self, x):
        # branch 1
        f_f = self.conv_branch(x)

        # branch 2
        attn_out = self.self_attn(x)
        pooled = self.pool(attn_out)
        f_l = self.mlp(self.norm(pooled).expand_as(x))

        out = self.cross_attn(f_f, f_l)
        return out


# ----------------------
#  STN
# ----------------------
class STN(nn.Module):
    def __init__(self, in_channels):
        super(STN, self).__init__()

        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(64, 6)
        self.fc.weight.data.zero_()
        self.fc.bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x).view(x.size(0), -1)
        theta = self.fc(xs).view(-1, 2, 3)  # [B,2,3]
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


# ----------------------
# ELK Block
# ----------------------
class ELKBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ELKBlock, self).__init__()
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv1x5 = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 5), padding=(0, 2))
        self.conv5x1 = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 1), padding=(2, 0))
        self.conv1x1 = nn.Conv2d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        out1 = self.conv3x3(x)
        out2 = self.conv1x5(x)
        out3 = self.conv5x1(x)
        out = torch.cat([out1, out2, out3], dim=1)
        out = self.conv1x1(out)
        return out


# ----------------------
# SDIM Block
# ----------------------
class SDIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SDIM, self).__init__()
        self.stn = STN(in_channels)
        self.dcn = DeformConv2d(in_channels, in_channels, kernel_size=3, padding=1)

        self.weight_gen = nn.Conv2d(in_channels * 2, 2, kernel_size=1)

        # ELK
        self.elk = ELKBlock(in_channels, out_channels)

    def forward(self, x, offset=None):
        Fs = self.stn(x)
        Fd = self.dcn(x, offset) if offset is not None else self.dcn(x, torch.zeros(x.size(0), 18, x.size(2), x.size(3),
                                                                                    device=x.device))

        fusion = torch.cat([Fs, Fd], dim=1)
        weights = F.softmax(self.weight_gen(fusion), dim=1)
        Ws, Wd = weights[:, 0:1], weights[:, 1:2]

        out = Fs * Ws + Fd * Wd

        # ELK
        out = self.elk(out)
        return out


# # ----------------------

# # ----------------------
# class EncoderWithSDIM(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super(EncoderWithSDIM, self).__init__()

#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, 3, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#         self.pool = nn.MaxPool2d(2)
#
#     def forward(self, x):

#         x = self.conv(x)
#         p = self.pool(x)
#         return x, p


class conv_block(nn.Module):
    def __init__(self, ch_in, ch_out, stride=2):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.ConvTranspose2d(ch_in, ch_out, kernel_size=2, stride=2)

    def forward(self, x):
        return self.up(x)


class U_Net_4layer_SDIM(nn.Module):
    def __init__(self, img_ch=3, output_ch=1):
        super(U_Net_4layer_SDIM, self).__init__()

        self.Enc1 = EncoderWithSDIM(in_channels=img_ch, out_channels=32)
        self.Enc2 = EncoderWithSDIM(in_channels=32, out_channels=64)
        self.Enc3 = EncoderWithSDIM(in_channels=64, out_channels=128)
        self.Enc4 = EncoderWithSDIM(in_channels=128, out_channels=256)

        self.Up4 = up_conv(ch_in=256, ch_out=128)
        self.Up_conv4 = conv_block(ch_in=256, ch_out=128)

        self.Up3 = up_conv(ch_in=128, ch_out=64)
        self.Up_conv3 = conv_block(ch_in=128, ch_out=64)

        self.Up2 = up_conv(ch_in=64, ch_out=32)
        self.Up_conv2 = conv_block(ch_in=64, ch_out=32)

        self.Conv_1x1 = nn.Conv2d(32, output_ch, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1, p1 = self.Enc1(x)
        x2, p2 = self.Enc2(p1)
        x3, p3 = self.Enc3(p2)
        x4, p4 = self.Enc4(p3)

        d4 = self.Up4(x4)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        out = self.Conv_1x1(d2)
        return out


class EncoderWithSDIM(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.sdim = SDIM(in_channels, in_channels)
        self.conv = conv_block(in_channels, out_channels)

    def forward(self, x):
        x = self.sdim(x)
        x = self.conv(x)
        return x


class PyramidPooling(nn.Module):
    def __init__(self, in_channels=512, pool_sizes=(2, 4, 8)):
        super(PyramidPooling, self).__init__()
        out_channels = in_channels // len(pool_sizes)

        self.paths = nn.ModuleList()
        for ps in pool_sizes:
            self.paths.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(ps),
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.GroupNorm(2, out_channels),
                    nn.ReLU(inplace=True)
                )
            )

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_channels + len(pool_sizes) * out_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(2, in_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, c, h, w = x.size()
        feats = [x]

        for path in self.paths:
            out = path(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            feats.append(out)

        out = torch.cat(feats, dim=1)
        out = self.conv_out(out)
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels=None):

        super(DecoderBlock, self).__init__()

        if out_channels is None:
            out_channels = in_channels // 2

        self.upsample = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=2,
            stride=2
        )

        self.conv_block = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channels + skip_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):

        x = self.upsample(x)  # [batch, out_channels, 2H, 2W]
        skip = self.upsample(skip)

        if x.shape[2:] != skip.shape[2:]:
            skip = nn.functional.interpolate(skip, size=x.shape[2:], mode='bilinear', align_corners=True)

        x = torch.cat([x, skip], dim=1)  # [batch, out_channels + skip_channels, 2H, 2W]

        x = self.conv_block(x)  # [batch, out_channels, 2H, 2W]

        return x


# ---- Circular Convolution ----
class CircularConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super(CircularConv2d, self).__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride,
            padding=0, bias=False
        )
        self.pad = padding

    def forward(self, x):
        # circular padding
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='circular')
        return self.conv(x)


# ---- Sobel Edge Detector ----
class SobelLayer(nn.Module):
    def __init__(self, in_channels):
        super(SobelLayer, self).__init__()
        # Sobel kernels
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]], dtype=torch.float32)
        ky = torch.tensor([[-1., -2., -1.],
                           [0., 0., 0.],
                           [1., 2., 1.]], dtype=torch.float32)

        # create weight shape [in_channels, 1, 3, 3] for grouped conv
        weight_x = kx.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)  # (C,1,3,3)
        weight_y = ky.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1)

        # register as buffers so they're moved with .to(device) and not trained
        self.register_buffer('weight_x', weight_x)
        self.register_buffer('weight_y', weight_y)
        self.in_channels = in_channels

    def forward(self, x):
        # x: [B, C, H, W]
        # grouped conv: each input channel convolved with corresponding kernel
        edge_x = F.conv2d(x, self.weight_x, padding=1, groups=self.in_channels)
        edge_y = F.conv2d(x, self.weight_y, padding=1, groups=self.in_channels)

        # magnitude per channel -> shape [B, C, H, W]
        edge = torch.sqrt(edge_x ** 2 + edge_y ** 2 + 1e-12)

        # reduce to single-channel edge map (you may also use max instead of mean)
        edge_single = edge.mean(dim=1, keepdim=True)  # [B,1,H,W]

        return edge_single


# ---- Circular Convolution Block ----
class CircularBlock(nn.Module):
    def __init__(self, in_ch=32, growth_ch=32, num_layers=5):
        super(CircularBlock, self).__init__()
        self.layers = nn.ModuleList()
        cur_ch = in_ch
        for i in range(num_layers):
            self.layers.append(CircularConv2d(cur_ch, growth_ch, kernel_size=3, padding=1))
            cur_ch += growth_ch  # Dense-like connection

        self.gmp = nn.AdaptiveMaxPool2d(1)  # Global Max Pool
        self.sobel = SobelLayer(in_ch)
        self.fuse = nn.Conv2d(cur_ch, in_ch, kernel_size=1)
        self.out_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, 1, 1)  # segmentation output
        )

    def forward(self, x):
        features = [x]
        out = x
        for layer in self.layers:
            new_feat = layer(torch.cat(features, dim=1))
            features.append(new_feat)
            out = out + new_feat

        out = torch.cat(features, dim=1)

        gmp_feat = self.gmp(out)
        gmp_feat = gmp_feat.expand_as(out)

        sobel_feat = self.sobel(x)
        sobel_feat = sobel_feat.expand_as(out)

        out = out + gmp_feat + sobel_feat
        out = self.fuse(out)
        out = self.out_conv(out)  # segmentation output

        return out


class DABC_UNet_4enc_4dec(nn.Module):
    def __init__(self, img_ch=1, output_ch=1):
        super().__init__()
        self.conv33 = nn.Conv2d(img_ch, 32, kernel_size=3, stride=1, padding=1)  # 32, H/1
        # --- Encoder: En-1..En-4 ---

        self.En1 = EncoderWithSDIM(32, 64)  # 64, H/2
        self.En2 = EncoderWithSDIM(64, 128)  # 128, H/4
        self.En3 = EncoderWithSDIM(128, 256)  # 256, H/8
        self.En4 = EncoderWithSDIM(256, 512)  # 512, H/8

        self.ppm = PyramidPooling()
        # --- Decoder: De-1..De-4 ---
        self.De4 = DecoderBlock(512, 256)
        self.De3 = DecoderBlock(256, 128)
        self.De2 = DecoderBlock(128, 64)
        self.De1 = DecoderBlock(64, 32)

        self.dbam1 = DBAM(64)
        self.dbam2 = DBAM(128)
        self.dbam3 = DBAM(256)
        self.dbam4 = DBAM(512)

        self.bam = CircularBlock(32)

    def forward(self, x):
        x_32 = self.conv33(x)
        # ----- Encoder -----
        x1 = self.En1(x_32)
        x2 = self.En2(x1)
        x3 = self.En3(x2)
        x4 = self.En4(x3)

        s1 = self.dbam1(x1)
        s2 = self.dbam2(x2)
        s3 = self.dbam3(x3)
        s4 = self.dbam4(x4)

        btm = self.ppm(x4)  # 512, H/16

        d4 = self.De4(btm, s4)
        d3 = self.De3(d4, s3)
        d2 = self.De2(d3, s2)
        d1 = self.De1(d2, s1)

        out = self.bam(d1)
        return out


#  quick check
if __name__ == "__main__":
    net = DABC_UNet_4enc_4dec(img_ch=1, output_ch=1)
    x = torch.randn(4, 1, 256, 256)
    y = net(x)
    print(y.shape)  # -> torch.Size([4, 1, 256, 256])
