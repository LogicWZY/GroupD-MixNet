

import torch
import torch.nn as nn
from torch.nn import functional as F



def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

class DySample(nn.Module):
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h],indexing='ij')).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h],indexing='ij')
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.reshape(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)  # view --> reshape
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)

class PatchEmbed(nn.Module):
    def __init__(self,
                 in_chs,
                 embed_dim,
                 patch_size,
                 stride,
                 padding,              
                ):
        super(PatchEmbed,self).__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels=in_chs, out_channels=embed_dim, kernel_size=patch_size, stride=stride, padding=padding),
            nn.BatchNorm2d(embed_dim),
            )

    def forward(self, x):
        return self.proj(x)
    
class Conv_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(Conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x
    
class DynamicConv2d(nn.Module):
    def __init__(self,
                 dim,
                 kernel_size=7, # 
                 reduction_ratio=4,
                 num_groups=4,
                 bias=True):
        super(DynamicConv2d, self).__init__()
        self.num_groups = num_groups
        self.K = kernel_size
        self.bias_type = bias   
        self.weight = nn.Parameter(torch.empty(num_groups, dim, kernel_size, kernel_size), requires_grad=True)       
        self.pool = nn.AdaptiveAvgPool2d(output_size=(kernel_size, kernel_size))       
        self.proj = nn.Sequential(
            nn.Conv2d(dim,  dim//reduction_ratio, kernel_size=1 ),
            nn.GELU(),
            nn.BatchNorm2d(dim//reduction_ratio),
            nn.Conv2d(dim//reduction_ratio, dim*num_groups, kernel_size=1),)

        if bias:
            self.bias = nn.Parameter(torch.empty(num_groups, dim), requires_grad=True)
        else:
            self.bias = None

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.trunc_normal_(self.weight, std=0.02)
        if self.bias is not None:
            nn.init.trunc_normal_(self.bias, std=0.02)
    
    def forward(self, x):

        b, c, h, w = x.shape

        # x1, x2, x3, x4 = torch.chunk(x, chunks=4, dim=1)
        a = self.pool(x)  # b , c ,3,3
        proj_a = self.proj(a)
        re_a = proj_a.reshape(b, self.num_groups, c ,self.K, self.K)
        scale = torch.softmax(re_a, dim=1)  # b, 1, c, 3,3
    
        weight = scale * self.weight.unsqueeze(0)
        weight = torch.sum(weight, dim=1, keepdim=False)
        weight = weight.reshape(-1, 1, self.K, self.K)  # b*c ,1, k,k

        if self.bias is not None:
            bisa_scale = self.proj(torch.mean(x, dim=[-2, -1], keepdim=True))
            # bisa_scale = torch.mean(proj_a, dim=[-2, -1], keepdim=True)

            scale = torch.softmax(bisa_scale.reshape(b, self.num_groups, c), dim=1)
            bias = scale * self.bias.unsqueeze(0)
            bias = torch.sum(bias, dim=1).flatten(0)
        else:
            bias = None

        x = F.conv2d(x.reshape(1, -1, h, w),
                     weight=weight,
                     padding=self.K//2,
                     groups=b*c,
                     bias=bias)
    
        return x.reshape(b, c, h,w)
    
class FFN(nn.Module):
    def __init__(self, dim,
                 ffn_dim,
                 ):
        super().__init__()

        self.fc1 = nn.Sequential(
            nn.Conv2d(dim, ffn_dim, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(ffn_dim)

        )
        self.dewconv = nn.Conv2d(ffn_dim, ffn_dim, kernel_size=3, padding=1, groups=ffn_dim)
        self.act_layer = nn.GELU()
        self.norm = nn.BatchNorm2d(ffn_dim)
        self.fc2 = nn.Sequential(
            nn.Conv2d(ffn_dim, dim, kernel_size=1),
            nn.BatchNorm2d(dim)
        )
        self.drop = nn.Dropout(0) #

    def forward(self, x):

        x = self.fc1(x)
        x = self.dewconv(x)+x
        x = self.norm(self.act_layer(x))
        x = self.drop(x)


        x = self.drop(self.fc2(x))
        return x

class GroupDynamicBlock(nn.Module):
    def __init__(self,dim, 
                 num_groups,
                 red_ratio,
                 norm_layer = nn.GroupNorm):
        super( GroupDynamicBlock, self).__init__()

        self.pos_embed = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = norm_layer(num_groups=1, num_channels=dim)

        self.dy_conv1 = DynamicConv2d(dim//4, 1, red_ratio, num_groups)  # in_dim, k, ratio, num_groups
        self.dy_conv3 = DynamicConv2d(dim//4, 3, red_ratio, num_groups)
        self.dy_conv5 = DynamicConv2d(dim//4, 5, red_ratio, num_groups)
        self.dy_conv7 = DynamicConv2d(dim//4, 7, red_ratio, num_groups)
      
        self.proj = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.BatchNorm2d(dim),
            nn.Conv2d(dim, dim*4, kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim*4),
            nn.Conv2d(dim*4, dim, kernel_size=1),
            nn.BatchNorm2d(dim),)

        self.norm2 = norm_layer(num_groups=1,num_channels=dim)
        self.ffn = FFN(dim, dim//16)


    def forward(self, x):
        res = x
        x = self.pos_embed(x) + res
        x1, x2, x3, x4 = torch.chunk(self.norm1(x), chunks=4, dim=1)
        
        y1 = self.dy_conv1(x1)
        y2 = self.dy_conv3(x2)
        y3 = self.dy_conv5(x3)
        y4 = self.dy_conv7(x4)

        y = torch.cat([y1, y2, y3, y4],dim=1)

        y_ = self.proj(y) + res
        y = self.ffn(self.norm2(y_)) + y_

        return y

    
class MLP(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = self.proj(x)
        return x
    
class BottConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels, kernel_size, stride=1, padding=0, bias=True):
        super(BottConv, self).__init__()
        self.pointwise_1 = nn.Conv2d(in_channels, mid_channels, 1, bias=bias)
        self.depthwise = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride, padding, groups=mid_channels, bias=False)
        self.pointwise_2 = nn.Conv2d(mid_channels, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.pointwise_1(x)
        x = self.depthwise(x)
        x = self.pointwise_2(x)
        return x


class IGB(nn.Module):
    def __init__(self, in_channels, 
                 norm_layer = nn.GroupNorm):
        super().__init__()

        self.block1 = nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1),
            norm_layer(num_channels=in_channels, num_groups=in_channels//16),
            nn.ReLU()

        )

        self.block2 =nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 3, 1, 1),
            norm_layer(num_channels=in_channels, num_groups=in_channels//16),
            nn.ReLU()

        )

        self.block3 =nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0),
            norm_layer(num_channels=in_channels, num_groups=in_channels//16),
            nn.ReLU()

        )

        self.block4 =nn.Sequential(
            BottConv(in_channels, in_channels, in_channels//8, 1, 1, 0),
            norm_layer(num_channels=in_channels, num_groups=16),
            nn.ReLU()

        )

    def forward(self, x):
        x_residual = x

        x1 = self.block1(x)
        x1 = self.block2(x1)
        x2 = self.block3(x)

        x = x1*x2

        x = self.block4(x)

        return x + x_residual

class GroupDynamic_Mix(nn.Module):
    def __init__(self, in_c,
                 embed_dim,
                 num_class,
                 dims = [64,128,320,512],
                 depth = [2, 4, 8, 4], 
                 ):
        super(GroupDynamic_Mix,self).__init__()

        self.stem = PatchEmbed(in_chs=in_c, embed_dim=dims[0],patch_size=3, stride=2, padding=1)

        self.embed_dim= embed_dim  

        self.conv_layer1 = Conv_block(ch_in=dims[0], ch_out=dims[1])
        self.GroupDy_layer1 = nn.ModuleList()
        for _ in range(depth[0]):
            dynamic_block = GroupDynamicBlock(dim=dims[0],num_groups=2,red_ratio=4)
            self.GroupDy_layer1.append(dynamic_block)
      
        self.conv_layer2 = Conv_block(ch_in=dims[1], ch_out=dims[2])
        self.GroupDy_layer2 = nn.ModuleList()
        for _ in range(depth[1]):
            dynamic_block = GroupDynamicBlock(dim=dims[1],num_groups=3,red_ratio=4)
            self.GroupDy_layer2.append(dynamic_block)

        self.conv_layer3 = Conv_block(ch_in=dims[2], ch_out=dims[3])
        self.GroupDy_layer3 = nn.ModuleList()
        for _ in range(depth[2]):
            dynamic_block = GroupDynamicBlock(dim=dims[2],num_groups=4,red_ratio=8)
            self.GroupDy_layer3.append(dynamic_block)
       
        self.GroupDy_layer4 = nn.ModuleList()  
        for _ in range(depth[3]):
            dynamic_block = GroupDynamicBlock(dim=dims[3],num_groups=3,red_ratio=16)
            self.GroupDy_layer4.append(dynamic_block)
    

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.mlp_1 = MLP(input_dim=dims[0], embed_dim=embed_dim)
        self.mlp_2 = MLP(input_dim=dims[1], embed_dim=embed_dim)
        self.mlp_3 = MLP(input_dim=dims[2], embed_dim=embed_dim)
        self.mlp_4 = MLP(input_dim=dims[3], embed_dim=embed_dim)
    
        self.up_dysample_1 = DySample(in_channels=embed_dim, scale=2)
        self.up_dysample_2 = DySample(in_channels=embed_dim, scale=4)
        self.up_dysample_3 = DySample(in_channels=embed_dim, scale=8)
        self.up_dysample_4 = DySample(in_channels=embed_dim, scale=16)
        self.mfs = IGB(embed_dim*4)
        self.linear_fuse = BottConv(embed_dim*4, embed_dim, embed_dim//8, kernel_size=1, padding=0, stride=1)


        self.dropout = nn.Dropout(p=0.1)

        self.linear_pred = nn.Sequential(
            BottConv(embed_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, num_class, kernel_size=1))

        self.linear_pred_4 = nn.Sequential(
            BottConv(embed_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, num_class, kernel_size=1))
        self.linear_pred_3 = nn.Sequential(
            BottConv(embed_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, num_class, kernel_size=1))
        self.linear_pred_2 = nn.Sequential(
            BottConv(embed_dim, 1, 1, kernel_size=1),
            nn.Conv2d(1, num_class, kernel_size=1))


    def forward(self, x):
        x1 = self.stem(x)

        for dy_conv in self.GroupDy_layer1:
            x1 = dy_conv(x1) # 8, 64, 128, 128]
   
        
        x2 = self.Maxpool(self.conv_layer1(x1))
        for dy_conv in self.GroupDy_layer2:
            x2 = dy_conv(x2)

        x3 = self.Maxpool(self.conv_layer2(x2))
        for dy_conv in self.GroupDy_layer3:
            x3 = dy_conv(x3)

        x4 = self.Maxpool(self.conv_layer3(x3))
        for dy_conv in self.GroupDy_layer4:
            x4 = dy_conv(x4)

        b, c_4, h_4, w_4 = x4.shape  # b, c, h*w ---> b, h*w, c
   
        fx4 = self.mlp_4(x4.reshape(b, c_4, h_4*w_4).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embed_dim,h_4, w_4 )
        x_4 = self.up_dysample_4(fx4)
        out_4 = self.linear_pred_4(x_4)

        b, c_3, h_3, w_3 = x3.shape
        fx3 = self.mlp_3(x3.reshape(b, c_3, h_3*w_3).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embed_dim,h_3, w_3 )
        x_3 = self.up_dysample_3(fx3)
        out_3 = self.linear_pred_4(x_3)
                      
        b, c_2, h_2, w_2 = x2.shape
     
        fx2 = self.mlp_2(x2.reshape(b, c_2, h_2*w_2).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embed_dim,h_2, w_2 )
        x_2 = self.up_dysample_2(fx2)
        out_2 = self.linear_pred_4(x_2)

        b, c_1, h_1, w_1 = x1.shape
     
        fx1 = self.mlp_1(x1.reshape(b, c_1, h_1*w_1).permute(0, 2, 1)).permute(0, 2, 1).reshape(b, self.embed_dim,h_1, w_1 )
        x_1 = self.up_dysample_1(fx1)

        fuse = torch.cat([x_4, x_3, x_2, x_1],dim=1)

        y = self.mfs(fuse)

        out_ = self.dropout(self.linear_fuse(y))

        out = self.linear_pred(out_)

        return out , out_4, out_3, out_2
     
if __name__ =='__main__':
    f = torch.randn(8,3,224,224)
    n = GroupDynamic_Mix(3,32,1)
    # print(n)
    o = n(f)
    print("__main__", o[0].shape)


        