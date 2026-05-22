import torch
from .Unet.unet import UNet
from .Group_Mix.GroupD_Mix import GroupDynamic_Mix




def building_model(args,parser):  
    if args.model == "GroupD_Mix":
       model = GroupDynamic_Mix(in_c=3,embed_dim=32,num_class=1).cuda()
    return model