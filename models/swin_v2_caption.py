#%%
import torch
import torch.nn as nn
import numpy as np
import swin_transformer_v2


#%%
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

class Swin_V2_Caption_Encoder(nn.Module):
    def __init__(self, pre_trianed = True, fine_tune = False):
        super(Swin_V2_Caption_Encoder,self).__init__()

        self.model = swin_transformer_v2.SwinTransformerV2(img_size=256,
                    patch_size=4,
                    in_chans=3,
                    num_classes=1000,
                    embed_dim=96,
                    depths=[2, 2, 6, 2],
                    num_heads=[3, 6, 12, 24],
                    window_size=8,
                    mlp_ratio=4.,
                    qkv_bias=True,
                    drop_rate=0.0,
                    drop_path_rate=0.2,
                    ape=False,
                    patch_norm=True,
                    use_checkpoint="swinv2_tiny_patch4_window8_256.pth",
                    pretrained_window_sizes=[0, 0, 0, 0])
    

        checkpoint = torch.load("swinv2_tiny_patch4_window8_256.pth")

        if pre_trianed:
            self.model.load_state_dict(checkpoint['model'], strict=False)

            if fine_tune:
                for param in self.model.parameters():
                    param.requires_grad = True
            else:
                for param in self.model.parameters():
                    param.requires_grad = False

        self.model.head = Identity()

    def forward(self, x):
        return self.model(x)
        
#%%
model = Swin_V2_Caption_Encoder()

# %%
