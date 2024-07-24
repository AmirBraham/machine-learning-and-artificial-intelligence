import torch
import torch.nn as nn

class DownConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,use_act=True):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,padding_mode="reflect"),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        ) 
    def forward(self,x):
        return self.conv(x)


class UpConvBlock(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,stride,padding,use_act=True,**kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,stride=stride,padding=padding,**kwargs),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        ) 
    def forward(self,x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self,channels):
        super().__init__()
        self.block = nn.Sequential(
            UpConvBlock(channels,channels,kernel_size=3,padding=1,stride=1),
            UpConvBlock(channels,channels,kernel_size=3,padding=1,stride=1,use_act=False)
        )
    def forward(self,x):
        return x + self.block(x)
    
    
class Generator(nn.Module):
    def __init__(self,img_channels,num_features=64,num_residuals=6):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels,64,kernel_size=7,stride=1,padding=3,padding_mode="reflect"),
            nn.ReLU(inplace=True),
        )
        self.down_blocks = nn.ModuleList([
            DownConvBlock(num_features,num_features*2,kernel_size=3,stride=2,padding=1),
            DownConvBlock(num_features*2,num_features*4,kernel_size=3,stride=2,padding=1),
            
        ])
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        self.up_blocks = nn.ModuleList([
            UpConvBlock(num_features*4,num_features*2,kernel_size=3,stride=2,padding=1,output_padding=1),
            UpConvBlock(num_features*2,num_features,kernel_size=3,stride=2,padding=1,output_padding=1),
            
        ])
        
        self.last = nn.Conv2d(num_features,img_channels,kernel_size=7,stride=1,padding=3,padding_mode="reflect")
    def forward(self,x):
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.residual_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
        
def test():
    img_channels=3
    size = 128
    x = torch.randn((1,img_channels,size,size))
    
    gen = Generator(img_channels)
    print(gen)
    output = gen(x)
    print(output.shape) # We should get the same size as the input image 
    
if __name__ == "__main__":
    test()