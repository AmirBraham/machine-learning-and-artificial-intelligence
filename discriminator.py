import torch
import torch.nn as nn
from torchviz import make_dot

class Block(nn.Module):
    def __init__(self,in_channels,out_channels,stride,should_use_instance_norm=True):
        super().__init__()
        # in_channels is the number of channels of the image (For RGB , in_channels = 3)
        # out_channels is the number of filters we want
        # Kernel size is always equal to 4 in PATCHGAN Paper
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=4, stride=stride,padding=1,bias=True,padding_mode="reflect"),
            nn.InstanceNorm2d(num_features=out_channels) if should_use_instance_norm else nn.Identity(),
            nn.LeakyReLU()
        )

    def forward(self,x):
        return self.conv(x)
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        layers = [
            Block(in_channels=3,out_channels=64,stride=2,should_use_instance_norm=False),
            Block(in_channels=64,out_channels=128,stride=2),
            Block(in_channels=128,out_channels=256,stride=2),
            Block(in_channels=256,out_channels=512,stride=2),
            Block(in_channels=512,out_channels=1,stride=1),
                ]
        self.model = nn.Sequential(*layers)

    def forward(self,input):
        return torch.sigmoid(self.model(input))
    
    
def test():
    x = torch.randn((1,3,256,256))
    model = Discriminator()
    output =  model(x)
    print(model)
    print(output)
    
if __name__ == "__main__":
    test()