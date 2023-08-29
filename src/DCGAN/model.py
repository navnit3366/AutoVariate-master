import torch 
import torch.nn as nn


class Disc(nn.Module):
    def __init__(self, channels, features):
        super(Disc, self).__init__()
        self.disc = nn.Sequential(
          nn.Conv2d(channels, features, kernel_size=4, stride=2, padding=1, bias=False),
          nn.LeakyReLU(0.2, inplace=True),
          self._block(features, features*2, 4, 2, 1),
          self._block(features*2, features*4, 4, 2, 1),
          self._block(features*4, features*8, 4, 2, 1),
          nn.Conv2d(features*8, 1, kernel_size=4, stride=2, padding=0, bias=False),
          nn.Sigmoid()
        ) 
    
    def _block(self, input, output, k_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(input, output, k_size, stride, padding, bias=False),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
    def forward(self, x):
        return self.disc(x)
    
    def set_lr_rate(self, lr):
        self.lr = lr
        return self.lr
    
class Generator(nn.Module):
    def __init__(self, noise_dim, channels, features):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            self._block(noise_dim, features*16, 4, 1, 0),
            self._block(features*16, features*8, 4, 2, 1),
            self._block(features*8, features*4, 4, 2, 1),
            self._block(features*4, features*2, 4, 2, 1),
            nn.ConvTranspose2d(features*2, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def _block(self, input,output,k_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, k_size, stride, padding, bias=False),
            nn.BatchNorm2d(output),
            nn.ReLU()
        )
    def forward(self, x):
        return self.gen(x)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            

    
    
def main():
    N, in_channels, H, W = 8, 3, 64, 64
    z_dim = 100
    x = torch.randn((N, in_channels, H, W))
    disc = Disc(in_channels, 8)
    initialize_weights(disc)
    assert disc(x).shape == (N, 1, 1, 1), "Discriminator test failed"
    gen = Generator(z_dim, in_channels, 8)
    initialize_weights(gen)
    z = torch.randn((N, z_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"