import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import Disc, Generator, initialize_weights
import sys
sys.path.append('/Users/arihanttripathi/Documents/AutoVariateGithub')
from AutoVariate.utils import util
import os


util_class = util.auto_util()

class Train:
    
    def __init__(self, lr=2e-4, batch_size=128, image_size=64, channels=1, features_disc=64, features_gen=64, z_dim=100, num_epochs=100):
        self.lr = lr
        self.batch_size = batch_size
        self.image_size = image_size
        self.channels = channels
        self.features_disc = features_disc
        self.features_gen = features_gen
        self.z_dim = z_dim
        self.name = os.path.basename(__file__)
        self.num_epochs = num_epochs
        self.dataset = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def transform_set(self):
        transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5 for _ in range(self.channels)], [0.5 for _ in range(self.channels)]
            ),
        ])
        return transform
    
    def set_dataset(self, dataset=None):
        if dataset == None:
            self.dataset = datasets.MNIST(root="dataset/", train=True, transform=self.transform_set(), download=True)
            self.name = str(util_class.list_sub_dir("dataset/")[0])
        else:
            self.dataset = dataset
            self.name = str(util_class.list_sub_dir("dataset/")[0])
        return self.dataset
    
    def return_class_attributes(self):
        print(f"lr: {self.lr}, batch_size: {self.batch_size}, image_size: {self.image_size}, channels: {self.channels}, features_disc: {self.features_disc}, features_gen: {self.features_gen}, z_dim: {self.z_dim}, num_epochs: {self.num_epochs}, dataset: {self.dataset}, device: {self.device}")
    
    def init_loader(self, dataset, batch_size=128):
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        return loader
    def init_gen(self):
        gen = Generator(self.z_dim, self.channels, self.features_gen).to(self.device)
        return gen
    def init_disc(self):
        disc = Disc(self.channels, self.features_disc).to(self.device)
        return disc
    def set_optimizer(self, model):
        opt = optim.Adam(model.parameters(), lr=self.lr, betas=(0.5, 0.999))
        return opt
    def set_loss(self, loss=nn.BCELoss()):
        return loss
    def sample_noise(self, n_samples=32, z_dim=100, device="cuda" if torch.cuda.is_available() else "cpu"):
        return torch.randn(n_samples, z_dim, 1, 1).to(device)


def main():
    t = Train()
    t.return_class_attributes()
    loader = t.init_loader(t.set_dataset(), t.batch_size)
    gen = t.init_gen()
    disc = t.init_disc()
    initialize_weights(gen)
    initialize_weights(disc)
    opt_gen = t.set_optimizer(gen)
    opt_disc = t.set_optimizer(disc)
    criterion = t.set_loss()
    fixed_noise = t.sample_noise()
    writer_real = SummaryWriter(f"logs/real")
    writer_fake = SummaryWriter(f"logs/fake")
    step = 0
    gen.train()
    disc.train()
    for epoch in (range(t.num_epochs)):
        for batch_idx, (real, _) in enumerate(loader):
            real = real.to(t.device)
            noise = torch.randn(t.batch_size, t.z_dim, 1, 1).to(t.device)
            fake = gen(noise)
            disc_real = disc(real).reshape(-1)
            loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
            disc_fake = disc(fake).reshape(-1)
            loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            disc.zero_grad()
            loss_disc.backward(retain_graph=True)
            opt_disc.step()
            
            output = disc(fake).reshape(-1)
            loss_gen = criterion(output, torch.ones_like(output))
            gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch}/{t.num_epochs}] Batch {batch_idx}/{len(loader)} \
                        Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
                )
                
                with torch.no_grad():
                    fake = gen(fixed_noise)
                    grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                    grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                    writer_real.add_image("Real", grid_real, global_step=step)
                    writer_fake.add_image("Fake", grid_fake, global_step=step)
                    
                step += 1
            
        

main()