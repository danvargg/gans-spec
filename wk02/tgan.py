"""Video Generation with TGAN."""
import torch
import torch.nn as nn
import numpy as np
from moviepy.editor import ImageSequenceClip


class TemporalGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # Create a sequential model to turn one vector into 16
        self.model = nn.Sequential(
            nn.ConvTranspose1d(100, 512, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.ConvTranspose1d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.ConvTranspose1d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.ConvTranspose1d(128, 100, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

        # initialize weights according to paper
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose1d:
            nn.init.xavier_uniform_(m.weight, gain=2 ** 0.5)

    def forward(self, x):
        # reshape x so that it can have convolutions done
        x = x.view(-1, 100, 1)
        # apply the model and flip the
        x = self.model(x).transpose(1, 2)
        return x


class VideoGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        # instantiate the temporal generator
        self.temp = TemporalGenerator()

        # create a transformation for the temporal vectors
        self.fast = nn.Sequential(
            nn.Linear(100, 256 * 4 ** 2, bias=False),
            nn.BatchNorm1d(256 * 4 ** 2),
            nn.ReLU()
        )

        # create a transformation for the content vector
        self.slow = nn.Sequential(
            nn.Linear(100, 256 * 4 ** 2, bias=False),
            nn.BatchNorm1d(256 * 4 ** 2),
            nn.ReLU()
        )

        # define the image generator
        self.model = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

        # initialize weights according to the paper
        self.fast.apply(self.init_weights)
        self.slow.apply(self.init_weights)
        self.model.apply(self.init_weights)

    def init_weights(self, m):
        if type(m) == nn.ConvTranspose2d or type(m) == nn.Linear:
            nn.init.uniform_(m.weight, a=-0.01, b=0.01)

    def forward(self, x):
        # pass our latent vector through the temporal generator and reshape
        z_fast = self.temp(x).contiguous()
        z_fast = z_fast.view(-1, 100)

        # transform the content and temporal vectors
        z_fast = self.fast(z_fast).view(-1, 256, 4, 4)
        z_slow = self.slow(x).view(-1, 256, 4, 4).unsqueeze(1)
        # after z_slow is transformed and expanded we can duplicate it
        z_slow = torch.cat([z_slow] * 16, dim=1).view(-1, 256, 4, 4)

        # concatenate the temporal and content vectors
        z = torch.cat([z_slow, z_fast], dim=1)

        # transform into image frames
        out = self.model(z)

        return out.view(-1, 16, 3, 64, 64).transpose(1, 2)


class VideoDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model3d = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=4, padding=1, stride=2),
            nn.LeakyReLU(0.2),
            nn.Conv3d(64, 128, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(128, 256, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.Conv3d(256, 512, kernel_size=4, padding=1, stride=2),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2)
        )

        self.conv2d = nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0)

        # initialize weights according to paper
        self.model3d.apply(self.init_weights)
        self.init_weights(self.conv2d)

    def init_weights(self, m):
        if type(m) == nn.Conv3d or type(m) == nn.Conv2d:
            nn.init.xavier_normal_(m.weight, gain=2 ** 0.5)

    def forward(self, x):
        h = self.model3d(x)
        # turn a tensor of R^NxTxCxHxW into R^NxCxHxW
        h = torch.reshape(h, (32, 512, 4, 4))
        h = self.conv2d(h)
        return h


# update discriminator
pr = dis(real)
fake = gen(torch.rand((batch_size, 100), device='cuda') * 2 - 1)
pf = dis(fake)
dis_loss = torch.mean(-pr) + torch.mean(pf)
dis_loss.backward()
disOpt.step()

# update generator
genOpt.zero_grad()
fake = gen(torch.rand((batch_size, 100), device='cuda') * 2 - 1)
pf = dis(fake)
gen_loss = torch.mean(-pf)
gen_loss.backward()
genOpt.step()

# instantiate the generator, load the weights, and create a sample
gen = VideoGenerator().cuda()
gen.load_state_dict(torch.load('state_normal81000.ckpt')['model_state_dict'][0])
genSamples(gen)
