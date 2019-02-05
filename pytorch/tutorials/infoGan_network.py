import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, db='mnist', z_dim=128, cc_dim=1, dc_dim=10):
        super(Generator, self).__init__()
        self.db = db

        if self.db == 'mnist':
            self.fc = nn.Sequential(
                nn.Linear(z_dim + cc_dim + dc_dim, 1024),
                nn.BatchNorm2d(1024),
                nn.ReLU(),

                nn.Linear(1024, 128*7*7),
                nn.BatchNorm2d(128*7*7),
                nn.ReLU()
            )
            self.conv = nn.Sequential(
                # [-1, 128, 7, 7] -> [-1, 64, 14, 14]
                nn.ConvTranspose2d(128,64,4,2,1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                # -> [-1, 1, 28, 28]
                nn.ConvTranspose2d(64,1,4,2,1),
                nn.Tanh()
            )
        else:
            self.main = nn.Sequential(
                # [-1, z + cc + dc, 1, 1] -> [-1, 512, 4, 4]
                nn.ConvTranspose2d(z_dim + cc_dim + dc_dim, 1024, 4, 1, 0),

                nn.ConvTranspose2d(1024, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.ReLU(),

                # [-1, 256, 8, 8]
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),

                # [-1, 128, 16, 16]
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),

                # [-1, 3, 32, 32]
                nn.ConvTranspose2d(128, 3, 4, 2, 1),
                nn.Tanh()
            )

    def forward(self, z):
        if self.db == 'mnist':
            # [-1, z]
            z = self.fc( z )

            # [-1, 128*7*7] -> [-1, 128, 7, 7]
            z = z.view(-1, 128, 7, 7)
            out = self.conv(z)
        else:
            # [-1, z] -> [-1, z, 1, 1]
            z = z.view(z.size(0), z.size(1), 1, 1)
            out = self.main( z )

        return out


class Discriminator(nn.Module):
    def __init__(self, db='mnist', cc_dim = 1, dc_dim = 10):
        super(Discriminator, self).__init__()
        self.db = db
        self.cc_dim = cc_dim
        self.dc_dim = dc_dim

        if self.db=='mnist':
            self.conv = nn.Sequential(
                # [-1, 1, 28, 28] -> [-1, 64, 14, 14]
                nn.Conv2d(1, 64, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 128, 7, 7]
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
            )
            self.fc = nn.Sequential(
                nn.Linear(128*7*7, 128),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.1, inplace=True),
                nn.Linear(128, 1 + cc_dim + dc_dim)
            )
        else:
            self.main = nn.Sequential(
                # [-1, 3, 32, 32] -> [-1, 128, 16, 16]
                nn.Conv2d(3, 128, 4, 2, 1),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 256, 8, 8]
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 512, 4, 4]
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.1, inplace=True),

                nn.Conv2d(512, 1024, 4, 2, 1),
                nn.BatchNorm2d(1024),
                nn.LeakyReLU(0.1, inplace=True),

                # [-1, 1 + cc_dim + dc_dim, 1, 1]
                nn.Conv2d(1024, 1 + cc_dim + dc_dim, 4, 1, 0)
            )

    def forward(self, x):
        if self.db == 'mnist':
            # -> [-1, 128*7*7]
            tmp = self.conv(x).view(-1, 128*7*7)

            # -> [-1, 1 + cc_dim + dc_dim]
            out = self.fc(tmp)
        else:
            # -> [-1, 1 + cc_dim + dc_dim]
            out = self.main(x).squeeze()

        # Discrimination Output
        out[:, 0] = F.sigmoid(out[:, 0].clone())

        # Continuous Code Output = Value Itself
        # Discrete Code Output (Class -> Softmax)
        out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim] = F.softmax(out[:, self.cc_dim + 1:self.cc_dim + 1 + self.dc_dim].clone())

        return out
