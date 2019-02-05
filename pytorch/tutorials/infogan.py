import argparse
import os
import torch
import torchvision
import numpy as np
from torch.backends import cudnn
from torch import optim
from torch.autograd import Variable
from torch.utils import data
from torchvision import transforms
from torchvision import datasets
from PIL import Image
from infoGan_network import *

parser = argparse.ArgumentParser(description='Easy Implementation of InfoGAN')

# model hyper-parameters
parser.add_argument('--image_size', type=int, default=28) # 64 for CelebA
parser.add_argument('--z_dim', type=int, default=62) # 128 for CelebA

# training hyper-parameters
parser.add_argument('--num_epochs', type=int, default=30) # 30 or 50 for MNIST / 4 for CelebA
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--lrD', type=float, default=0.0002) # Learning Rate for D
parser.add_argument('--lrG', type=float, default=0.001) # Learning Rate for G
parser.add_argument('--beta1', type=float, default=0.5)  # momentum1 in Adam
parser.add_argument('--beta2', type=float, default=0.999)  # momentum2 in Adam

# InfoGAN parameters
parser.add_argument('--cc_dim', type=int, default=1)
parser.add_argument('--dc_dim', type=int, default=10)
parser.add_argument('--continuous_weight', type=float, default=0.5) # 0.1~0.5 for MNIST / 1.0 for CelebA

# misc
parser.add_argument('--db', type=str, default='mnist')  # Model Tmp Save
parser.add_argument('--model_path', type=str, default='./models')  # Model Tmp Save
parser.add_argument('--sample_path', type=str, default='./results')  # Results
parser.add_argument('--image_path', type=str, default='./CelebA/128_crop')  # Training Image Directory
parser.add_argument('--sample_size', type=int, default=100)
parser.add_argument('--log_step', type=int, default=50)
parser.add_argument('--sample_step', type=int, default=100)

##### Helper Function for Image Loading
class ImageFolder(data.Dataset):
    def __init__(self, root, transform=None):  # Initializes image paths and preprocessing module.
        self.image_paths = list(map(lambda x: os.path.join(root, x), os.listdir(root)))
        self.transform = transform

    def __getitem__(self, index):  # Reads an image from a file and preprocesses it and returns.
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image

    def __len__(self):  # Returns the total number of image files.
        return len(self.image_paths)

##### Helper Function for GPU Training
def to_variable(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

##### Helper Function for Math
def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# InfoGAN Function (Gaussian)
def gen_cc(n_size, dim):
    return torch.Tensor(np.random.randn(n_size, dim) * 0.5 + 0.0)

# InfoGAN Function (Multi-Nomial)
def gen_dc(n_size, dim):
    codes=[]
    code = np.zeros((n_size, dim))
    random_cate = np.random.randint(0, dim, n_size)
    code[range(n_size), random_cate] = 1
    codes.append(code)
    codes = np.concatenate(codes,1)
    return torch.Tensor(codes)

######################### Main Function
def main():
    # Pre-Settings
    cudnn.benchmark = True
    global args
    args = parser.parse_args()
    print(args)

    transform = transforms.Compose([
        transforms.Scale((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    if args.db == 'mnist':  # MNIST
        dataset = datasets.MNIST('./MNIST', train=True, transform=transform, target_transform=None, download=True)
    else: # CelebA
        dataset = ImageFolder(args.image_path, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=args.batch_size,
                                  shuffle=True,
                                  num_workers=args.num_workers)

    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)
    if not os.path.exists(args.sample_path):
        os.makedirs(args.sample_path)

    # Networks
    generator = Generator(args.db, args.z_dim, args.cc_dim, args.dc_dim)
    discriminator = Discriminator(args.db, args.cc_dim, args.dc_dim)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), args.lrD, [args.beta1, args.beta2])
    d_optimizer = optim.Adam(discriminator.parameters(), args.lrG, [args.beta1, args.beta2])

    if torch.cuda.is_available():
        generator.cuda()
        discriminator.cuda()

    """Train generator and discriminator."""
    fixed_noise = to_variable(torch.Tensor(np.zeros((args.sample_size, args.z_dim))))  # For Testing

    total_step = len(data_loader)  # For Print Log
    for epoch in range(args.num_epochs):
        for i, images in enumerate(data_loader):
            # ===================== Train D =====================#
            if args.db == 'mnist': # To Remove Label
                images = to_variable(images[0])
            else:
                images = to_variable(images)

            batch_size = images.size(0)
            noise = to_variable(torch.randn(batch_size, args.z_dim))

            cc = to_variable(gen_cc(batch_size, args.cc_dim))
            dc = to_variable(gen_dc(batch_size, args.dc_dim))

            # Fake -> Fake & Real -> Real
            fake_images = generator(torch.cat((noise, cc, dc),1))
            d_output_real = discriminator(images)
            d_output_fake = discriminator(fake_images)

            d_loss_a = -torch.mean(torch.log(d_output_real[:,0]) + torch.log(1 - d_output_fake[:,0]))

            # Mutual Information Loss
            output_cc = d_output_fake[:, 1:1+args.cc_dim]
            output_dc = d_output_fake[:, 1+args.cc_dim:]
            d_loss_cc = torch.mean((((output_cc - 0.0) / 0.5) ** 2))
            d_loss_dc = -(torch.mean(torch.sum(dc * output_dc, 1)) + torch.mean(torch.sum(dc * dc, 1)))

            d_loss = d_loss_a + args.continuous_weight * d_loss_cc + 1.0 * d_loss_dc

            # Optimization
            discriminator.zero_grad()
            d_loss.backward(retain_graph=True)
            d_optimizer.step()

            # ===================== Train G =====================#
            # Fake -> Real
            g_loss_a = -torch.mean(torch.log(d_output_fake[:,0]))

            g_loss = g_loss_a + args.continuous_weight * d_loss_cc + 1.0 * d_loss_dc

            # Optimization
            generator.zero_grad()
            g_loss.backward()
            g_optimizer.step()

            # print the log info
            if (i + 1) % args.log_step == 0:
                print('Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f'
                      % (epoch + 1, args.num_epochs, i + 1, total_step, d_loss.data[0], g_loss.data[0]))

            # save the sampled images (10 Category(Discrete), 10 Continuous Code Generation : 10x10 Image Grid)
            if (i + 1) % args.sample_step == 0:
                tmp = np.zeros((args.sample_size, args.cc_dim))
                for k in range(10):
                    tmp[k * 10:(k + 1) * 10, 0] = np.linspace(-2, 2, 10)
                cc = to_variable(torch.Tensor(tmp))
                tmp = np.zeros((args.sample_size, args.dc_dim))
                for k in range(10):
                    tmp[k * 10:(k + 1) * 10, k] = 1
                dc = to_variable(torch.Tensor(tmp))

                fake_images = generator(torch.cat((fixed_noise, cc, dc), 1))
                torchvision.utils.save_image(denorm(fake_images.data),
                                             os.path.join(args.sample_path,
                                                          'generated-%d-%d.png' % (epoch + 1, i + 1)), nrow=10)

        # save the model parameters for each epoch
        g_path = os.path.join(args.model_path, 'generator-%d.pkl' % (epoch + 1))
        torch.save(generator.state_dict(), g_path)

if __name__ == "__main__":
    main()