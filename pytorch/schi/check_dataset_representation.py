
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable

transform = transforms.Compose(
         [transforms.ToTensor()])

train_dataset = dsets.CIFAR10(root='./data-cifar', train=True, transform=transform, download=False)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=100,
                                           shuffle=True)

#"taste" the data
it = iter(train_loader)
im,_=it.next()
print(type(im))
print(im.shape)
print(im.min())
print(im.max())
torchvision.utils.save_image(im,'./data-cifar/example.png')