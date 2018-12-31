
import torch
import torch.nn as nn
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable


# Hyper Parameters 
input_size = 784
hidden_size = 64
num_classes = 10
num_epochs = 10
batch_size = 100
learning_rate = 0.001


transform = transforms.Compose(
         [transforms.ToTensor(),
         transforms.Normalize((0.1307,), (0.3015,))])

# MNIST Dataset (Images and Labels)
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transform,
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transform)

# Dataset Loader (Input Pipline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)
#"taste" the data
it = iter(train_loader)
im,_=it.next()
torchvision.utils.save_image(im,'./data/example.png')

# Model
class NeuralNet(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size//2)  
        self.fc3 = nn.Linear(hidden_size//2, num_classes) 

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

net = NeuralNet(input_size, num_classes)

# Loss and Optimizer
# Softmax is internally computed.
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

print ('number of parameters: ', sum(param.numel() for param in net.parameters()))
# Training the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 200 == 0:
            print ('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f'
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
    
# Test the Model
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images.view(-1, 28*28))
    outputs = net(images)
    _, predicted = torch.max(outputs.data, 1)
    
    total += labels.size(0)
    correct += (predicted == labels).sum()
    
print ('Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

#Save the Model
#torch.save(model.state_dict(), 'model.pkl')

