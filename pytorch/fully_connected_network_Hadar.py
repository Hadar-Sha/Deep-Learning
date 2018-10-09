# Code in file autograd/two_layer_net_autograd.py
import torch
from torch.autograd import Variable
import pandas as pd


dtype = torch.LongTensor
# dtype = torch.FloatTensor
# dtype = torch.cuda.FloatTensor # Uncomment this to run on GPU

# input_size = 784
# hidden_size = 64
# num_classes = 10
# num_epochs = 10
# batch_size = 100
# learning_rate = 0.001

# N is batch size; D_in is input dimension;
# H is hidden dimension; D_out is output dimension.
# N, D_in, H, D_out = 64, 1000, 100, 10
N, D_in, H, D_out = 100, 24, 12, 5500

digits_train = pd.read_csv('../data/data-deep-train.csv',header = None)
digits_train_tensor = digits_train.as_matrix()
training_data= torch.from_numpy(digits_train_tensor)
training_input= training_data[: , :24]
training_labels= training_data[: , 24]

# Create random Tensors to hold input and outputs, and wrap them in Variables.
# Setting requires_grad=False indicates that we do not need to compute gradients
# with respect to these Variables during the backward pass.
# x = Variable(torch.randn(N, D_in).type(dtype), requires_grad=False)
# y = Variable(torch.randn(N, D_out).type(dtype), requires_grad=False)

x= Variable(training_input, requires_grad=False)
y= Variable(training_labels, requires_grad=False)

# Create random Tensors for weights, and wrap them in Variables.
# Setting requires_grad=True indicates that we want to compute gradients with
# respect to these Variables during the backward pass.
w1 = Variable(torch.randn(D_in, H).type(dtype), requires_grad=True)
w2 = Variable(torch.randn(H, D_out).type(dtype), requires_grad=True)

learning_rate = 0.01
# learning_rate = 1e-6

for t in range(500):
    # Forward pass: compute predicted y using operations on Variables; these
    # are exactly the same operations we used to compute the forward pass using
    # Tensors, but we do not need to keep references to intermediate values since
    # we are not implementing the backward pass by hand.
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # Compute and print loss using operations on Variables.
    # Now loss is a Variable of shape (1,) and loss.data is a Tensor of shape
    # (1,); loss.data[0] is a scalar value holding the loss.
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.data[0])

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Variables with requires_grad=True.
    # After this call w1.grad and w2.grad will be Variables holding the gradient
    # of the loss with respect to w1 and w2 respectively.
    loss.backward()

    # Update weights using gradient descent; w1.data and w2.data are Tensors,
    # w1.grad and w2.grad are Variables and w1.grad.data and w2.grad.data are
    # Tensors.
    w1.data -= learning_rate * w1.grad.data
    w2.data -= learning_rate * w2.grad.data

    # Manually zero the gradients
    w1.grad.data.zero_()
    w2.grad.data.zero_()