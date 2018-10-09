import torch

batch_size = 5
nb_digits = 10
# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y1 = torch.LongTensor(batch_size).random_() % nb_digits
print(list(y1.size()))
y = y1.view(-1,1)
# One hot encoding buffer that you create out of the loop and just keep reusing
y_onehot = torch.FloatTensor(batch_size, nb_digits)

# print(y_onehot)

# In your for loop
y_onehot.zero_()

# print(y_onehot)
y_onehot.scatter_(1, y, 1)

print(y)
print(y.size())
print(y_onehot)

# y.size()
