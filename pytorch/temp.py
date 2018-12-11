import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy import stats

batch_size = 5
nb_digits = 10

kl_criterion = nn.KLDivLoss(reduce=False)

# Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
y1_int = torch.LongTensor(batch_size, 1).random_() % nb_digits
print(y1_int.size())
# y1_int = y1_int.view(-1,1)
# print(y1_int.size())
y1_int = y1_int.view(y1_int.size(0), y1_int.size(1), -1)
print(y1_int.size())
print(y1_int)

# One hot encoding buffer that you create out of the loop and just keep reusing
y1_one_hot = torch.FloatTensor(batch_size, 1, nb_digits)
y1_one_hot.zero_()
print(y1_one_hot.size())
y1_one_hot.scatter_(2, y1_int, 1)

# y1_one_hot = torch.FloatTensor(batch_size, nb_digits)
# y1_one_hot.zero_()
# y1_one_hot.scatter_(1, y1_int, 1)

# y1_one_hot.unsqueeze(0)
# y1_one_hot.view(-1, batch_size, 10)

# y1_one_hot = F.softmax(y1_one_hot, dim=1)
# y1_one_hot.log_()

print(y1_one_hot)
print(y1_one_hot.size())



t = torch.ones((2, 3, 4))
print(t.size())

print(t.view(-1, 12).size())

# # Dummy input that HAS to be 2D for the scatter (you can use view(-1,1) if needed)
# y2_int = torch.LongTensor(batch_size, 1).random_() % nb_digits
# y2_int = y2_int.view(-1, 1)
# # One hot encoding buffer that you create out of the loop and just keep reusing
# y2_one_hot = torch.FloatTensor(batch_size, nb_digits)
#
# # print(y_onehot)
# # # y.size()
#
# y2_one_hot.zero_()
# y2_one_hot.scatter_(1, y2_int, 1)
# print(y2_one_hot)
# y2_one_hot = F.log_softmax(y2_one_hot, dim=1)

# print(y2_one_hot)
# print(y2_one_hot.size())
# print(y2_int)

# # y2_one_hot = torch.ones(batch_size, nb_digits, dtype=torch.float)
#
# # y2_one_hot = torch.ones(batch_size, nb_digits, dtype=torch.float) - y1_one_hot
# y2_one_hot = F.log_softmax(torch.randn(batch_size, nb_digits, dtype=torch.float), dim=1)
#
# # y1_one_hot = torch.rand(batch_size, nb_digits, dtype=torch.float)
# # y1_one_hot = y1_one_hot/torch.sum(y1_one_hot)
#
# # print(y1_one_hot)
# # print(torch.sum(y1_one_hot))
# # print((list(y1_one_hot.shape))[1])
#
# # y2_one_hot = F.softmax(torch.ones(batch_size, nb_digits, dtype=torch.float) - y1_one_hot/(nb_digits-1), dim=1)
#
# # print(y2_one_hot)
# # y2_one_hot.log_()
#
# # print(y2_one_hot)
#
# # kl_criterion(log_prob, prob)
# res = kl_criterion(y2_one_hot, y1_one_hot)
# res = res/(batch_size*nb_digits)
# res_min = kl_criterion(y1_one_hot.log_(), y1_one_hot)
# # res_rev = kl_criterion(y1_one_hot, y2_one_hot)
# # res_equal = kl_criterion(y2_one_hot, y2_one_hot)
#
#
# # # print(res)
# # print((torch.sum(res)).item())
# # print(res_min)
# # # print(res_equal.item())
#
# # temp = torch.tensor(list(range(11,21)))
# # mylist = list(range(10))
# # print(temp)
#
# # y3_vect = np.random.rand(nb_digits)
# y3_vect = np.ones(nb_digits)/nb_digits
# # y3_vect = y3_vect/sum(y3_vect)
# print(y3_vect)
#
# entrop = stats.entropy(y3_vect)
# print(entrop)
#
#
# a1= [False, False, True, False, True, True]
#
# a2= [True, False, True, False, False, True]
#
# b1= np.nonzero(a1)
# b2= np.nonzero(a2)
#
# un = np.union1d(b1, b2)
#
# print(un)
# print(b1)
# print(b2)
#
# a= []
# a.append('st')
# a.append(2.5)
# a.append([99,25])
# a.append([[17,25],[6,1]])
#
# for it in a:
#     if isinstance(it,(list,)):
#         if isinstance(it[0], (list,)):
#             print(len(it[0]))
#     # print(type(it))