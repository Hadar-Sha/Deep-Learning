import torch
from torch.autograd import Variable
from utee import selector
import svnh
# model_raw, ds_fetcher, is_imagenet = selector.select('svhn')
model_raw, ds_fetcher, is_imagenet = selector.select('svhn', cuda=False)
ds_val = ds_fetcher(batch_size=10, train=False, val=True)
for idx, (data, target) in enumerate(ds_val):
    data = Variable(torch.FloatTensor(data))
    # data = Variable(torch.FloatTensor(data)).cuda()
    output = model_raw(data)



# save_checkpoint({'epoch': 0,
#                       'state_dict': model_raw.state_dict(),
#                       'optim_dict': model_raw.state_dict()}, is_best=False, checkpoint='./')