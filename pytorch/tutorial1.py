# -*- coding: utf-8 -*-
"""
Created on Wed Mar 21 17:55:41 2018

@author: H
"""

import torch
#direct init
a = torch.Tensor([[[1,2,3],[4,5,6]],
[[7,8,9],[10,11,12]],
[[13,14,15],[16,17,18]],
[[19,20,21],[22,23,24]]])
#random init
b = torch.rand(3,4,5) # uniform (0,1)
b1 = torch.randn(3,4,5) # standard normal
#useful shortcuts
c = torch.ones(4,5)
d = torch.zeros(4,5)
e = torch.ones(4,5)*3
print ("a: ",a)
print ("b: ",b)
print ("b1: ",b1)
print ("c: ",c)
print ("d: ",d)
print ("e: ",e)

a = torch.Tensor([[1,2,3],[4,5,6]])
a_new = a # indirect assignment - they are both pointing to same storage
a_clone = a.clone() # create new memory copy
a_new.zero_() # zero in-place
print (a,a_new,a_clone)