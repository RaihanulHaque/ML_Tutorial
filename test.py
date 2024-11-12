# import timm 
# import torch

# # model = timm.create_model('resnet34')
# # x     = torch.randn(1, 3, 224, 224)
# # model(x).shape

# model = torch.load("model.pth")
# x     = torch.randn(1, 3, 328, 328)
# print(model(x).shape)

# Given values

import numpy as np
a = 9
b = 40
c = 28
d = 15

# Step 1: Calculate the semi-perimeter
s = (a + b + c + d) / 2

# Step 2: Use Brahmagupta's formula to calculate the area
area = np.sqrt((s - a) * (s - b) * (s - c) * (s - d))

# Step 3: Calculate the radius of the inscribed circle
r = area / s
print(s)
print(area)
print(r)

rah = "rahi, ma"
print(rah.split(","))