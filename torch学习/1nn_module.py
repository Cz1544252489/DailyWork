import torch
from torch import nn

class Zhuo(nn.Module):

    def forward(self, input):
        output = input + 1
        return output

zhuo = Zhuo()
x =torch.tensor(1.0)
output = zhuo(x)
print(output)