from torch import  nn
import torchvision

vgg16_false = torchvision.models.vgg16(pretrained=False)


# vgg16_true = torchvision.models.vgg16(pretrained=True)

vgg16_false.add_module('add_linear', nn.Linear(1000, 10))
print(vgg16_false)


vgg16_false1 = torchvision.models.vgg16(pretrained=False)
vgg16_false1.classifier[6] = nn.Linear(4096, 10)
print(vgg16_false1)