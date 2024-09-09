import torch
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)

torch.save(vgg16, "../vgg16_mth.pth")

# torch.load("vgg_mth.path")


# 保存模型参数（推荐方法），以字典格式保存
torch.save(vgg16.state_dict(), "vgg16_mth1.pth")

# vgg16 = torchvision.models.vgg16(pretrained=False)
# vgg16.load_state_dict(torch.load("vgg16_mth1.pth"))
# model = torch.load("vgg16_mth1.pth")