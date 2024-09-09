from mpmath.identification import transforms
from torch.onnx.symbolic_opset9 import tensor
from torchvision import *
from PIL import Image
import cv2

# transforms.ToTensor的使用

img_path = "others/1/1.png"
img = Image.open(img_path)

tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

print(tensor_img)

cv_img = cv2.imread(img_path)
tensor_img1 = tensor_trans(cv_img)

# 两种调用方式
class Person:
    def __call__(self, name):
        print("__call__"+ " Hello "+ name)

    def hello(self, name):
        print("hello " + name)

person = Person()
person("huifang")
person.hello("zhuo")

#