from builtins import Exception, print
from calendar import prcal
import torch
import numpy as np
import math
import time
import json
import os
from easydict import EasyDict as edict
import os

from torch.autograd import Function

torch.set_printoptions(precision=10)
print('position 0 in inference.py')
device = torch.device("cpu")
device_ids = range(torch.cuda.device_count())


class GELU(Function):
    @staticmethod
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / 3.1415926536) * (x + 0.044715 * torch.pow(x, 3))))
    @staticmethod
    def symbolic(g, input):
        return g.op("pmx::GELU", input)

class MyGELU(torch.nn.Module):
    def __init__(self):
        super(MyGELU, self).__init__()
        self.torch_PI = 3.1415926536
        self.func = GELU.apply

    def forward(self, x):
        x = self.func(x)
        return x


def json2Parser(json_path):
    """load json and return parser-like object"""
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)


class Net(torch.nn.Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        neurons = layers
        self.depth = len(neurons) - 1
        self.actfun = MyGELU()
        self.layers = []
        for i in range(self.depth - 1):
            self.layers.append(torch.nn.Linear(neurons[i], neurons[i + 1]))
            self.layers.append(self.actfun)
        self.layers.append(torch.nn.Linear(neurons[-2], neurons[-1]))  # last layer
        self.fc = torch.nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.fc(x)
        return x


def findAllPTFile(base):
    for root, ds, fs in os.walk(base):
        for f in fs:
            fullname = os.path.join(root, f)
            if fullname.endswith('.pt'):
            	yield fullname

#load module
for pt_path in findAllPTFile('./HE04_Hydrogen_ESH2_GMS_sub_20221101/'):
    print(pt_path)
    dir_path = pt_path[0: pt_path.rfind('/')]
    json_path = os.path.join(dir_path, 'settings.json')
    setting = json2Parser(json_path)

    model = Net(setting.layers)
    model.load_state_dict(torch.load(pt_path))
    
    inputs = torch.rand(1000, setting.layers[0])
    torch.onnx.export(model, inputs, os.path.join(dir_path, "export_model.onnx"),
            input_names = ['input'],
            output_names = ['output'],
            opset_version=12,
            dynamic_axes={
            "input" : {0 : "batch"},
            "output" : {0 : "batch"}})
