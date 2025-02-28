from builtins import Exception, print
from calendar import prcal
import torch
import numpy as np
import math
import time
import json
import os
from easydict import EasyDict as edict
import torch.profiler
import os


torch.set_printoptions(precision=10)


class MyGELU(torch.nn.Module):
    def __init__(self):
        super(MyGELU, self).__init__()
        self.torch_PI = 3.1415926536

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            math.sqrt(2 / self.torch_PI) * (x + 0.044715 * torch.pow(x, 3))))


def json2Parser(json_path):
    """load json and return parser-like object"""
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)


class Net(torch.nn.Module):
    def __init__(self):
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
try:
    #load variables from constant/CanteraTorchProperties
    path_r = r"./constant/CanteraTorchProperties"
    with open(path_r, "r") as f:
        data = f.read()
        i = data.index('torchModel') 
        a = data.index('"',i) 
        b = data.index('sub',a) 
        c = data.index('"',b+1)
        modelName_split1 = data[a+1:b+3]
        modelName_split2 = data[b+3:c]

        modelPath = str(modelName_split1+modelName_split2)
        model1Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"1"+modelName_split2+"/checkpoint/")
        model2Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"2"+modelName_split2+"/checkpoint/")
        model3Path = str("mechanisms/"+modelPath+"/"+modelName_split1+"3"+modelName_split2+"/checkpoint/")
        
        i = data.index('GPU')
        a = data.index(';', i)
        b = data.rfind(' ',i+1,a)
        switch_GPU = data[b+1:a]

    #load OpenFOAM switch
    switch_on = ["true", "True", "on", "yes", "y", "t", "any"]
    switch_off = ["false", "False", "off", "no", "n", "f", "none"]
    if switch_GPU in switch_on:
        device = torch.device("cuda")
        device_ids = range(torch.cuda.device_count())
    elif switch_GPU in switch_off:
        device = torch.device("cpu")
        device_ids = [0]
    else:
        print("invalid setting!")
        os._exit(0)



    #glbal variable will only init once when called interperter
    #load parameters from json

    norm0 = json2Parser(str(model1Path+"norm.json"))
    norm1 = json2Parser(str(model2Path+"norm.json"))
    norm2 = json2Parser(str(model3Path+"norm.json"))
    setting0 = json2Parser(str(model1Path+"settings.json"))
    lamda = setting0.power_transform
    delta_t = setting0.delta_t
    dim = setting0.dim
    layers = setting0.layers
    

    Xmu0 = torch.tensor(norm0.input_mean).unsqueeze(0).to(device=device)
    Xstd0 = torch.tensor(norm0.input_std).unsqueeze(0).to(device=device)
    Ymu0 = torch.tensor(norm0.label_mean).unsqueeze(0).to(device=device)
    Ystd0 = torch.tensor(norm0.label_std).unsqueeze(0).to(device=device)

    Xmu1 = torch.tensor(norm1.input_mean).unsqueeze(0).to(device=device)
    Xstd1 = torch.tensor(norm1.input_std).unsqueeze(0).to(device=device)
    Ymu1 = torch.tensor(norm1.label_mean).unsqueeze(0).to(device=device)
    Ystd1 = torch.tensor(norm1.label_std).unsqueeze(0).to(device=device)

    Xmu2 = torch.tensor(norm2.input_mean).unsqueeze(0).to(device=device)
    Xstd2 = torch.tensor(norm2.input_std).unsqueeze(0).to(device=device)
    Ymu2 = torch.tensor(norm2.label_mean).unsqueeze(0).to(device=device)
    Ystd2 = torch.tensor(norm2.label_std).unsqueeze(0).to(device=device)

    #load model  
    model0 = Net()
    model1 = Net()
    model2 = Net()

    path_list=os.listdir(model1Path)
    for filename in path_list:
        if os.path.splitext(filename)[1] == '.pt':
            modelname = filename
    
    
    if torch.cuda.is_available()==False:
        check_point0 = torch.load(str(model1Path+modelname), map_location='cpu')
        check_point1 = torch.load(str(model2Path+modelname), map_location='cpu')
        check_point2 = torch.load(str(model3Path+modelname), map_location='cpu')
    else:
        check_point0 = torch.load(str(model1Path+modelname))
        check_point1 = torch.load(str(model2Path+modelname))
        check_point2 = torch.load(str(model3Path+modelname))
    
    model0.load_state_dict(check_point0)
    model1.load_state_dict(check_point1)
    model2.load_state_dict(check_point2)
    model0.to(device=device)
    model1.to(device=device)
    model2.to(device=device)

    if len(device_ids) > 1:
        model0 = torch.nn.DataParallel(model0, device_ids=device_ids)
        model1 = torch.nn.DataParallel(model1, device_ids=device_ids)
        model2 = torch.nn.DataParallel(model2, device_ids=device_ids)
except Exception as e:
    print(e.args)


def inference(vec0, vec1, vec2):
    '''
    use model to inference
    '''
    #args = np.reshape(args, (-1, 9)) #reshape to formed size
    #vec0 = np.reshape(vec0, (-1, 24))
    #vec1 = np.reshape(vec1, (-1, 24))
    #vec2 = np.reshape(vec2, (-1, 24))
    vec0 = np.reshape(vec0, (-1, 10))
    vec1 = np.reshape(vec1, (-1, 10))
    vec2 = np.reshape(vec2, (-1, 10))

    try:
        with torch.no_grad():
            input0_ = torch.from_numpy(vec0).double().to(device=device) #cast ndarray to torch tensor
            input1_ = torch.from_numpy(vec1).double().to(device=device) #cast ndarray to torch tensor
            input2_ = torch.from_numpy(vec2).double().to(device=device) #cast ndarray to torch tensor

            # pre_processing
            rho0 = input0_[:, 0].unsqueeze(1)
            input0_Y = input0_[:, 3:].clone()
            input0_bct = input0_[:, 1:]
            input0_bct[:, 2:] = (input0_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input0_normalized = (input0_bct - Xmu0) / Xstd0
            # input0_normalized[:, -1] = 0 #set Y_AR to 0
            input0_normalized = input0_normalized.float()

            rho1 = input1_[:, 0].unsqueeze(1)
            input1_Y = input1_[:, 3:].clone()
            input1_bct = input1_[:, 1:]
            input1_bct[:, 2:] = (input1_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input1_normalized = (input1_bct - Xmu1) / Xstd1
            # input1_normalized[:, -1] = 0 #set Y_AR to 0
            input1_normalized = input1_normalized.float()


            rho2 = input2_[:, 0].unsqueeze(1)
            input2_Y = input2_[:, 3:].clone()
            input2_bct = input2_[:, 1:]
            input2_bct[:, 2:] = (input2_bct[:, 2:]**(lamda) - 1) / lamda #BCT
            input2_normalized = (input2_bct - Xmu2) / Xstd2
            # input2_normalized[:, -1] = 0 #set Y_AR to 0
            input2_normalized = input2_normalized.float()

            #inference
            output0_normalized = model0(input0_normalized)
            output1_normalized = model1(input1_normalized)
            output2_normalized = model2(input2_normalized)
                       
            # post_processing
            output0_bct = (output0_normalized * Ystd0 + Ymu0) * delta_t + input0_bct
            output0_Y = (lamda * output0_bct[:, 2:] + 1)**(1 / lamda)
            output0_Y = output0_Y / torch.sum(input=output0_Y, dim=1, keepdim=True)
            output0 = (output0_Y - input0_Y) * rho0 / delta_t   
            output0 = output0.cpu().numpy()

            output1_bct = (output1_normalized * Ystd1 + Ymu1) * delta_t + input1_bct
            output1_Y = (lamda * output1_bct[:, 2:] + 1)**(1 / lamda)
            output1_Y = output1_Y / torch.sum(input=output1_Y, dim=1, keepdim=True)
            output1 = (output1_Y - input1_Y) * rho1 / delta_t
            output1 = output1.cpu().numpy()

            output2_bct = (output2_normalized * Ystd2 + Ymu2) * delta_t + input2_bct
            output2_Y = (lamda * output2_bct[:, 2:] + 1)**(1 / lamda)
            output2_Y = output2_Y / torch.sum(input=output2_Y, dim=1, keepdim=True)
            output2 = (output2_Y - input2_Y) * rho2 / delta_t
            output2 = output2.cpu().numpy()

            result = np.append(output0, output1, axis=0)
            result = np.append(result, output2, axis=0)
            return result
    except Exception as e:
        print(e.args)
