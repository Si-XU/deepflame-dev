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

import onnxruntime


torch.set_printoptions(precision=10)

def json2Parser(json_path):
    """load json and return parser-like object"""
    with open(json_path, 'r') as f:
        args = json.load(f)
    return edict(args)

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
    

    Xmu0 = np.array(norm0.input_mean)
    Xstd0 = np.array(norm0.input_std)
    Ymu0 = np.array(norm0.label_mean)
    Ystd0 = np.array(norm0.label_std)

    Xmu1 = np.array(norm1.input_mean)
    Xstd1 = np.array(norm1.input_std)
    Ymu1 = np.array(norm1.label_mean)
    Ystd1 = np.array(norm1.label_std)
    
    Xmu2 = np.array(norm2.input_mean)
    Xstd2 = np.array(norm2.input_std)
    Ymu2 = np.array(norm2.label_mean)
    Ystd2 = np.array(norm2.label_std)

    #load model  
    model0 = onnxruntime.InferenceSession(str(model1Path+"export_model.onnx"))
    model1 = onnxruntime.InferenceSession(str(model2Path+"export_model.onnx"))
    model2 = onnxruntime.InferenceSession(str(model3Path+"export_model.onnx"))

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
        input0_ = np.array(vec0) #cast ndarray to torch tensor
        input1_ = np.array(vec1) #cast ndarray to torch tensor
        input2_ = np.array(vec2) #cast ndarray to torch tensor

        # pre_processing
        rho0 = input0_[:, 0:1]
        input0_Y = input0_[:, 3:].copy()
        input0_bct = input0_[:, 1:]
        input0_bct[:, 2:] = (np.power(input0_bct[:, 2:], lamda) - 1) / lamda #BCT
        input0_normalized = (input0_bct - Xmu0) / Xstd0
        # input0_normalized[:, -1] = 0 #set Y_AR to 0
        input0_normalized = {"input" : input0_normalized.astype(np.float32)}

        rho1 = input1_[:, 0:1]
        input1_Y = input1_[:, 3:].copy()
        input1_bct = input1_[:, 1:]
        input1_bct[:, 2:] = (np.power(input1_bct[:, 2:], lamda) - 1) / lamda #BCT
        input1_normalized = (input1_bct - Xmu1) / Xstd1
        # input1_normalized[:, -1] = 0 #set Y_AR to 0
        input1_normalized = {"input" : input1_normalized.astype(np.float32)}


        rho2 = input2_[:, 0:1]
        input2_Y = input2_[:, 3:].copy()
        input2_bct = input2_[:, 1:]
        input2_bct[:, 2:] = (np.power(input2_bct[:, 2:], lamda) - 1) / lamda #BCT
        input2_normalized = (input2_bct - Xmu2) / Xstd2
        # input2_normalized[:, -1] = 0 #set Y_AR to 0
        input2_normalized = {"input" : input2_normalized.astype(np.float32)}

        #inference
        output0_normalized = model0.run(["output"], input0_normalized)[0]
        output1_normalized = model1.run(["output"], input1_normalized)[0]
        output2_normalized = model2.run(["output"], input2_normalized)[0]
        
        # post_processing
        output0_bct = (output0_normalized * Ystd0 + Ymu0) * delta_t + input0_bct
        output0_Y = (lamda * output0_bct[:, 2:] + 1)**(1 / lamda)
        output0_Y = output0_Y / np.sum(output0_Y, axis=1, keepdims=True)
        output0 = (output0_Y - input0_Y) * rho0 / delta_t

        output1_bct = (output1_normalized * Ystd1 + Ymu1) * delta_t + input1_bct
        output1_Y = (lamda * output1_bct[:, 2:] + 1)**(1 / lamda)
        output1_Y = output1_Y / np.sum(output1_Y, axis=1, keepdims=True)
        output1 = (output1_Y - input1_Y) * rho1 / delta_t

        output2_bct = (output2_normalized * Ystd2 + Ymu2) * delta_t + input2_bct
        output2_Y = (lamda * output2_bct[:, 2:] + 1)**(1 / lamda)
        output2_Y = output2_Y / np.sum(output2_Y, axis=1, keepdims=True)
        output2 = (output2_Y - input2_Y) * rho2 / delta_t

        result = np.append(output0, output1, axis=0)
        result = np.append(result, output2, axis=0)
        return result
    except Exception as e:
        print(e.args) 

tmp_input = np.random.rand(10,100,24)
inference(tmp_input, [], [])