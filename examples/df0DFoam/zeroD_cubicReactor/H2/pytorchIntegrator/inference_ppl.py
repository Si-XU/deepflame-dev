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
import sys
import logging
import ml_collections as mlc

from pyppl import nn as pplnn
from pyppl import common as pplcommon

pplnn.SetLoggingLevel(6)

def ParseInShapes(in_shapes_str):
    ret = []
    shape_strs = list(filter(None, in_shapes_str.split(",")))
    for s in shape_strs:
        dims = [int(d) for d in s.split("_")]
        ret.append(dims)
    return ret

def CalcElementCount(dims):
    count = 1
    for d in dims:
        count = count * d
    return count

def CreateCudaEngine(args):
    cuda_options = pplnn.cuda.EngineOptions()
    cuda_options.device_id = args.device_id
    if args.mm_policy == "perf":
        cuda_options.mm_policy = pplnn.cuda.MM_BEST_FIT
    elif args.mm_policy == "mem":
        cuda_options.mm_policy = pplnn.cuda.MM_COMPACT

    cuda_engine = pplnn.cuda.EngineFactory.Create(cuda_options)
    if not cuda_engine:
        logging.error("create cuda engine failed.")
        sys.exit(-1)

    if args.quick_select:
        status = cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_USE_DEFAULT_ALGORITHMS)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(ENGINE_CONF_USE_DEFAULT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.in_shapes:
        shapes = ParseInShapes(args.in_shapes)
        status = cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_SET_INPUT_DIMS, shapes)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(ENGINE_CONF_SET_INPUT_DIMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.export_algo_file:
        status = cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_EXPORT_ALGORITHMS, args.export_algo_file)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(ENGINE_CONF_EXPORT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.import_algo_file:
        # import and export from the same file
        if args.import_algo_file == args.export_algo_file:
            # try to create this file first
            f = open(args.export_algo_file, "a")
            f.close()

        status = cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_IMPORT_ALGORITHMS, args.import_algo_file)
        if status != pplcommon.RC_SUCCESS:
            logging.error("cuda engine Configure(ENGINE_CONF_IMPORT_ALGORITHMS) failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

    if args.kernel_type:
        upper_type_str = args.kernel_type.upper()
        kernel_type = pplcommon.DATATYPE_UNKNOWN
        for i in range(pplcommon.DATATYPE_MAX):
            if pplcommon.GetDataTypeStr(i) == upper_type_str:
                kernel_type = i
                break
        if kernel_type != pplcommon.DATATYPE_UNKNOWN:
            cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_SET_KERNEL_TYPE, kernel_type)
        else:
            logging.error("invalid kernel type[" + args.kernel_type + "]. valid types: int8/16/32/64, float16/32.")
            sys.exit(-1)

    if args.quant_file:
        with open(args.quant_file, 'r') as f:
            cuda_engine.Configure(pplnn.cuda.ENGINE_CONF_SET_QUANT_INFO, f.read())

    return cuda_engine
    
class PPLModel(object):
    def __init__(self, model_path) -> None:
        self.config = mlc.ConfigDict(
            {
                'device_id' : 0,
                'mm_policy' : "perf",
                'quick_select' : False,
                'in_shapes': "10000_9",
                'onnx_model': model_path,
                'export_algo_file': None,
                'import_algo_file': "mechanisms/algo.json",
                'kernel_type': "FLOAT16",
                'quant_file': None
            }
        )
        self.prepare(config = self.config)

    def prepare(self, config) -> None:
        
        engines = CreateCudaEngine(config)
        engines = [engines]
        if not engines:
            logging.error("no engine is specified. run '" + sys.argv[0] + " -h' to see supported device types marked with '--use-*'.")
            sys.exit(-1)

        if config.onnx_model:
            runtime_builder = pplnn.onnx.RuntimeBuilderFactory.Create()
            if not runtime_builder:
                logging.error("create OnnxRuntimeBuilder failed.")
                sys.exit(-1)

            status = runtime_builder.LoadModelFromFile(config.onnx_model)
            if status != pplcommon.RC_SUCCESS:
                logging.error("init OnnxRuntimeBuilder failed: " + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)
            
            resources = pplnn.onnx.RuntimeBuilderResources()
            resources.engines = engines
            status = runtime_builder.SetResources(resources)
            if status != pplcommon.RC_SUCCESS:
                logging.error("onnx RuntimeBuilder set resources failed: " + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

            status = runtime_builder.Preprocess()
            if status != pplcommon.RC_SUCCESS:
                logging.error("OnnxRuntimeBuilder preprocess failed: " + pplcommon.GetRetCodeStr(status))
                sys.exit(-1)

            self._runtime = runtime_builder.CreateRuntime()
            if not self._runtime:
                logging.error("create Runtime instance failed.")
                sys.exit(-1)

    def set_input(self, input, shapes = None):
        for i in range(len(input)):
            tensor = self._runtime.GetInputTensor(i)
            status = tensor.ConvertFromHost(input[i])
            if status != pplcommon.RC_SUCCESS:
                logging.error("copy data to tensor[" + tensor.GetName() + "] failed: " +
                            pplcommon.GetRetCodeStr(status))
                sys.exit(-1)
            

    def get_output(self):
        output = []
        for i in range(self._runtime.GetOutputCount()):
            tensor = self._runtime.GetOutputTensor(i)
            shape = tensor.GetShape()
            dims = shape.GetDims()
            element_count = CalcElementCount(dims)
            if element_count > 0:
                tensor_data = tensor.ConvertToHost()
                output.append(np.array(tensor_data, copy=False))
            else:
                output.append(np.empty(dims, dtype=np.float32))

        return output

    def forward(self, input):
        
        self.set_input(input)

        status = self._runtime.Run()
        if status != pplcommon.RC_SUCCESS:
            logging.error("Run() failed: " + pplcommon.GetRetCodeStr(status))
            sys.exit(-1)

        return self.get_output()

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
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

	#Create Model
    model0 = PPLModel(str(model1Path+"export_model.onnx"))
    model1 = PPLModel(str(model2Path+"export_model.onnx"))
    model2 = PPLModel(str(model3Path+"export_model.onnx"))

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

    #try:
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
    if len(input0_normalized):
        input0_normalized = input0_normalized.cpu().numpy()
        output0_normalized = torch.from_numpy(model0.forward([input0_normalized])[0]).to(device=device)
    else:
        output0_normalized = torch.from_numpy(np.empty([0,9])).to(device=device)
    if len(input1_normalized):
        input1_normalized = input1_normalized.cpu().numpy()
        output1_normalized = torch.from_numpy(model1.forward([input1_normalized])[0]).to(device=device)
    else:
        output1_normalized = torch.from_numpy(np.empty([0,9])).to(device=device)
    if len(input2_normalized):
        input2_normalized = input2_normalized.cpu().numpy()
        output2_normalized = torch.from_numpy(model2.forward([input2_normalized])[0]).to(device=device)
    else:
        output2_normalized = torch.from_numpy(np.empty([0,9])).to(device=device)        

    # post_processing
    output0_bct = (output0_normalized * Ystd0 + Ymu0) * delta_t + input0_bct
    output0_Y = (lamda * output0_bct[:, 2:] + 1)**(1 / lamda)
    output0_Y = output0_Y / torch.sum(input=output0_Y, dim=1, keepdim=True)
    output0 = (output0_Y - input0_Y) * rho0 / delta_t   
    output0 = (output0_Y - input0_Y) * rho0 / delta_t
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
