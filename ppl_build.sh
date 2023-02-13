#!/bin/sh

# export onnx model
cd mechanisms
python pytorch2onnxmodel.py
cd ..

# build pplnn with python api
if [ ! -d "ppl.nn" ]; then
  git clone -b xusi/deepflame git@github.com:Si-XU/ppl.nn.git
fi
cd ppl.nn
./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_PYTHON_API=ON -DPPLNN_ENABLE_CUDA_JIT=OFF -DPPLNN_ENABLE_PMX_MODEL=ON
./pplnn-build/tools/pplnn --use-cuda --onnx-model ../mechanisms/HE04_Hydrogen_ESH2_GMS_sub_20221101/HE04_Hydrogen_ESH2_GMS_sub1_20221101/checkpoint/export_model.onnx --in-shapes 10000_9 --export-algo-file ../mechanisms/algo.json
cd ..

# copy lib
if [ ! -d "pyppl" ]; then
  mkdir pyppl
fi
cp ppl.nn/pplnn-build/install/lib/pyppl/nn*so pyppl/nn.so
cp ppl.nn/pplnn-build/install/lib/pyppl/common*so pyppl/common.so
export PYTHONPATH=$PYTHONPATH:`pwd`