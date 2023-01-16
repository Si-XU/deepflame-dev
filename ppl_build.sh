#!/bin/sh

# build pplnn with python api
if [ ! -d "ppl.nn" ]; then
  git clone git@github.com:openppl-public/ppl.nn.git
fi
cd ppl.nn
./build.sh -DPPLNN_USE_CUDA=ON -DPPLNN_ENABLE_PYTHON_API=ON -DPPLNN_ENABLE_CUDA_JIT=OFF
cd ..

# copy lib
if [ ! -d "pyppl" ]; then
  mkdir pyppl
fi
cp ppl.nn/pplnn-build/install/lib/pyppl/nn.*.so pyppl/nn.so
cp ppl.nn/pplnn-build/install/lib/pyppl/common.*.so pyppl/common.so
export PYTHONPATH=$PYTHONPATH:`pwd`

# export onnx model
cd mechanisms
python pytorch2onnxmodel.py
cd ..
