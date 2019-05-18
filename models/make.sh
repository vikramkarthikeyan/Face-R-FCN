#!/usr/bin/env bash

# CUDA_PATH=/usr/local/cuda/

export CUDA_PATH=/usr/local/cuda-10.0/

python3 setup.py build_ext --inplace
rm -rf build

CUDA_ARCH="-gencode arch=compute_60,code=sm_60"

# clean build file
rm psroi_pooling/src/cuda/*.o


cd psroi_pooling/src/cuda
echo "Compiling psroi pooling kernels by nvcc..."
nvcc -c -o psroi_pooling.cu.o psroi_pooling_kernel.cu \
	 -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC $CUDA_ARCH
cd ../../

python3 build.py install
