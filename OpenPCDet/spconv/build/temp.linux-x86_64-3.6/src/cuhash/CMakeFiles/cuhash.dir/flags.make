# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.13

# compile CUDA with /usr/local/cuda-10.0/bin/nvcc
# compile CXX with /usr/bin/c++
CUDA_FLAGS = "--expt-relaxed-constexpr" -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ -gencode arch=compute_70,code=sm_70 -O3 -DNDEBUG -Xcompiler=-fPIC   -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14

CUDA_DEFINES = -DPYTORCH_VERSION=10600 -DTV_CUDA

CUDA_INCLUDES = -I/home/mjamali/proj/G_All_b/3D_Object_Detection_Benchmarks/OpenPCDet/spconv/include -I/usr/local/cuda-10.0/targets/x86_64-linux/include -isystem=/home/mjamali/anaconda3/envs/pcdet/lib/python3.6/site-packages/torch/include -isystem=/home/mjamali/anaconda3/envs/pcdet/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem=/usr/local/cuda-10.0/include 

CXX_FLAGS = -DVERSION_INFO=\"1.2.1\" -O3 -DNDEBUG -fPIC   -D_GLIBCXX_USE_CXX11_ABI=0 -std=c++14 -std=c++14

CXX_DEFINES = -DPYTORCH_VERSION=10600 -DTV_CUDA

CXX_INCLUDES = -I/home/mjamali/proj/G_All_b/3D_Object_Detection_Benchmarks/OpenPCDet/spconv/include -I/usr/local/cuda-10.0/targets/x86_64-linux/include -isystem /home/mjamali/anaconda3/envs/pcdet/lib/python3.6/site-packages/torch/include -isystem /home/mjamali/anaconda3/envs/pcdet/lib/python3.6/site-packages/torch/include/torch/csrc/api/include -isystem /usr/local/cuda-10.0/include 

