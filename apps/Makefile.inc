OPENCL_LDFLAGS=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/ -lOpenCL
#OPENCL_LDFLAGS=-L$(AMDAPPSDKROOT)/lib/x86_64 -lOpenCL
ANDROID_OPENCL_LDFALGS=-lOpenCL
ANDROID_CXX?=arm-linux-androideabi-g++
ANDROID_CXX_FLAGS=-llog -landroid -lOpenCL -I src/support -I src/fusion -I src/android
ANDROID_PNGFLAGS=-lpng
FRAMEWORK_PATH=../../framework
PNGFLAGS?=$(shell libpng-config --L_opts )
PNGFLAGS+=$(shell libpng-config --libs)
CXX=clang++

export HALIDE_PATH=$(HOME)/src/Halide
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$(HOME)/src/Halide/bin



