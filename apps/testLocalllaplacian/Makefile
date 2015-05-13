include ../Makefile.inc
all:Release

halide/local_laplacian: halide/local_laplacian_gen.cpp
	@$(CXX) halide/local_laplacian_gen.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/local_laplacian -lpthread -ldl -lz -lHalide -std=c++11
halide/x86_64/local_laplacian_cpu.o:halide/local_laplacian
	@cd halide/;mkdir -p x86_64;cd x86_64;../local_laplacian;HL_TARGET=host-opencl ../local_laplacian;
Release:main.cpp halide/x86_64/local_laplacian_cpu.o
	$(CXX) halide/x86_64/local_laplacian_gpu.o halide/x86_64/local_laplacian_cpu.o  main.cpp -o run -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -I$(HALIDE_PATH)/include -DCOMPILING_FOR_OPENCL
clean:
	rm -f halide/local_laplacian halide/x86_64/local_laplacian_cpu.o halide/x86_64/local_laplacian_gpu.o run
