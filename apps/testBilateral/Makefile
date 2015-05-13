include ../Makefile.inc
all:Release

halide/bilateral_grid: halide/bilateral_grid.cpp
	@$(CXX) halide/bilateral_grid.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/bilateral_grid -lpthread -ldl -lz -lHalide -std=c++11
halide/x86_64/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p x86_64;cd x86_64;../bilateral_grid 15;HL_TARGET=host-opencl ../bilateral_grid 4;
Release:main.cpp halide/x86_64/bilateral_grid_gpu.o
	$(CXX) halide/x86_64/bilateral_grid_gpu.o halide/x86_64/bilateral_grid_cpu.o  main.cpp -o run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL
clean:
	rm -f halide/local_laplacian halide/x86_64/bilateral_grid_gpu.o halide/x86_64/bilateral_grid_cpu.o run
