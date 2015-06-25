include ../Makefile.inc
all:Release

halide/bilateral_grid: halide/bilateral_grid.cpp
	g++ halide/bilateral_grid.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/bilateral_grid -lpthread -ldl -lz -lHalide -std=c++11
halide/x86_64/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p x86_64;cd x86_64;../bilateral_grid 15;HL_TARGET=host-opencl ../bilateral_grid 15;
halide/debug_info/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p debug_info;cd debug_info;../bilateral_grid 15;HL_TARGET=host-opencl-debug ../bilateral_grid 15;
halide/android/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p android;cd android;HL_TARGET=arm-32-android ../bilateral_grid 8;HL_TARGET=arm-32-android-opencl ../bilateral_grid 8;
halide/android_debug/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p android_debug;cd android_debug;HL_TARGET=arm-32-android ../bilateral_grid 8;HL_TARGET=arm-32-android-opencl-debug ../bilateral_grid 8;
halide/cuda/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p cuda;cd cuda;HL_TARGET=host ../bilateral_grid 15;HL_TARGET=host-cuda ../bilateral_grid 15;
halide/cuda_info/bilateral_grid_gpu.o:halide/bilateral_grid
	@cd halide/;mkdir -p cuda_info;cd cuda_info;HL_TARGET=host ../bilateral_grid 15;HL_TARGET=host-cuda-debug ../bilateral_grid 15;
Release:main.cpp halide/x86_64/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p release
	$(CXX) halide/x86_64/bilateral_grid_gpu.o halide/x86_64/bilateral_grid_cpu.o  main.cpp -o bin/release/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL -L/usr/lib/x86_64-linux-gnu/
Release_info:main.cpp halide/debug_info/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p release_info
	$(CXX) halide/debug_info/bilateral_grid_gpu.o halide/debug_info/bilateral_grid_cpu.o  main.cpp -o bin/release_info/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/debug_info -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL -L/usr/lib/x86_64-linux-gnu/
Cuda:main.cpp halide/cuda/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p cuda
	$(CXX) halide/cuda/bilateral_grid_gpu.o halide/cuda/bilateral_grid_cpu.o  main.cpp -o bin/cuda/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/cuda -lpthread -std=c++11 $(PNGFLAGS) -lcuda -L/usr/lib/x86_64-linux-gnu/
Cuda_info:main.cpp halide/cuda_info/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p cuda_info
	$(CXX) halide/cuda_info/bilateral_grid_gpu.o halide/cuda_info/bilateral_grid_cpu.o  main.cpp -o bin/cuda_info/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/cuda -lpthread -std=c++11 $(PNGFLAGS) -lcuda -L/usr/lib/x86_64-linux-gnu/
Debug:main.cpp halide/x86_64/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p debug
	$(CXX) halide/x86_64/bilateral_grid_gpu.o halide/x86_64/bilateral_grid_cpu.o  main.cpp -o bin/debug/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL -DDEBUG
Android:main.cpp halide/android/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p android
	$(ANDROID_CXX) halide/android/bilateral_grid_gpu.o halide/android/bilateral_grid_cpu.o  main.cpp -o bin/android/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/android -ldl -landroid -lOpenCL -lpng -std=c++11 -llog -DCOMPILING_FOR_OPENCL
AndroidDebug:main.cpp halide/android_debug/bilateral_grid_gpu.o
	@mkdir -p bin;cd bin;mkdir -p android_debug
	$(ANDROID_CXX) halide/android_debug/bilateral_grid_gpu.o halide/android_debug/bilateral_grid_cpu.o  main.cpp -o bin/android_debug/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/android_debug -ldl -landroid -lOpenCL -lpng -std=c++11 -llog -DCOMPILING_FOR_OPENCL -DANDROID

clean:
	rm -rf halide/android/ halide/android_debug/ halide/x86_64/ run halide/bilateral_grid bin/
