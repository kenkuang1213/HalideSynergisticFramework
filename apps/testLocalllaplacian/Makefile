include ../Makefile.inc
all:Release

halide/local_laplacian: halide/local_laplacian.cpp
	g++ halide/local_laplacian.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/local_laplacian -lpthread -ldl -lz -lHalide -std=c++11
halide/x86_64/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p x86_64;cd x86_64;../local_laplacian;HL_TARGET=host-opencl ../local_laplacian;
halide/x86_64_info/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p x86_64_info;cd x86_64_info;../local_laplacian;HL_TARGET=host-opencl-debug ../local_laplacian
halide/x86_64_debug/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p x86_64_debug;cd x86_64_debug;../local_laplacian;HL_TARGET=host-opencl ../local_laplacian;
halide/cuda/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p cuda;cd cuda;../local_laplacian;HL_TARGET=host-cuda-debug ../local_laplacian;
halide/android/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p android;cd android;HL_TARGET=arm-32-android ../local_laplacian;HL_TARGET=arm-32-android-opencl ../local_laplacian;
halide/android_debug/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p android_debug;cd android_debug;HL_TARGET=arm-32-android ../local_laplacian;HL_TARGET=arm-32-android-opencl-debug ../local_laplacian;
Release_info:main.cpp halide/x86_64_info/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p release_info
	$(CXX) halide/x86_64_info/local_laplacian_gpu.o halide/x86_64_info/local_laplacian_cpu.o  main.cpp -o bin/release_info/run -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -I$(HALIDE_PATH)/include -DCOMPILING_FOR_OPENCL
Release:main.cpp halide/x86_64/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p release
	$(CXX) halide/x86_64/local_laplacian_gpu.o halide/x86_64/local_laplacian_cpu.o  main.cpp -o bin/release/run -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -I$(HALIDE_PATH)/include -DCOMPILING_FOR_OPENCL

Debug:main.cpp halide/x86_64/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p debug
	$(CXX) halide/x86_64/local_laplacian_gpu.o halide/x86_64/local_laplacian_cpu.o  main.cpp -o bin/debug/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL -DDEBUG
Android:main.cpp halide/android/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p android
	$(ANDROID_CXX) halide/android/local_laplacian_gpu.o halide/android/local_laplacian_cpu.o  main.cpp -o bin/android/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/android  -ldl -landroid -lOpenCL -lpng -std=c++11 -llog -DCOMPILING_FOR_OPENCL
Android_debug:main.cpp halide/android_debug/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p android_debug
	$(ANDROID_CXX) halide/android_debug/local_laplacian_gpu.o halide/android_debug/local_laplacian_cpu.o  main.cpp -o bin/android_debug/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/android  -ldl -landroid -lOpenCL -lpng -std=c++11 -llog -DCOMPILING_FOR_OPENCL -DANDROID
Cuda:main.cpp halide/cuda/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p cuda
	$(CXX) halide/cuda/local_laplacian_gpu.o halide/cuda/local_laplacian_cpu.o  main.cpp -o bin/cuda/run -I$(FRAMEWORK_PATH)/include -Ihalide/cuda -lpthread -std=c++11 $(PNGFLAGS) -I$(HALIDE_PATH)/include -lcuda -L/usr/lib/x86_64-linux-gnu/
clean:
	rm -rf halide/local_laplacian halide/x86_64/ halide/android/ run bin
