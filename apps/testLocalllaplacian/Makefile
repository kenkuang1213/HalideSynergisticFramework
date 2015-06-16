include ../Makefile.inc
all:Release

halide/local_laplacian: halide/local_laplacian.cpp
	g++ halide/local_laplacian.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/local_laplacian -lpthread -ldl -lz -lHalide -std=c++11
halide/x86_64/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p x86_64;cd x86_64;../local_laplacian;HL_TARGET=host-opencl ../local_laplacian;
halide/android/local_laplacian_cpu.o:halide/local_laplacian
	cd halide/;mkdir -p android;cd android;HL_TARGET=arm-32-android ../local_laplacian;HL_TARGET=arm-32-android-opencl ../local_laplacian;
Release:main.cpp halide/x86_64/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p release
	$(CXX) halide/x86_64/local_laplacian_gpu.o halide/x86_64/local_laplacian_cpu.o  main.cpp -o bin/release/run -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -I$(HALIDE_PATH)/include -DCOMPILING_FOR_OPENCL
Debug:main.cpp halide/x86_64/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p debug
	$(CXX) halide/x86_64/local_laplacian_gpu.o halide/x86_64/local_laplacian_cpu.o  main.cpp -o bin/debug/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/x86_64 -lpthread -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -DCOMPILING_FOR_OPENCL -DDEBUG
Android:main.cpp halide/android/local_laplacian_cpu.o
	@mkdir -p bin;cd bin;mkdir -p android
	$(ANDROID_CXX) halide/android/local_laplacian_gpu.o halide/android/local_laplacian_cpu.o  main.cpp -o bin/android/run -I$(HALIDE_PATH)/include -I$(FRAMEWORK_PATH)/include -Ihalide/android  -ldl -landroid -lOpenCL -lpng -std=c++11 -llog -DCOMPILING_FOR_OPENCL

clean:
	rm -rf halide/local_laplacian halide/x86_64/ halide/android/ run bin
