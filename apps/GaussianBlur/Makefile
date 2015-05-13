OPENCL_LDFLAGS=-L/opt/AMDAPPSDK-3.0-0-Beta/lib/x86_64/sdk -lOpenCL
ANDROID_OPENCL_LDFALGS=-lOpenCL
ANDROID_CXX?=arm-linux-androideabi-g++
ANDROID_CXX_FLAGS=-llog -landroid -lOpenCL -I src/support -I src/fusion -I src/android

ANDROID_PNGFLAGS=-lpng

PNGFLAGS?=$(shell libpng-config --L_opts )
PNGFLAGS+=$(shell libpng-config --libs)
CXX=g++
CXX_FLAGS+=-I src/support -I src/fusion -I src/halide
all:Release

src/halide/gaussinBlur: src/halide/gaussinBlur.cpp
	$(CXX) src/halide/gaussinBlur.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o src/halide/gaussinBlur -lpthread -ldl -lz -lHalide
	@rm src/halide/gaussinBlur_* -rf
	@rm src/android/* -rf
src/halide/gray: src/halide/gray.cpp
	$(CXX) src/halide/gray.cpp -I$(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o src/halide/gray -lpthread -ldl -lz -lHalide
src/halide/halide_gray.o: src/halide/gray
	@cd src/halide; ./gray
src/halide/gaussinBlur_cpu.o: src/halide/gaussinBlur
	@cd src/halide; ./gaussinBlur
src/halide/gaussinBlur_gpu.o: src/halide/gaussinBlur
	cd src/halide; HL_TARGET=host-opencl ./gaussinBlur
src/android/halide_gray.o: src/halide/gray
	@cd src/;mkdir -p android;cd android;HL_TARGET=arm-32-android ../halide/gray
src/android/gaussinBlur_cpu.o: src/halide/gaussinBlur
	@cd src/;mkdir -p android;cd android;HL_TARGET=arm-32-android ../halide/gaussinBlur
src/android/gaussinBlur_gpu.o: src/halide/gaussinBlur
	@cd src/;mkdir -p android;cd android; HL_TARGET=arm-32-android-opencl  ../halide/gaussinBlur
Debug: src/main.cpp src/halide/gaussinBlur_cpu.o src/halide/gaussinBlur_gpu.o src/halide/halide_gray.o
	@mkdir -p bin;cd bin;mkdir -p Debug
	$(CXX) -g src/main.cpp src/halide/gaussinBlur_gpu.o src/halide/gaussinBlur_cpu.o  src/halide/halide_gray.o  $(CXX_FLAGS)  -lpthread -ldl -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -o bin/Debug/process

Release: src/main.cpp src/halide/gaussinBlur_gpu.o src/halide/gaussinBlur_cpu.o src/halide/halide_gray.o
	@mkdir -p bin;cd bin;mkdir -p Release
	$(CXX) -O3 src/main.cpp src/halide/gaussinBlur_gpu.o src/halide/gaussinBlur_cpu.o src/halide/halide_gray.o  $(CXX_FLAGS)  -lpthread  -ldl -std=c++11 $(PNGFLAGS) $(OPENCL_LDFLAGS) -o bin/Release/process

Android: src/main.cpp src/android/gaussinBlur_gpu.o src/android/gaussinBlur_cpu.o src/android/halide_gray.o
	@mkdir -p bin;cd bin;mkdir -p Android
	$(ANDROID_CXX) -O3 src/main.cpp src/android/gaussinBlur_gpu.o src/android/gaussinBlur_cpu.o src/android/halide_gray.o  $(ANDROID_CXX_FLAGS) -std=c++11  $(ANDROID_PNGFLAGS) -o bin/Android/process

clean:
	rm -rf src/halide/gaussinBlur src/halide/gaussinBlur_gpu.o  src/halide/gaussinBlur_gpu.h src/halide/gaussinBlur_cpu.o src/halide/gaussinBlur_cpu.h bin/ src/halide/halide_gray.h src/halide/halide_gray.o src/halide/gray src/android/
