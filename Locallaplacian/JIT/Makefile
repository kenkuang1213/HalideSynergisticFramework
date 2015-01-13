export HALIDE_PATH=/home/ken/src/Halide
export LD_LIBRARY_PATH="/home/ken/src/Halide/bin:$LD_LIBRARY_PATH"
include $(HALIDE_PATH)/apps/support/Makefile.inc
TARGET?=host
DEBUG?=0
all:jit
jit:main.cpp  CPULL.h CPULL.cpp JitInterface.h Syner.h
	 $(CXX)  -g main.cpp CPULL.cpp Syner.cpp -I $(HALIDE_PATH)/include $(HALIDE_PATH)/bin/libHalide.so -o process -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS) $(OPENCL_LDFAGS)
debug:main.cpp  CPULL.h CPULL.cpp JitInterface.h Syner.h
	 $(CXX)  main.cpp CPULL.cpp Syner.cpp -I $(HALIDE_PATH)/include $(HALIDE_PATH)/bin/libHalide.so -DDEBUG -o process -ldl -lpthread -lz $(LDFLAGS) $(PNGFLAGS) $(OPENCL_LDFAGS)
clean:
	rm -f process
