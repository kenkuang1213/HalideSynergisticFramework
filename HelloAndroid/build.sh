#!/bin/bash
set -e
android update project -p . --target android-19
cd jni
c++ gaussinBlur.cpp -L $HALIDE_PATH/bin -lHalide -I $HALIDE_PATH/include -ldl -lpthread -lz -o gaussinBlur
c++ sobel.cpp -L $HALIDE_PATH/bin -lHalide -I $HALIDE_PATH/include -ldl -lpthread -lz -o sobel
# 64-bit MIPS (mips-64-android,mips64) currently does not build since
# llvm will not compile for the R6 version of the ISA without Nan2008
# and the gcc toolchain used by the Android build setup requires those
# two options together.
for archs in arm-32-android,armeabi arm-32-android-armv7s,armeabi-v7a ; do
    IFS=,
    set $archs
    hl_target=$1
    android_abi=$2
    mkdir -p halide_generated_$android_abi
    cd halide_generated_$android_abi
    HL_TARGET=$hl_target DYLD_LIBRARY_PATH=$HALIDE_PATH/bin LD_LIBRARY_PATH=$HALIDE_PATH/bin:$LD_LIBRARY_PATH ../gaussinBlur
	 HL_TARGET=$hl_target-opencl DYLD_LIBRARY_PATH=$HALIDE_PATH/bin LD_LIBRARY_PATH=$HALIDE_PATH/bin:$LD_LIBRARY_PATH ../gaussinBlur
    HL_TARGET=$hl_target DYLD_LIBRARY_PATH=$HALIDE_PATH/bin LD_LIBRARY_PATH=$HALIDE_PATH/bin:$LD_LIBRARY_PATH ../sobel
	 HL_TARGET=$hl_target-opencl DYLD_LIBRARY_PATH=$HALIDE_PATH/bin LD_LIBRARY_PATH=$HALIDE_PATH/bin:$LD_LIBRARY_PATH ../sobel
    cd ..
    unset IFS
done

cd ..
pwd
ndk-build # NDK_LOG=1
ant debug
adb install -r bin/RobertsOperation-debug.apk
#adb logcat
