# Can't use "APP_ABI = all" as 64-bit MIPS currently does not build since
# llvm will not compile for the R6 version of the ISA without Nan2008
# and the gcc toolchain used by the Android build setup requires those
# two options together.
APP_ABI := armeabi armeabi-v7a 
APP_PLATFORM := android-19
APP_STL:=gnustl_static
APP_CPPFLAGS += -std=c++11
NDK_TOOLCHAIN_VERSION=4.8
