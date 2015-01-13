#include "CPULL.h"
//#include <iostream>
Halide::Func CPULL::downsample(Halide::Func f) {
    Halide::Func downx, downy;
    downx(x, y,Halide::_) = (f(2*x-1, y, Halide::_) + 3.0f * (f(2*x, y,Halide::_) + f(2*x+1, y, Halide::_)) + f(2*x+2, y, Halide::_)) / 8.0f;
    downy(x, y, Halide::_) = (downx(x, 2*y-1, Halide::_) + 3.0f * (downx(x, 2*y, Halide::_) + downx(x, 2*y+1, Halide::_)) + downx(x, 2*y+2, Halide::_)) / 8.0f;
    return downy;
}
Halide::Func CPULL::upsample(Halide::Func f) {
		Halide::Func upx, upy;

		upx(x, y, Halide::_) = 0.25f * f((x/2) - 1 + 2*(x % 2), y, Halide::_) + 0.75f * f(x/2, y, Halide::_);
		upy(x, y, Halide::_) = 0.25f * upx(x, (y/2) - 1 + 2*(y % 2), Halide::_) + 0.75f * upx(x, y/2, Halide::_);

		return upy;

	}
void CPULL::Algorithm(){
	/* THE ALGORITHM */

    // Make the remapping function as a lookup table.
    
    Halide::Expr fx = Halide::cast<float>(x) / 256.0f;
    remap(x) = alpha*fx*exp(-fx*fx/2.0f);

    // Convert to floating point
  
    floating(x, y, c) = Halide::cast<float>(input(x, y, c)) / 65535.0f;

    // Set a boundary condition

    clamped(x, y, c) = floating(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1), c);
    // Get the luminance channel

    gray(x, y) = 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2);
	
    // Make the processed Gaussian pyramid.

    // Do a lookup into a lut with 256 entires per intensity level
    Halide::Expr level = k * (1.0f / (levels - 1));
    Halide::Expr idx = gray(x, y)*Halide::cast<float>(levels-1)*256.0f;
    idx = clamp(Halide::cast<int>(idx), 0, (levels-1)*256);
    gPyramid[0](x, y, k) = beta*(gray(x, y) - level) + level + remap(idx - 256*k);
    for (int j = 1; j < J; j++) {
        gPyramid[j](x, y, k) = downsample(gPyramid[j-1])(x, y, k);
    }

    // Get its laplacian pyramid

    lPyramid[J-1](x, y, k) = gPyramid[J-1](x, y, k);
    for (int j = J-2; j >= 0; j--) {
        lPyramid[j](x, y, k) = gPyramid[j](x, y, k) - upsample(gPyramid[j+1])(x, y, k);
    }

    // Make the Gaussian pyramid of the input

    inGPyramid[0](x, y) = gray(x, y);
    for (int j = 1; j < J; j++) {
        inGPyramid[j](x, y) = downsample(inGPyramid[j-1])(x, y);
    }

    // Make the laplacian pyramid of the output

    for (int j = 0; j < J; j++) {
        // Split input pyramid value into integer and floating parts
        Halide::Expr level = inGPyramid[j](x, y) * Halide::cast<float>(levels-1);
        Halide::Expr li = clamp(Halide::cast<int>(level), 0, levels-2);
        Halide::Expr lf = level - Halide::cast<float>(li);
        // Linearly interpolate between the nearest processed pyramid levels
        outLPyramid[j](x, y) = (1.0f - lf) * lPyramid[j](x, y, li) + lf * lPyramid[j](x, y, li+1);
    }

    // Make the Gaussian pyramid of the output

    outGPyramid[J-1](x, y) = outLPyramid[J-1](x, y);
    for (int j = J-2; j >= 0; j--) {
        outGPyramid[j](x, y) = upsample(outGPyramid[j+1])(x, y) + outLPyramid[j](x, y);
    }

    // Reintroduce color (Connelly: use eps to avoid scaling up noise w/ apollo3.png input)

    float eps = 0.01f;
    color(x, y, c) = outGPyramid[0](x, y) * (clamped(x, y, c)+eps) / (gray(x, y)+eps);
    // Convert back to 16-bit
    output(x, y, c) = Halide::cast<uint16_t>(clamp(color(x, y, c), 0.0f, 1.0f) * 65535.0f);
}
Halide::Buffer CPULL::Realize(int x,int y){
	Halide::Buffer buf(Halide::UInt(16),x,y,3);
	output.realize(buf);
	buf.copy_to_host();
	return buf;
}
Halide::Buffer CPULL::Realize(Halide::Buffer buf){
	output.realize(buf);
	buf.copy_to_host();
	return buf;
}
Halide::Buffer CPULL::Realize(int x,int y,Halide::Buffer &buf,int cpuHeight){

}
void CPULL::Schedule(){
	
	
	remap.compute_root();
	Halide::Var yi;
	output.parallel(y, 4).vectorize(x, 8);
	gray.compute_root().parallel(y, 4).vectorize(x, 8);
	for (int j = 0; j < 4; j++) {
		if (j > 0) inGPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
		if (j > 0) gPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
		outGPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
	}
	for (int j = 4; j < J; j++) {
		inGPyramid[j].compute_root().parallel(y);
		gPyramid[j].compute_root().parallel(k);
		outGPyramid[j].compute_root().parallel(y);
	}
	output.compile_jit();
//
//	Chucked Schedule
//	remap.compute_at(output,y);
//	Halide::Var yi;
//	output.split(y,y,yi,64).parallel(y, 4).vectorize(x, 8);
//	gray.compute_at(output,y)vectorize(x, 8);
//	for (int j = 0; j < 4; j++) {
////		if (j > 0) inGPyramid[j].parallel(y, 4).vectorize(x, 8);
//		if (j > 0) inGPyramid[j].compute_at(output,y);
//		if (j > 0) gPyramid[j].compute_at(output,y);
//		outGPyramid[j].compute_at(output,y);
//	}
//	for (int j = 4; j < J; j++) {
//		inGPyramid[j].compute_at(output,y);
//		gPyramid[j].compute_at(output,y);
//		outGPyramid[j].compute_at(output,y);
//	}
//	output.compile_jit();

}
void CPULL::Schedule(Halide::Target target){
	remap.compute_root();
	if (target.has_gpu_feature()) {
        // gpu schedule
        output.compute_root().gpu_tile(x, y, 16, 8, Halide::GPU_Default);
        for (int j = 0; j < J; j++) {
            int blockw = 16, blockh = 8;
            if (j > 3) {
                blockw = 2;
                blockh = 2;
            }
            if (j > 0) inGPyramid[j].compute_root().gpu_tile(x, y, blockw, blockh, Halide::GPU_Default);
            if (j > 0) gPyramid[j].compute_root().reorder(k, x, y).gpu_tile(x, y, blockw, blockh, Halide::GPU_Default);
            outGPyramid[j].compute_root().gpu_tile(x, y, blockw, blockh, Halide::GPU_Default);
        }
    } else {
        // cpu schedule
        Halide::Var yi;
        output.parallel(y, 4).vectorize(x, 8);
        gray.compute_root().parallel(y, 4).vectorize(x, 8);
        for (int j = 0; j < 4; j++) {
            if (j > 0) inGPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
            if (j > 0) gPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
            outGPyramid[j].compute_root().parallel(y, 4).vectorize(x, 8);
        }
        for (int j = 4; j < J; j++) {
            inGPyramid[j].compute_root().parallel(y);
            gPyramid[j].compute_root().parallel(k);
            outGPyramid[j].compute_root().parallel(y);
        }
    }
    output.compile_jit(target);
}
