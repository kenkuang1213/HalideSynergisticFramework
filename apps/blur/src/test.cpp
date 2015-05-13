#include <emmintrin.h>
#include <math.h>
#include <sys/time.h>
#include <stdint.h>
#include <cstdio>
#include "static_image.h"
#include "image_io.h"
//#define cimg_display 0
//#include "CImg.h"
//using namespace cimg_library;

timeval t1, t2;
#define begin_timing gettimeofday(&t1, NULL); for (int i = 0; i < 10; i++) {
#define end_timing } gettimeofday(&t2, NULL);

// typedef CImg<uint16_t> Image;

Image<uint16_t> blur(Image<uint16_t> in) {
    Image<uint16_t> tmp(in.width()-8, in.height());
    Image<uint16_t> out(in.width()-8, in.height()-2);

//    begin_timing;

    for (int y = 0; y < tmp.height(); y++)
        for (int x = 0; x < tmp.width(); x++)
            tmp(x, y) = (in(x, y) + in(x+1, y) + in(x+2, y))/3;

    for (int y = 0; y < out.height(); y++)
        for (int x = 0; x < out.width(); x++)
            out(x, y) = (tmp(x, y) + tmp(x, y+1) + tmp(x, y+2))/3;

//    end_timing;

    return out;
}




extern "C" {
#include "halide_blur.h"
}

Image<uint16_t> blur_halide(Image<uint16_t> in) {
    Image<uint16_t> out(in.width()-8, in.height()-2);

    // Call it once to initialize the halide runtime stuff
//    halide_blur(in, out);

//    begin_timing;

    // Compute the same region of the output as blur_fast (i.e., we're
    // still being sloppy with boundary conditions)
    halide_blur(in, out);

//    end_timing;

    return out;
}

int main(int argc, char **argv) {

    Image<uint16_t> input(6408, 4802);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }

    Image<uint16_t> blurry = blur(input);
//    float slow_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;


    //Image<uint16_t> speedy2 = blur_fast2(input);
    //float fast_time2 = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

    Image<uint16_t> halide = blur_halide(input);
//    float halide_time = (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec) / 1000000.0f;

    // fast_time2 is always slower than fast_time, so skip printing it
//    printf("times: %f %f %f\n", slow_time, fast_time, halide_time);
//
	std::string name="a.png";
//	halide.copy_to_host();
	save (halide,name);
    for (int y = 64; y < input.height() - 64; y++) {
        for (int x = 64; x < input.width() - 64; x++) {
            if ( blurry(x, y) != halide(x, y)){
                printf("difference at (%d,%d): %d %d \n", x, y, blurry(x, y),  halide(x, y));
				return -1;
            }
        }
    }
    

    return 0;
}
