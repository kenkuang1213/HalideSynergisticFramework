#include <Halide.h>
using namespace Halide;
int main()
{
    ImageParam input(UInt(16),3,"input");
   	Func floating("floating");
   	Var x("x"),y("y"),c("c");
	floating(x, y, c) = cast<float>(input(x, y, c)) / 65535.0f;

	// Set a boundary condition
	Func clamped("clamped");
	clamped(x, y, c) = floating(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1), c);

	// Get the luminance channel
	Func gray("gray"),output("output");
	gray(x, y) = 0.299f * clamped(x, y, 0) + 0.587f * clamped(x, y, 1) + 0.114f * clamped(x, y, 2);
	output(x, y) = cast<uint16_t>(clamp(gray(x, y), 0.0f, 1.0f) * 65535.0f);
       Halide::Var _x0, _y1;
        output.split(x, x, _x0, 4)
        .split(y, y, _y1, 32)
        .reorder(_y1, _x0, y, x)
        .reorder_storage(x, y)
        .vectorize(_y1, 8)
        .parallel(x)
        .compute_root()
        ;

    output.compile_to_file("halide_gray",input);
}
