#include <Halide.h>
using namespace Halide;
int main(){
	ImageParam input(UInt(8),2,"input");
	Image<uint8_t> Gx(3, 3);
	Gx(0, 0) = -1;
	Gx(0, 1) = -2;
	Gx(0, 2) = -1;
	Gx(1, 0) = 0;
	Gx(1, 1) = 0;
	Gx(1, 2) = 0;
	Gx(2, 0) = 1;
	Gx(2, 1) = 2;
	Gx(2, 2) = 1;

	Image<uint8_t> Gy(3, 3);
	Gy(0, 0) = -1;
	Gy(0, 1) = 0;
	Gy(0, 2) = 1;
	Gy(1, 0) = -2;
	Gy(1, 1) = 0;
	Gy(1, 2) = 2;
	Gy(2, 0) = -1;
	Gy(2, 1) = 0;
	Gy(2, 2) = 1;
	Func clamped("clamp");
	Var x("x"),y("y");
	clamped(x,y)=cast<float>(input(clamp(x,0,input.width()-1),clamp(y,0,input.height()-1)));
	RDom r(Gx);
	Func FGX("FGX"),FGY("FGY"),output("output");
	FGX(x,y)=(clamped(x,y)-clamped(x+1,y+1))*(clamped(x,y)-clamped(x+1,y+1));
	FGY(x,y)=(clamped(x+1,y)-clamped(x,y+1))*(clamped(x+1,y)-clamped(x,y+1));        


    	// FGX(x,y)=-clamped(x-1,y-1)-2*clamped(x,y-1)-clamped(x+1,y-1)+clamped(x,y+1)+clamped(x-1,y+1)+2*clamped(x,y+1)+clamped(x+1,y+1);
    	// FGY(x,y)=-clamped(x-1,y-1)+clamped(x+1,y-1)-(2*clamped(x-1,y))+(2*clamped(x+1,y))-clamped(x-1,y+1)+clamped(x+1,y+1);
	// FGX(x,y)+=(Gx(r.x,r.y)*clamped(x+r.x-1,y+r.y-1));
	// FGY(x,y)+=(Gy(r.x,r.y)*clamped(x+r.x-1,y+r.y-1));



	output(x,y)=select((FGX(x,y)+FGY(x,y))>cast<float>(169),cast<uint8_t>(0),cast<uint8_t>(255));

	    Target target = get_target_from_environment();
	 if (target.has_gpu_feature())
    {
        if(target.os==Target::Android)
        {
           // clamped.compute_root().vectorize(x,8).gpu_tile(x, y, 4, 4, GPU_Default);
           // floating.compute_root().gpu_tile(x, y, 16, 4, GPU_Default);
      
           // blur_y.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            // FGX.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            //  FGY.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            output.compute_root().gpu_tile(x, y, 8, 8, GPU_Default);
        }
        else
        {
//            clamped.compute_root().gpu_tile(x, y, 16, 8, GPU_Default);
//            floating.compute_root().gpu_tile(x, y, 16, 4, GPU_Default);
            // blur.compute_root().vectorize(x,8).gpu_tile(x, y, 4, 4, GPU_Default);
            output.compute_root().gpu_tile(x, y, 16, 8, GPU_Default);

        }
        output.compile_to_file("sobel_gpu", input, target);
    }
    else
    {
        Halide::Var _x0, _x2, _x4, _y5, _x6, _x8, _y9,yi;
        // blur.split(x, x, _x0, 32)
        // .reorder(_x0, x, y)
        // .reorder_storage(x, y)
        // .vectorize(_x0, 4)
        // .parallel(y)
        // .compute_root()
        // ;
        // clamped.split(x, x, _x2, 32)
        // .reorder(_x2, x, y)
        // .reorder_storage(y, x)
        // .vectorize(_x2, 8)
        // .compute_at(floating, y)
        // ;
        // floating.split(x, x, _x4, 16)
        // .split(y, y, _y5, 4)
        // .reorder(_x4, _y5, x, y)
        // .reorder_storage(x, y)
        // .vectorize(_x4, 8)
        // .parallel(y)
        // .compute_root()
        // ;
        // blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);


        output.vectorize(x, 8);

        output.compile_to_file("sobel_cpu", input, target);
    }

}