#include <Halide.h>

#define AUTOTUNE_HOOK(x)
#define BASELINE_HOOK(x)

using namespace Halide;

int main()
{
    ImageParam input(UInt(8),2,"input");
    Image<float> mask(5, 5);

    mask(0, 0) = 1;
    mask(0, 1) = 4;
    mask(0, 2) = 7;
    mask(0, 3) = 4;
    mask(0, 4) = 1;

    mask(1, 0) = 4;
    mask(1, 1) = 16;
    mask(1, 2) = 26;
    mask(1, 3) = 16;
    mask(1, 4) = 4;

    mask(2, 0) = 7;
    mask(2, 1) = 26;
    mask(2, 2) = 41;
    mask(2, 3) = 26;
    mask(2, 4) = 7;

    mask(3, 0) = 4;
    mask(3, 1) = 16;
    mask(3, 2) = 26;
    mask(3, 3) = 16;
    mask(3, 4) = 4;


    mask(4, 0) = 1;
    mask(4, 1) = 4;
    mask(4, 2) = 7;
    mask(4, 3) = 4;
    mask(4, 4) = 1;

    Expr constOfMask("constOfMask");
    constOfMask=273;

    RDom r(mask),rx(0,4,0,4);

    Func output("output"),clamped("clamped"),floating("floating"),blur("blur"),blur_x,blur_y;

    Var x("x"),y("y");

    clamped(x,y)=input(clamp(x,0,input.width()-1),clamp(y,0,input.height()-1));

   blur_x(x,y)=(clamped(x-1, y) + clamped(x, y) + clamped(x+1, y))/3;
     output(x, y) =(blur_x(x, y-1) + blur_x(x, y+1) + blur_x(x, y-1))/3;
    // blur(x,y)=blur(x,y)/constOfMask;
    // output(x, y) = cast<uint8_t>(blur_y(x, y));
 


     Func Gx,Gy,gray;
    //  Expr value=cast<float>(input(x,y));
    // gray(x,y)=value;
 
    // // Set a boundary condition
    // Func clamped;
    // clamped(x, y) = gray(clamp(x, 0, input.width()-1), clamp(y, 0, input.height()-1));
    // //
    // //algorithm part
    // Gx(x,y)=(clamped(x,y)-clamped(x+1,y+1))*(clamped(x,y)-clamped(x+1,y+1));
    // Gy(x,y)=(clamped(x+1,y)-clamped(x,y+1))*(clamped(x+1,y)-clamped(x,y+1));        

    // Gx(x,y)=(blur(x,y)-blur(x+1,y+1))*(blur(x,y)-blur(x+1,y+1));
    // Gy(x,y)=(blur(x+1,y)-blur(x,y+1))*(blur(x+1,y)-blur(x,y+1));      

    // output(x,y)=select(((Gx(x,y)+Gy(x,y))>cast<float>(169)),cast<uint8_t>(0),cast<uint8_t>(blur(x,y)));


    AUTOTUNE_HOOK(output);



    Target target = get_target_from_environment();
    if (target.has_gpu_feature())
    {
        if(target.os==Target::Android)
        {
           // clamped.compute_root().vectorize(x,8).gpu_tile(x, y, 4, 4, GPU_Default);
           // floating.compute_root().gpu_tile(x, y, 16, 4, GPU_Default);
           blur_x.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,4, GPU_Default);
           // blur_y.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            // Gx.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            //  Gy.compute_root().vectorize(x,8).gpu_tile(x, y, 16 ,8, GPU_Default);
            output.compute_root().vectorize(x,8).gpu_tile(x, y, 16, 8, GPU_Default);
        }
        else
        {
//            clamped.compute_root().gpu_tile(x, y, 16, 8, GPU_Default);
//            floating.compute_root().gpu_tile(x, y, 16, 4, GPU_Default);
            // blur.compute_root().vectorize(x,8).gpu_tile(x, y, 4, 4, GPU_Default);
            output.compute_root().gpu_tile(x, y, 16, 8, GPU_Default);

        }
        output.compile_to_file("gaussinBlur_gpu", input, target);
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
    blur_x.vectorize(x, 8).compute_root();

        output.parallel(y).vectorize(x, 8);

        output.compile_to_file("gaussinBlur_cpu", input, target);
    }
    BASELINE_HOOK(output)




}
