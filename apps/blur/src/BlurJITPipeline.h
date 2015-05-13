#ifndef BLURJTPIPELINE_H_H
#define BLURJTPIPELINE_H_H
#include "PipelineJITInterface.h"
#include "clock.h"
#include <cstdio>
template<typename T>
class BlurJITPipeline:  public   PipelineJITInterface <T>{
public:
	Halide::Image< T > input;
	BlurJITPipeline(Halide::Image<T> in):input(in){}

	void CPUSchedule ();
	void GPUSchedule ();
	Halide::Image<T>  Realize(int x,int y);
	void test_performance(int x,int y);

private:
	Halide::Func blur_x;
	Halide::Func blur_y;
	Halide::Var x;
	Halide::Var y;
	Halide::Var yi;
	Halide::Var xi;



};

template <typename T>

template <typename T>
Halide::Image<T> BlurJITPipeline<T>::Realize(int x,int y){
	Halide::Image<T> out=blur_y.realize(x,y);
	return out;
}
template <typename T>
void BlurJITPipeline<T>::test_performance(int x,int y) {
	// Test the performance of the scheduled Pipeline.

	// If we realize curved into a Halide::Image, that will
	// unfairly penalize GPU performance by including a GPU->CPU
	// copy in every run. Halide::Image objects always exist on
	// the CPU.

	// Halide::Buffer, however, represents a buffer that may
	// exist on either CPU or GPU or both.
	Halide::Buffer output(Halide::UInt(16), x, y);
	// Run the filter once to initialize any GPU runtime state.
	blur_y.realize(output);
	// Now take the best of 3 runs for timing.
	double best_time;
	for (int i = 0; i < 3; i++) {
		double t1 = current_time();
		// Run the filter 100 times.
		for (int j = 0; j < 100; j++) {
			blur_y.realize(output);
		}
		// Force any GPU code to finish by copying the buffer back to the CPU.
		output.copy_to_host();
		double t2 = current_time();

		double elapsed = (t2 - t1)/100;
		printf("%1.4f milliseconds\n", elapsed);
		if (i == 0 || elapsed < best_time) {
			best_time = elapsed;
		}
	}
	printf("%1.4f milliseconds\n", best_time);
}

template <typename T>

void BlurJITPipeline< T>::Algorithm(){
    blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;
//    
}
template <typename T>
void BlurJITPipeline< T>::CPUSchedule (){
	blur_y.split(y, y, yi, 8).parallel(y).vectorize(x, 8);
	blur_x.store_at(blur_y, y).compute_at(blur_y, yi).vectorize(x, 8);  
	blur_y.compile_jit();
}
template <typename T>
void BlurJITPipeline< T>::GPUSchedule (){
	blur_y.compute_root().gpu_tile(x,y,16,16);
	blur_x.compute_at(blur_y,Halide::Var::gpu_threads());
	Halide::Target target = Halide::get_host_target();
	target.set_feature(Halide::Target::OpenCL); 
	blur_y.compile_jit(target);
}



#endif // BLURJTPIPELINE_H_H
