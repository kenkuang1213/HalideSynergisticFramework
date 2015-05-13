#ifndef BLURJITCG_H_
#define BLURJITCG_H_
#include "Halide.h"
#include <cstdio>
#include "clock.h"
#include <string>
#include <pthread.h>
#include "PipelineJITInterface.h"
using namespace Halide;
struct PthreadArgs{
	Halide::Buffer buf;
	Halide::Func GPU;
};
template <typename T>
class blurJITCG :public PipelineJITInterface <T>{
public:
	blurJITCG(Image<T> in):input(in){}
	void Algorithm();
	void CPUAlgorithm();
	void GPUAlgorithm();
	void CPUAlgorithm(int startX,int endX,int startY,int endY);
	void GPUAlgorithm(int startX,int endX,int startY,int endY);
	void CPUSchedule ();
	Halide::Image<T>  Realize(std::string target,int x,int y);
	void GPUSchedule ();
	Halide::Image< T > input;
	Halide::Image<T>  Realize(int x,int y);
	void test_CPUperformance(int x,int y);
	void test_GPUperformance(int x,int y);
	void test_CGperformance(int x,int y);
private:
	pthread_t thread;
	int CPUSize[4];
	int GPUSize[4];
	Halide::Func CPUblur_x;
	Halide::Func CPUblur_y;
	Halide::Func GPUblur_x;
	Halide::Func GPUblur_y;
	Halide::Var CpuX;
	Halide::Var CpuY;
	Halide::Var GpuX;
	Halide::Var GpuY;
	Halide::Var yi;
	Halide::Var xi;
	Expr CPU_clamped_x;
	Expr CPU_clamped_y;
	Expr GPU_clamped_x;
	Expr GPU_clamped_y;
	static void* GPUThread(void* arguments){
		struct PthreadArgs* args=(struct PthreadArgs*)arguments;
//		args=(Args*)arguments;
//		args->GPU.realize(args->buf);
		args->buf.copy_to_host();
//		gpuFunc.realize(buf);
//		buf.copy_to_host();	
	  return NULL;
	}

};
template <typename T>
void blurJITCG<T>::CPUAlgorithm(){
	CPUblur_x(CpuX, CpuY) = (input(CpuX, CpuY) + input(CpuX+1, CpuY) + input(CpuX+2, CpuY))/3;
    CPUblur_y(CpuX, CpuY) = (CPUblur_x(CpuX, CpuY) + CPUblur_x(CpuX, CpuY+1) + CPUblur_x(CpuX, CpuY+2))/3;

}
template <typename T>
void blurJITCG<T>::GPUAlgorithm(){
	GPUblur_x(GpuX, GpuY) = (input(GpuX, GpuY) + input(GpuX+1, GpuY) + input(GpuX+2, GpuY))/3;
    GPUblur_y(GpuX, GpuY) = (GPUblur_x(GpuX, GpuY) + GPUblur_x(GpuX, GpuY+1) + GPUblur_x(GpuX, GpuY+2))/3;
}
template <typename T>
void blurJITCG<T>::CPUAlgorithm(int startX,int endX,int startY,int endY){
	CPUSize[0]=startX;
	CPUSize[1]=endX;
	CPUSize[2]=startY;
	CPUSize[3]=endY;
	CPUblur_x(CpuX, CpuY) = (input(CpuX, CpuY) + input(CpuX+1, CpuY) + input(CpuX+2, CpuY))/3;
    CPUblur_y(CpuX, CpuY) = (CPUblur_x(CpuX, CpuY) + CPUblur_x(CpuX, CpuY+1) + CPUblur_x(CpuX, CpuY+2))/3;
}
template <typename T>
void blurJITCG<T>::GPUAlgorithm(int startX,int endX,int startY,int endY){
	GPUSize[0]=startX;
	GPUSize[1]=endX;
	GPUSize[2]=startY;
	GPUSize[3]=endY;
	GPU_clamped_x = clamp(GpuX, startX, endX);
	GPU_clamped_y = clamp(GpuY, startY, endY);
	GPUblur_x(GpuX, GpuY) = (input(GpuX, GpuY) + input(GpuX+1, GpuY) + input(GpuX+2, GpuY))/3;
    GPUblur_y(GpuX, GpuY) = (GPUblur_x(GpuX, GpuY) + GPUblur_x(GpuX, GpuY+1) + GPUblur_x(GpuX, GpuY+2))/3;
}
template <typename T>
void blurJITCG< T>::CPUSchedule (){
	CPUblur_y.split(CpuY, CpuY, yi, 8).parallel(CpuY).vectorize(CpuX, 8);
	CPUblur_x.store_at(CPUblur_y, CpuY).compute_at(CPUblur_y, yi).vectorize(CpuX, 8);  
	CPUblur_y.compile_jit();
}
template <typename T>
void blurJITCG< T>::GPUSchedule (){
	GPUblur_y.compute_root().gpu_tile(GpuX,GpuY,16,16);
	GPUblur_x.compute_at(GPUblur_y,Halide::Var::gpu_threads());
	Halide::Target target = Halide::get_host_target();
	std::vector<Target::Feature> gpuFeatures;
	gpuFeatures.push_back(Halide::Target::OpenCL);
//	gpuFeatures.push_back(Halide::Target::Debug);
	target.set_features(gpuFeatures); 
	GPUblur_y.compile_jit(target);
}
template <typename T>
Halide::Image<T> blurJITCG<T>::Realize(std::string target,int x,int y){
	Halide::Image<T> out;
	if(target=="CPU"){
		 out=CPUblur_y.realize(x,y);
	}
	else if(target=="GPU"){
			
		out=GPUblur_y.realize(x,y);
//		out.copy_to_host;
	}
	else if(target=="CG"){
		Halide::Buffer buf(Halide::UInt(16),x,y);
		Halide::Buffer CPUbuf(Halide::UInt(16),CPUSize[1],(CPUSize[3]-CPUSize[2]+1),0,0,buf.raw_buffer()->host);
		Halide::Buffer GPUbuf(Halide::UInt(16),GPUSize[1],GPUSize[3]-GPUSize[2]+1,0,0,buf.raw_buffer()->host+(GPUSize[1]*(CPUSize[3]-1)*buf.raw_buffer()->elem_size));
		GPUbuf.set_min(0,CPUSize[3]-1);
		struct PthreadArgs args;
		args.GPU=GPUblur_y;
		args.buf=GPUbuf;
		GPUblur_y.realize(GPUbuf);
		pthread_create(&thread, NULL, blurJITCG::GPUThread, &args);
		CPUblur_y.realize(CPUbuf);
		
		Halide::Image<T> CG(buf);
		return CG;
	}
	return out;
}
template <typename T>
void blurJITCG<T>::test_CPUperformance(int x,int y) {
	// Test the performance of the scheduled Pipeline.

	// If we realize curved into a Halide::Image, that will
	// unfairly penalize GPU performance by including a GPU->CPU
	// copy in every run. Halide::Image objects always exist on
	// the CPU.

	// Halide::Buffer, however, represents a buffer that may
	// exist on either CPU or GPU or both.
	Halide::Buffer output(Halide::UInt(16), x, y);
	// Run the filter once to initialize any GPU runtime state.
	CPUblur_y.realize(output);
	// Now take the best of 3 runs for timing.
	double best_time;
	for (int i = 0; i < 3; i++) {
		double t1 = current_time();
		// Run the filter 100 times.
		for (int j = 0; j < 100; j++) {
			CPUblur_y.realize(output);
			output.copy_to_host();
		}
		// Force any GPU code to finish by copying the buffer back to the CPU.
		
		double t2 = current_time();

		double elapsed = (t2 - t1)/100;
//		printf("%1.4f milliseconds\n", elapsed);
		if (i == 0 || elapsed < best_time) {
			best_time = elapsed;
		}
	}
	printf("%1.4f milliseconds\n", best_time);
}
template <typename T>
void blurJITCG<T>::test_GPUperformance(int x,int y) {
	// Test the performance of the scheduled Pipeline.

	// If we realize curved into a Halide::Image, that will
	// unfairly penalize GPU performance by including a GPU->CPU
	// copy in every run. Halide::Image objects always exist on
	// the CPU.

	// Halide::Buffer, however, represents a buffer that may
	// exist on either CPU or GPU or both.
	Halide::Buffer output(Halide::UInt(16), x, y);
	// Run the filter once to initialize any GPU runtime state.
	GPUblur_y.realize(output);
	// Now take the best of 3 runs for timing.
	double best_time;
	for (int i = 0; i < 3; i++) {
		double t1 = current_time();
		// Run the filter 100 times.
		for (int j = 0; j < 100; j++) {
			GPUblur_y.realize(output);
			output.copy_to_host();
		}
		// Force any GPU code to finish by copying the buffer back to the CPU.
		
		double t2 = current_time();

		double elapsed = (t2 - t1)/100;
//		printf("%1.4f milliseconds\n", elapsed);
		if (i == 0 || elapsed < best_time) {
			best_time = elapsed;
		}
	}
	printf("%1.4f milliseconds\n", best_time);
}
template <typename T>
void blurJITCG<T>::test_CGperformance(int x,int y) {
	// Test the performance of the scheduled Pipeline.
//
//	// If we realize curved into a Halide::Image, that will
//	// unfairly penalize GPU performance by including a GPU->CPU
//	// copy in every run. Halide::Image objects always exist on
//	// the CPU.
//
//	// Halide::Buffer, however, represents a buffer that may
//	// exist on either CPU or GPU or both.
//	Halide::Buffer output(Halide::Int(32), x, y);
//	// Run the filter once to initialize any GPU runtime state.
//	GPUblur_y.realize(output);
//	// Now take the best of 3 runs for timing.
	Halide::Buffer buf(Halide::UInt(16),x,y);
	Halide::Buffer CPUbuf(Halide::UInt(16),CPUSize[1],(CPUSize[3]-CPUSize[2]+1),0,0,buf.raw_buffer()->host);
	Halide::Buffer GPUbuf(Halide::UInt(16),GPUSize[1],GPUSize[3]-GPUSize[2]+1,0,0,buf.raw_buffer()->host+(GPUSize[1]*(CPUSize[3]-1)*buf.raw_buffer()->elem_size));
	GPUbuf.set_min(0,CPUSize[3]-1);
	CPUblur_y.realize(CPUbuf);
	GPUblur_y.realize(GPUbuf);
	double best_time;
	for (int i = 0; i < 3; i++) {
		double t1 = current_time();
		// Run the filter 100 times.
		for (int j = 0; j < 100; j++) {
//			GPUblur_y.realize(GPUbuf);
			struct PthreadArgs args;
			args.GPU=GPUblur_y;
			args.buf=GPUbuf;
			GPUblur_y.realize(GPUbuf);
			pthread_create(&thread, NULL, blurJITCG::GPUThread, &args);
			CPUblur_y.realize(CPUbuf);
//			GPUblur_y.realize(GPUbuf);
//			GPUbuf.copy_to_host();	
			void* unused = NULL;
			pthread_join(thread, &unused);
			
			// Force any GPU code to finish by copying the buffer back to the CPU.
			

		}
		
		double t2 = current_time();

		double elapsed = (t2 - t1)/100;
//		printf("%1.4f milliseconds\n", elapsed);
		if (i == 0 || elapsed < best_time) {
			best_time = elapsed;
		}
	}
	printf("%1.4f milliseconds\n", best_time);
}

#endif // BLURJITCG_H_
