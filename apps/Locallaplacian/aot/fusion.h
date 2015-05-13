#ifndef FUSION_H
#define FUSION_H
#include <iostream>
#include "local_laplacian_cpu.h"
#include "local_laplacian_gpu.h"
#include <thread>
//#include <HalideRuntime.h>
#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdint.h>
typedef struct buffer_t {
	uint64_t dev;
	uint8_t* host;
	int32_t extent[4];
	int32_t stride[4];
	int32_t min[4];
	int32_t elem_size;
	bool host_dirty;
	bool dev_dirty;
} buffer_t;
#endif
using namespace std;
#ifndef BUFFER_DEFINED
#define BUFFER_DEFINED
#include <stdint.h>
typedef struct Buffer {
	Buffer(int x,int y=0,int z=0,int w=0) {}
	Buffer(buffer_t _buf,uint8_t* _ptr):buffer(_buf),ptr(_ptr) {}
	Buffer(buffer_t _buf,buffer_t _bufc,buffer_t _bufg,uint8_t* _ptr):buffer(_buf),bufc(_bufc),bufg(_bufg) ,ptr(_ptr) {};
	buffer_t buffer,bufc,bufg;
	~Buffer(){
		delete ptr;
	}
	uint8_t* ptr;

} Buffer;
#endif // BUFFET_DEFINED


#ifndef GPUTHREAD
#define GPUTHREAD
void GPUthread(int levels,float alpha,float beta,buffer_t* input,buffer_t* output,int s) {

	local_laplacian_gpu(levels, alpha/(levels-1), beta,input,output);
//	halide_copy_to_host(NULL,output);
}
#endif


extern "C" int halide_copy_to_host(void* user_context, buffer_t* buf);
template <typename T>
class Fusion {
public:
	Fusion(int _levels,int _alpha,int _beta):levels(_levels),alpha(_alpha),beta(_beta) {}
	Buffer realizeCPU(buffer_t* input);

	Buffer realizeGPU(buffer_t* input);
	Buffer realize(buffer_t* input);
	Buffer realize(buffer_t* input,int s);

private:
	uint8_t* createBuffer_t(int x,int y,int z,int w,buffer_t* buf);
	buffer_t* createBuffers(int x,int y,int z,int w,int s);
	Buffer createBuffer(int x,int y,int z ,int w,int s);
	buffer_t divBuffer(buffer_t buf,int start,int end);
	buffer_t divBuffer(buffer_t *buf,int start,int nend) ;
	Buffer createBuffer(int x,int y,int z,int w);
	int levels;
	float alpha , beta ;

};
template <typename T>
uint8_t* Fusion<T>::createBuffer_t(int x,int y,int z,int w,buffer_t* buf) {
	buf->extent[0] = x;
	buf->extent[1] = y;
	buf->extent[2] = z;
	buf->extent[3] = w;
	buf->stride[0] = 1;
	buf->stride[1] = x;
	buf->stride[2] = x*y;
	buf->stride[3] = x*y*z;
	buf->elem_size = sizeof(T);
	size_t size = 1;
	if (x) size *= x;
	if (y) size *= y;
	if (z) size *= z;
	if (w) size *= w;
	uint8_t* ptr = new uint8_t[sizeof(T)*size + 40];
	buf->host = ptr;
	buf->dev = 0;
	buf->host_dirty = false;
	buf->dev_dirty = false;
	buf->min[0] = 0;
	buf->min[1] = 0;
	buf->min[2] = 0;
	buf->min[3] = 0;
	while ((size_t)buf->host & 0x1f) buf->host++; // Memory Alias
	return ptr;
}
//Split Buffer to buffer for cpu and buffer for gpu
template <typename T>
Buffer Fusion<T>::createBuffer(int x,int y,int z,int w,int s) {
	buffer_t buf= {0};
	uint8_t* ptr=createBuffer_t(x,y,z,w,&buf);
	buffer_t bufc=divBuffer(buf,0,s);
	buffer_t bufg=divBuffer(buf,s,buf.extent[1]);

	Buffer bufer(buf,bufc,bufg,ptr);
	return bufer;
}
//Return a Full buffer
template <typename T>
Buffer Fusion<T>::createBuffer(int x,int y,int z,int w) {
	buffer_t buf= {0};
	uint8_t* ptr=createBuffer_t(x,y,z,w,&buf);
	Buffer bufer(buf,ptr);
	return bufer;
}
template <typename T>
buffer_t Fusion<T>::divBuffer(buffer_t buf,int start,int nend) {
	buffer_t buff= {0};
	buff.extent[0] = buf.extent[0];
	buff.extent[1] = nend-start;
	buff.extent[2] = buf.extent[2];
	buff.extent[3] = buf.extent[3];
	buff.stride[0] = buf.stride[0];
	buff.stride[1] = buf.stride[1];
	buff.stride[2] = buf.stride[2];
	buff.stride[3] = buf.stride[3];
	buff.elem_size = buf.elem_size;
	buff.host_dirty = false;
	buff.dev_dirty = false;
	buff.dev = 0;
	if(start==0) {
		buff.host = buf.host;
		return buff;
	}
	int offset=buf.extent[0]*start*buf.elem_size*1;
	buff.min[1]=start;
	buff.host= buf.host+offset;
	return buff;
}
template <typename T>
buffer_t Fusion<T>::divBuffer(buffer_t *buf,int start,int nend) {
	buffer_t buff= {0};
	buff.extent[0] = buf->extent[0];
	buff.extent[1] = buf->extent[1];
	buff.extent[2] = buf->extent[2];
	buff.extent[3] = buf->extent[3];
	buff.stride[0] = buf->stride[0];
	buff.stride[1] = buf->stride[1];
	buff.stride[2] = buf->stride[2];
	buff.stride[3] = buf->stride[3];
	buff.elem_size = buf->elem_size;
	buff.host_dirty = true;
	buff.dev_dirty = false;
	buff.dev = 0;
	if(start==0) {
		buff.host = buf->host;
		return buff;
	}
	int offset=buf->extent[0]*start*buf->elem_size*1;
	buff.min[1]=start;
	buff.host= buf->host+offset;
	return buff;
}
template <typename T>
Buffer Fusion<T>::realizeCPU(buffer_t* input) {
	Buffer output=createBuffer(input->extent[0] ,input->extent[1],input->extent[2],input->extent[3]);
	input->host_dirty=true;
	local_laplacian_cpu(levels, alpha/(levels-1),beta,input,&output.buffer);
	return output;
}
template <typename T>
Buffer Fusion<T>::realizeGPU(buffer_t* input) {
	Buffer output=createBuffer(input->extent[0] ,input->extent[1],input->extent[2],input->extent[3]);
	input->host_dirty=true;
	local_laplacian_gpu(levels, alpha/(levels-1),beta,input,&output.buffer);
	halide_copy_to_host(NULL,&output.buffer);
	return output;
}
template <typename T>
Buffer Fusion<T>::realize(buffer_t* input) {
	Buffer buf=createBuffer(input->extent[0] ,input->extent[1],input->extent[2],input->extent[3],400);
	local_laplacian_gpu(levels,alpha,beta,input,&buf.bufg);
	local_laplacian_cpu(levels, alpha/(levels-1), beta,input,&buf.bufc);
	halide_copy_to_host(NULL,&buf.bufg);
	return buf;
}
template <typename T>
Buffer Fusion<T>::realize(buffer_t* input,int s) {
	Buffer buf=createBuffer(input->extent[0] ,input->extent[1],input->extent[2],input->extent[3],s);
//		local_laplacian_gpu(levels, alpha/(levels-1), beta,input,&buf.bufg);
	std::thread gputThread(local_laplacian_gpu,levels,alpha/(levels-1),beta,input,&buf.bufg);
//		gputThread.join();



	local_laplacian_cpu(levels, alpha/(levels-1), beta,input,&buf.bufc);
		gputThread.join();
//	halide_copy_to_host(NULL,&buf.buffer);
	return buf;
}
#endif // FUSION_H
