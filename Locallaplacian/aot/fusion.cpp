#include "fusion.h"
buffer_t Fusion::createBuffer(int x,int y,int z,int w) {
	buffer_t buf = {0};
	buf.extent[0] = x;
	buf.extent[1] = y;
	buf.extent[2] = z;
	buf.extent[3] = w;
	buf.stride[0] = 1;
	buf.stride[1] = x;
	buf.stride[2] = x*y;
	buf.stride[3] = x*y*z;
	buf.elem_size = sizeof(T);
	size_t size = 1;
	if (x) size *= x;
	if (y) size *= y;
	if (z) size *= z;
	if (w) size *= w;
	uint8_t* ptr = new uint8_t[sizeof(T)*size + 40];
	buf.host = ptr;
	buf.host_dirty = false;
	buf.dev_dirty = false;
	buf.dev = 0;
	while ((size_t)buf.host & 0x1f) buf.host++;
	return buf;
//	contents = new Contents(buf, ptr);
}
buffer_t Fusion::realizeCPU(buffer_t *input) {
	buffer_t output=createBuffer(buf->extent[0] ,buf->extent[1],buf->extent[2],0);
	local_laplacian_cpu(levels,alpha,beta,input,output);
	return output;
}
buffer_t Fusion::realizeGPU(buffer_t *input) {
	buffer_t output=createOutput(buf->extent[0] ,buf->extent[1],buf->extent[2],0);
	local_laplacian_gpu(levels,alpha,beta,input,output);
	return output;
}
void Fusion::realizeCPU(buffer_t* input,buffer_t* output) {
	local_laplacian_cpu(levels,alpha,beta,input,output);
}
void Fusion::realizeGPU(buffer_t* input,buffer_t* output) {
	local_laplacian_gpu(levels,alpha,beta,input,output);
}
buffer_t Fusion::realizeGPU(buffer_t* input,int x,int y) {
	buffer_t output=createOutput(x,y,buf->extent[2],0);
	local_laplacian_gpu(levels,alpha,beta,&input,&output);
	return output;
}
buffer_t Fusion::realizeCPU(buffer_t* input,int x,int y) {
	buffer_t output=createOutput(x,y,buf.extent[2],0);
	local_laplacian_cpu(levels,alpha,beta,input,output);
	return output;
}
buffer_t Fusion::realize(buffer_t input) {
	buffer_t outputCPU=createOutput(buf.extent[0] ,200,buf.extent[2],0);
	buffer_t outputGPU=createOutput(buf.extent[0] ,buf.extent[1]-200 ,buf.extent[2],0);
	outputCPU.stride[2]=input.stride[2];
	outputGPU.stride[2]=input.stride[2];
	local_laplacian_cpu(levels,alpha,beta,input,outputCPU);
	local_laplacian_gpu(levels,alpha,beta,input,outputGPU);
	return output;
}
