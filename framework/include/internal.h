#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdint.h>

typedef struct buffer_t
{
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

#ifndef TOOL_H
#define TOOL_H
namespace Fusion
{
    namespace Internal{
        buffer_t* divBuffer(buffer_t *buf,int start,int nend)
        {
            buffer_t *buff= new buffer_t();
            buff->extent[0] = buf->extent[0];
            buff->extent[1] = nend-start;
            if(buff->extent[2])
                buff->extent[2] = buf->extent[2];
            if(buff->extent[3])
                buff->extent[3] = buf->extent[3];
            buff->stride[0] = buf->stride[0];
            buff->stride[1] = buf->stride[1];
            if(buff->extent[2])
                buff->stride[2] = buf->stride[2];
            if(buff->extent[3])
                buff->stride[3] = buf->stride[3];
            buff->elem_size = buf->elem_size;
            buff->host_dirty = true;
            buff->dev_dirty = false;
            buff->dev = 0;
            if(start==0)
            {
                buff->host = buf->host;
                return buff;
            }
            int offset=buf->extent[0]*start*buf->elem_size*1;
            buff->min[1]=start;
            buff->host= buf->host+offset;
            return buff;
        }


        uint8_t* initBuffer_t(int x,int y,int z,int w,buffer_t* buf,int bitOfInput)
        {
            buf->extent[0] = x;
            buf->extent[1] = y;
            buf->extent[2] = z;
            buf->extent[3] = w;
            buf->stride[0] = 1;
            buf->stride[1] = x;
            buf->stride[2] = x*y;
            buf->stride[3] = x*y*z;
            buf->elem_size = bitOfInput;
            size_t size = 1;
            if (x) size *= x;
            if (y) size *= y;
            if (z) size *= z;
            if (w) size *= w;
            uint8_t* ptr = new uint8_t[bitOfInput*size + 40];
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
    }
}
#endif // TOOL_H
