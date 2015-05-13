#include "tool.h"
//Return A split buffer_t pointer which host is base on argument buffer_t plus argument start
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
