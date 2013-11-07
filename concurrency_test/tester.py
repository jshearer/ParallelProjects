#!/usr/bin/python2
import numpy

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

increment = SourceModule("""
#include <cuComplex.h>
__global__ void increment(int* a,float* progress)
{	
	for(int i=0;i<500;i++){
		atomicAdd(progress,1.0f);
	}
}

""").get_function("increment")
print("Compiled and got function increment")

pagelocked_mem = cuda.pagelocked_zeros((1,1),numpy.float32, mem_flags=cuda.host_alloc_flags.DEVICEMAP)
pagelocked_mem_ptr = numpy.intp(pagelocked_mem.base.get_device_pointer())
print(pagelocked_mem[0,0])

a = numpy.int32(345)
a_gpu = cuda.mem_alloc(a.nbytes)
cuda.memcpy_htod(a_gpu,a)


increment(a_gpu,pagelocked_mem_ptr, block=(1,1,1), grid=(50,50,1))

while pagelocked_mem[0,0]<(50*50*500):
	print pagelocked_mem[0,0]

cuda.Context.synchronize()

print pagelocked_mem[0,0]

