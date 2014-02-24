#!/usr/bin/python2
import os
from multiprocessing import Pool
import utils
from PIL import Image
import numpy
import argparse
from Vector import Vector
import sys
import time
import traceback
import math
import time
import locale

locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

genChunk = SourceModule("""
#include <cuComplex.h>

__device__ int map2Dto1D(int x, int y,int x_width)
{
	return x+(y*x_width);	
}

__global__ void gen(int px_per_block[2],int px_per_thread[2],int size[2],float position[2],float *zoom,int *iterations,int *result, int* progress,int action)
{ 	//blockDim = size of threads per block
	//gridDim = size of blocks

	//int size[2] argument is just to make sure we don't fall off the edge and crash the entire machine...

	//actions: 0 = write
	//	   1 = read+write
	//     2 = none	
	//	   3 = atomicAddTest
	//	   4 = overlapMap

	int startx = (blockIdx.x*px_per_block[0])+(threadIdx.x*px_per_thread[0]);
	int starty = (blockIdx.y*px_per_block[1])+(threadIdx.y*px_per_thread[1]);

	float t_x, t_y;
	int i, x, y;

	cuFloatComplex z = cuFloatComplex();
	cuFloatComplex z_unchanging = cuFloatComplex();

	float z_real, z_imag;

	for(x = startx; x < startx+px_per_thread[0]; x++){
		for(y = starty; y < starty+px_per_thread[1]; y++){
			if(action==3)
			{
				atomicAdd(progress,1);
			}
			t_x = (x+position[0])/(*zoom);
			t_y = (y+position[1])/(*zoom);

			z.x = t_x;
			z.y = t_y;
			z_unchanging.x = t_x;
			z_unchanging.y = t_y; //optomize this with pointer magic?
			if(action==4) //generate overlap map
			{
				result[map2Dto1D(x,y,size[0])] = result[map2Dto1D(x,y,size[0])] + 1;
			} else
			{
				for(i = 0; i<(*iterations) + 1; i++){
					z = cuCmulf(z,z);
					z = cuCaddf(z,z_unchanging); //z = z^2 + z_orig
					z_real = cuCrealf(z);
					z_imag = cuCimagf(z);
					if((z_real*z_real + z_imag*z_imag)>4){
						if(action==0)//act cool, do the default
						{
							result[map2Dto1D(x,y,size[0])] = i;
						} else if(action==1)// read+write test
						{
							result[map2Dto1D(x,y,size[0])] = result[map2Dto1D(x,y,size[0])] + 1;
						}//else if action==2, do nothing
						break;
					}
				}
			}
		}
	}
}
""").get_function("gen")
print("Compiled and got function gen")

def In(thing):
	thing_pointer = cuda.mem_alloc(thing.nbytes)
	cuda.memcpy_htod(thing_pointer, thing)
	return thing_pointer

def GenerateFractal(dimensions,position,zoom,iterations,scale=1,action=0,block=(15,15,1),thread=(1,1,1), report=False, silent=False, debug=True,px_per_core=False,px_per_thread=False):
	#Force progress checking to False, otherwise it'll go on forever with reporting turned off in the kernel
	report = False
	zoom = zoom * scale
	dimensions = numpy.array([dimensions[0]*scale,dimensions[1]*scale],dtype=numpy.int32)
	position = [position[0]*scale*zoom,position[1]*scale*zoom]

	zoom = numpy.float32(zoom)
	iterations = numpy.int32(iterations)
	result = numpy.zeros(dimensions,dtype=numpy.int32)

	#Center position
	#position = Vector(position[0]*zoom,position[1]*zoom)
	position = Vector(position[0],position[1])
	position = position - (Vector(dimensions[0],dimensions[1])/2)
	position = numpy.array([int(position.x),int(position.y)]).astype(numpy.float32)

	#For progress reporting:
	ppc = cuda.pagelocked_zeros((1,1),numpy.int32, mem_flags=cuda.host_alloc_flags.DEVICEMAP) #pagelocked progress counter
	ppc[0,0] = 0
	ppc_ptr = numpy.intp(ppc.base.get_device_pointer()) #pagelocked memory counter, device pointer to
	#End progress reporting

	#For block, grid calculation:
	if (type(block) == type(1)) and (type(thread) == type(1)):
		import utils
		#(block, thread, px_per_block, px_per_thread)
		params = utils.calcParameters(block,thread,dimensions,silent=silent)
		px_per_block = px_per_block or numpy.array(params[2],dtype=numpy.int32)
		px_per_thread = px_per_thread or numpy.array(params[3],dtype=numpy.int32)
		block = numpy.append(params[0],1)
		thread = numpy.append(params[1],1)

		block = tuple([numpy.asscalar(block[i]) for i in range(len(block))])
		thread = tuple([numpy.asscalar(thread[i]) for i in range(len(thread))])

	else:
		px_per_block = px_per_block or numpy.array(dimensions) / numpy.array(block)
		px_per_thread = px_per_thread or px_per_block / numpy.array(thread)

	if debug:
		print "Position: "+str(position)
		print "Thread Dimensions: "+str(thread)
		print "Block Dimensions: "+str(block)
		print "Pixels per block: "+str(px_per_block)
		print "Pixels per thread: "+str(px_per_thread)

	#Copy parameters over to device
	posit = In(position)
	zoo = In(zoom)
	iters = In(iterations)
	res = In(result)
	act = numpy.int32(action)
	blockpx = In(px_per_block)
	threadpx = In(px_per_thread)
	dims = In(dimensions)

	if not silent:
		print("Calling CUDA function. Starting timer. progress starting at: "+str(ppc[0,0]))
	
	start = cuda.Event()
	end = cuda.Event()
	start.record()
	
	genChunk(blockpx, threadpx, dims, posit, zoo, iters, res, ppc_ptr, act, block=thread, grid=block)
	
	end.record()
	cuda.Context.synchronize()
	
	millis = start.time_till(end)
	seconds = millis / 1000.0

	if not silent:
		print("Done with call. Took "+str(seconds)+" seconds. Here's the repr'd arary:\n")

	#Copy result back from device
	cuda.memcpy_dtoh(result, res)

	if not silent: 
		print(result)
		
	if action!=4:  #not in overlap mode
		result[result.shape[0]/2,result.shape[1]/2]=iterations+1 #mark center of image
	return result,seconds,block,thread
