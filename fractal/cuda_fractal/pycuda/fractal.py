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
__global__ void gen(int size[2],float position[2],float *zoom,int *iterations,int *result, int* progress,int action)
{ 	//blockDim = size of threads per block
	//gridDim = size of blocks

	//actions: 0 = write
	//	   1 = read+write
	//     2 = none	
	//	   3 = atomicAddTest
	//	   4 = overlapMap

	int startx = (blockIdx.x*size[0])+(((float)threadIdx.x/blockDim.x)*size[0]);
	int starty = (blockIdx.y*size[1])+(((float)threadIdx.y/blockDim.y)*size[1]);

	float t_x, t_y;
	int i, x, y;

	cuFloatComplex z = cuFloatComplex();
	cuFloatComplex z_unchanging = cuFloatComplex();

	float z_real, z_imag;

	for(x = startx; x < (blockIdx.x*size[0])+((((float)(threadIdx.x+1))/blockDim.x)*size[0]); x++){
		for(y = starty; y < (blockIdx.y*size[1])+((((float)(threadIdx.y+1))/blockDim.y)*size[1]); y++){
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
				result[x+(y*size[0]*gridDim.x)] = result[x+(y*size[0]*gridDim.x)] + 1;
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
							result[x+(y*size[0]*gridDim.x)] = i;
						} else if(action==1)// read+write test
						{
							result[x+(y*size[0]*gridDim.x)] = result[x+(y*size[0]*gridDim.x)] + 1;
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

def GenerateFractal(dimensions,position,zoom,iterations,scale=1,action=0,block=(15,15,1),thread=(1,1,1), report=False, silent=False, debug=True):
	#Force progress checking to False, otherwise it'll go on forever with reporting turned off in the kernel
	report = False
	zoom = zoom * scale
	dimensions = [dimensions[0]*scale,dimensions[1]*scale]
	position = [position[0]*scale*zoom,position[1]*scale*zoom]

	chunkSize = numpy.array([dimensions[0]/block[0],dimensions[1]/block[1]],dtype=numpy.int32)
	zoom = numpy.float32(zoom)
	iterations = numpy.int32(iterations)
	blockDim = numpy.array([block[0],block[1]],dtype=numpy.int32)
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

	if debug:
		print "Chunk Size: "+str(chunkSize)
		print "Position: "+str(position)
		print "Block Dimensions: "+str(blockDim)
		print "Thread Dimensions: "+str(thread)
		print "Thread Size: "+str([chunkSize[0]/thread[0],chunkSize[1]/thread[1]])

	#Copy parameters over to device
	chunkS = In(chunkSize)
	posit = In(position)
	zoo = In(zoom)
	iters = In(iterations)
	res = In(result)
	act = numpy.int32(action)

	if not silent:
		print("Calling CUDA function. Starting timer. progress starting at: "+str(ppc[0,0]))
	
	start = cuda.Event()
	end = cuda.Event()
	start.record()
	
	genChunk(chunkS, posit, zoo, iters, res, ppc_ptr, act, block=thread, grid=block)
	
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
	return result,seconds
