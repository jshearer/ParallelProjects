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

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

genChunk = SourceModule("""
#include <cuComplex.h>
__global__ void gen(int size[2],float position[2],int realBlockDim[2],float *zoom,int *iterations,int *result, float* progress)
{	
	int startx = blockIdx.x*size[0];
	int starty = blockIdx.y*size[1];
	float t_x, t_y;
	int i, x, y;
	cuFloatComplex z, z_unchanging;
	float z_real, z_imag;
	for(x = startx; x <= size[0]+startx; x++){
		for(y = starty; y <= size[1]+starty; y++){
			t_x = (x+position[0])/(*zoom);
			t_y = (y+position[1])/(*zoom);
			z = make_cuFloatComplex(t_x,t_y);
			z_unchanging = make_cuFloatComplex(t_x,t_y); //optomize this with pointer magic?
			for(i = 0; i<(*iterations) + 1; i++){
				z = cuCmulf(z,z);
				z = cuCaddf(z,z_unchanging);
				z_real = cuCrealf(z);
				z_imag = cuCimagf(z);
				if((z_real*z_real + z_imag*z_imag)>4){
					result[x+(y*size[0]*realBlockDim[0])] = i;
					break;
				}
			}
		}
	}
	atomicAdd(progress,1);
}

""").get_function("gen")
print("Compiled and got function gen")

def GenerateFractal(dimensions,position,zoom,iterations,block=(20,20,1)):
	chunkSize = numpy.array([dimensions[0]/block[0],dimensions[1]/block[1]],dtype=numpy.int32)
	zoom = numpy.float32(zoom)
	iterations = numpy.int32(iterations)
	blockDim = numpy.array([block[0],block[1]],dtype=numpy.int32)
	result = numpy.zeros(dimensions,dtype=numpy.int32)

	#Center position
	position = Vector(position[0]*zoom,position[1]*zoom)
	position = position - (Vector(result.shape[0],result.shape[1])/2)
	position = numpy.array([int(position.x),int(position.y)]).astype(numpy.float32)

	print("Calling CUDA function. Starting timer.")
	start_time = time.time()

	genChunk(cuda.In(chunkSize), cuda.In(position), cuda.In(blockDim), cuda.In(zoom), cuda.In(iterations), cuda.InOut(result), block=(1,1,1), grid=block)
	end_time = time.time()
	elapsed_time = end_time-start_time
	print("Done with call. Took "+str(elapsed_time)+" seconds. Here's the repr'd arary:\n")
	result[result.shape[0]/2,result.shape[1]/2]=iterations+1
	print(result)
	return result

def SaveToPng(result,name):
	print("Resizing result to be in range 0-255")
	result = (result.astype(numpy.float32)*(255.0/result.max())).astype(numpy.uint8)
	print("Done resizing. Now generating image array.")

	result = result.reshape((result.shape[1],result.shape[0]))
	print("Done generating image array. Writing image file.")

	Image.fromstring("L",(result.shape[1],result.shape[0]),result.tostring()).save(name+".png")
	print("Image file written.")
