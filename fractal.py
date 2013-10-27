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

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

genChunk = SourceModule("""
#include <cuComplex.h>
__global__ void gen(int size[2],float position[2],int realBlockDim[2],float *zoom,int *iterations,int *result)
{	
	int startx = blockIdx.x*size[0];
	int starty = blockIdx.y*size[1];
	for(int x = startx; x <= size[0]+startx; x++){
		for(int y = starty; y <= size[1]+starty; y++){
			float t_x = (x+position[0])/(*zoom);
			float t_y = (y+position[1])/(*zoom);
			int i;
			cuFloatComplex z = make_cuFloatComplex(t_x,t_y);
			cuFloatComplex z_unchanging = make_cuFloatComplex(t_x,t_y); //optomize this with pointer magic?
			for(i = 0; i<(*iterations) + 1; i++){
				z = cuCmulf(z,z);
				z = cuCaddf(z,z_unchanging);
				float z_real = cuCrealf(z);
				float z_imag = cuCimagf(z);
				if((z_real*z_real + z_imag*z_imag)>4){
					result[x+y*(size[1]*(realBlockDim[1]))] = i;
					break;
				}
			}
		}
	}
}

""").get_function("gen")
print("Compiled and got function gen")
#Variable setup section
block = (40,40,1) #number of GPU cores launched is block[0]*block[1]
size = numpy.array([int(math.ceil(350.0/float(block[0]))),int(math.ceil(350.0/float(block[1])))],dtype=numpy.int32) #This is the chunk size each core is to calculate, in parallel
zoom = numpy.float32(270) #higher number = more zoomed in. Supposed to zoom into the center, scaling around position.
iterations = numpy.int32(600) #max number of iterations. Make higher for more pretty

position = numpy.array([-1,0],dtype=numpy.float32) #change this to offset fractal

#this math makes it so when you increase zoom, it acts like you would expect. Otherwise it would zoom around the origin, which no one wants.
scale = 10.0 #if you do, just comment this out

size = (size.astype(numpy.float32)*float(scale)).astype(numpy.int32)
res_shape = (size[0]*block[0],size[1]*block[1]) #Shape for the output array, that is passed into the kernel
result = numpy.zeros(res_shape,dtype=numpy.int32) #Actual output array that is passed into the kernel, and then read once generation is complete

zoom = numpy.float32(zoom*scale)
position = Vector(position[0]*zoom,position[1]*zoom)
position = position - (Vector(result.shape[0],result.shape[1])/2)
position = numpy.array([int(position.x),int(position.y)]).astype(numpy.float32)

print("Setup arguments, running now, size: "+str(size)+", zoom: "+str(zoom)+", iterations: "+str(iterations)+", position: "+str(position)+", result shape: ("+str(result.shape)+", "+str(res_shape)+"), result size: "+str((result.nbytes*8)/32))

genChunk(cuda.In(size),cuda.In(position),cuda.In(numpy.array([block[0],block[1]],dtype=numpy.int32)),cuda.In(zoom),cuda.In(iterations),cuda.InOut(result),block=(1,1,1),grid=block)
print("Executed.")
#call using cuda.In, cuda.Out etc
cuda.Context.synchronize()
print("Done running. synched.")
print(result)

print("Resizing result to be in range 0-255")
result = result.astype(numpy.float32)*(255.0/iterations)
print("Done resizing. Now generating image array.")

result2 = numpy.hstack((result,result,result))
result2.resize((result.shape[0],result.shape[1],3))
result2 = result2.astype(numpy.uint8)

Image.fromarray(result2,"RGB").save("testfractal.png")
