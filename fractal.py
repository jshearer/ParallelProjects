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

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

genChunk = SourceModule("""
#include <cuComplex.h>
__global__ void gen(int size[2],int position[2],int realBlockDim[2],float *zoom,int *iterations,int *result)
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
//result width = 
""").get_function("gen")
print("Compiled and got function gen")
#Variable setup section
block = (10,10,1) #number of GPU cores launched is block[0]*block[1]
size = numpy.array([600,600],dtype=numpy.int32) #This is the chunk size each core is to calculate, in parallel
res_shape = (size[0]*block[0],size[1]*block[1]) #Shape for the output array, that is passed into the kernel
result = numpy.zeros(res_shape,dtype=numpy.int32) #Actual output array that is passed into the kernel, and then read once generation is complete
zoom = numpy.float32(4000) #higher number = more zoomed in. Supposed to zoom into the center, scaling around position.
iterations = numpy.int32(100) #max number of iterations. Make higher for more pretty

position = numpy.array([(block[0]*size[0])/2,(block[1]*size[1])/2],dtype=numpy.int32) #center fractal on middle of generated image.
position = numpy.array([-7000,0],dtype=numpy.int32) #change this to offset fractal

#this math makes it so when you increase zoom, it acts like you would expect. Otherwise it would zoom around the origin, which no one wants.
scale = 1.0 #if you do, just comment this out
zoom = numpy.float32(zoom/scale)
position = Vector(position[0]/scale,position[1]/scale)
position = position - (Vector(result.shape[0],result.shape[1])/2)
position = numpy.array([int(position.x),int(position.y)]).astype(numpy.int32)


print("Setup arguments, running now, size: "+str(size)+", zoom: "+str(zoom)+", iterations: "+str(iterations)+", position: "+str(position)+", result shape: ("+str(result.shape)+", "+str(res_shape)+"), result size: "+str((result.nbytes*8)/32))

genChunk(cuda.In(size),cuda.In(position),cuda.In(numpy.array([block[0],block[1]],dtype=numpy.int32)),cuda.In(zoom),cuda.In(iterations),cuda.InOut(result),block=(1,1,1),grid=block)
print("Executed.")
#call using cuda.In, cuda.Out etc
cuda.Context.synchronize()
print("Done running. synched.")
print(result)

result2 = numpy.zeros((result.shape[1],result.shape[0],3)).astype(numpy.uint8)
result = result.astype(numpy.float32)*(255.0/iterations)
for x in range(0,result2.shape[0]-1):
	for y in range(0,result2.shape[1]-1):
		result2[x,y,0] = result[y,x]
		result2[x,y,1] = result[y,x]
		result2[x,y,2] = result[y,x]
Image.fromarray(result2,"RGB").save("testfractal.png")
