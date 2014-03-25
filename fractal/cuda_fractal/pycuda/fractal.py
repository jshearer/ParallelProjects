#!/usr/bin/python2
import numpy
import math
import locale

locale.setlocale(locale.LC_ALL, 'en_US.UTF_8')

import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# for debugging
import pdb

# local imports
from Vector import Vector
import utils

def In(thing):
	thing_pointer = cuda.mem_alloc(thing.nbytes)
	cuda.memcpy_htod(thing_pointer, thing)
	return thing_pointer

_genChunk=None
def compileCuda(srcN='cuda_fractal/pycuda/genchunk.cu'):
        global _genChunk
        _fp = open(srcN, 'r')
        _genSrc = _fp.read()
        _fp.close()
        
        _genChunk = SourceModule(_genSrc).get_function("gen")
        print("Compiled and got function gen")

def GenerateFractal(dimensions,position,zoom,iterations,scale=1,action=0,block=(15,15,1),thread=(1,1,1),
                    report=False, silent=False, debug=True,px_per_block=False,px_per_thread=False):
        global _genChunk

        if not _genChunk: compileCuda()

	#Force progress checking to False, otherwise it'll go on forever with reporting turned off in the kernel
	report = False
	zoom = zoom * scale
        #                             
	dimensions = numpy.array([dimensions[0]*scale,dimensions[1]*scale],dtype=numpy.int32)

	#For block, grid calculation:
	
	if (type(block) == type(1)) and (type(thread) == type(1)):
		#(block, thread, px_per_block, px_per_thread)
		params = utils.genParameters(block,thread,dimensions,silent=silent)
		if not px_per_block:
			px_per_block = numpy.array(params[2],dtype=numpy.int32)
		if not px_per_thread:
			px_per_thread = numpy.array(params[3],dtype=numpy.int32)
		block = numpy.append(params[0],1)
		thread = numpy.append(params[1],1)

		block = tuple([numpy.asscalar(block[i]) for i in range(len(block))])
		thread = tuple([numpy.asscalar(thread[i]) for i in range(len(thread))])

	else:
		if not px_per_block:
			px_per_block = (numpy.array(dimensions) / numpy.array(block))
		if not px_per_thread:
			px_per_thread = (px_per_block / numpy.array(thread))
	
	px_per_block = numpy.asarray(px_per_block).astype(numpy.int32)
	px_per_thread = numpy.asarray(px_per_thread).astype(numpy.int32)

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

	#pdb.set_trace()

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
	
	_genChunk(blockpx, threadpx, dims, posit, zoo, iters, res, ppc_ptr, act, block=thread, grid=block)
	
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
