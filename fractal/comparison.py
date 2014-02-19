import time

import numpy as np

from cpu_fractal import fractal_2 as cpu_f
from cuda_fractal.pycuda import fractal as cuda_f
import render

import fractal_data
import plot_data

def callCPU(position, zoom, dimensions, name, iterations=100,scale=1,save=True):
	start = time.time()
	result = cpu_f.gen(position,zoom,dimensions,iterations=iterations,silent=True,scale=scale)
	elapsed = time.time()-start
	if save:
		render.SaveToPngThread(result,"cpu_"+name,render.colors['default'],silent=True)
	return elapsed


def callCUDA(position, zoom, dimensions, name, iterations=100,scale=1,save=True,block=(5,5,1),thread=(1,1,1),mode=0):
	result,time = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,action=mode,scale=scale,block=block,thread=thread)
	if save:
		render.SaveToPngThread(result,"cuda_mode"+str(mode)+"-"+name,render.colors['default'],silent=False)
	return time

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"

def runComparison():

	bData = {
			0: #x
				#[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000],
				[50,100,200],
			1: #y
				[1]
	}

	tData = {
			0: #x
				[1,2,3,4,5,6,7,8,16,20,25,40,32,64,128],#,256,512,1024],
				#range(1,100),
			1: #y
				[1]
	}

	index = fractal_data.cudaCollect([0,0],900,[2000,2000],bData,tData,mode=0)
	cores,times,threads = fractal_data.extractCols(index)
	plot_data.makePlot(cores,times,threads,0,"/home/jshearer/ParallelProjects/graphs/fractal/")

runComparison()
