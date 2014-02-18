import matplotlib.pyplot as plt
import matplotlib as pltlib
import time

import numpy as np

from cpu_fractal import fractal_2 as cpu_f
from cuda_fractal.pycuda import fractal as cuda_f
import render

def callCPU(position, zoom, dimensions, name, iterations=100,scale=1,save=True):
	start = time.time()
	result = cpu_f.gen(position,zoom,dimensions,iterations=iterations,silent=True,scale=scale)
	elapsed = time.time()-start
	if save:
		render.SaveToPngThread(result,"cpu_"+name,render.colors['default'],silent=True)
	return elapsed


def callCUDA(position, zoom, dimensions, name, iterations=100,scale=1,save=True,block=(5,5,1),thread=(1,1,1),mode=0):
	result,time = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,mode=mode,scale=scale,block=block,thread=thread)
	if save:
		render.SaveToPngThread(result,"cuda_mode"+str(mode)+"-"+name,render.colors['default'],silent=False)
	return time

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"

def cudaCollect(position,zoom,dimensions,blockData,threadData,mode=0):
	#First run, block checking only
	times = {}

	#[0] = start,
	#[1] = end,
	#[2] = stride
	for x in blockData[0]:
		for y in blockData[1]:

			block = (x,y,1)

			for t_x in threadData[0]:
				for t_y in threadData[1]:
					
					thread = (t_x,t_y,1)
					time = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=True,mode=mode)
					times[(x,y,t_x,t_y)] = time
					print "\t"+str(block)+", "+str(thread)+": "+str(time)
	return times
	
def makePlot(dimensions,zoom,position,mode,directory,bdata,tdata):
	recData = cudaCollect(position,zoom,dimensions,bData,tData,mode=mode)
	
	cores = [xy[0]*xy[1] for xy in recData.keys()]
	times = recData.values()
	threads = [float(xy[2]*xy[3]) for xy in recData.keys()]

	threads_max = max(threads)
	threads_min = min(threads)
###############################################
	threads = np.array(threads,dtype=np.float)
	cores = np.array(cores,dtype=np.float)
	times = np.array(times,dtype=np.float)
###############################################
	threads = threads / threads.max()
	threads = np.log10(threads)
###############################################
	times = times / times.max()
	times = np.log10(times)
###############################################
	print threads_max

	#threads = [(x,x,x) for x in threads]
	print threads
	print cores
	print times

	#plt.plot(x_coords,y_coords,'ro')

	plt.scatter(cores,times,c=threads,marker="+")

	plt.ylabel("Time to compute \log_{10}(seconds)")
	plt.xlabel("Number of CUDA cores")

	title_identifier = {[0]:'write',[1]:'read and write',[2]:'raw compute'}[mode]

	plt.title("Fractal generation ["+title_identifier+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom))
	
	plt.savefig(directory+"mode_"+str(mode)+"--"+str(time.time())+".png")

def runComparison():

	bData = {
			0: #x
				[1,2,3,4,5,,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,150,200,250,300,350,400,450,500,600,700,800,900,1000],
			1: #y
				[1]
	}

	tData = {
			0: #x
				[1,2,3,4,5,6,7,8,16,20,25,40,32,64,128,256,512,1024],
				range(1,100),
			1: #y
				[1]
	}

	makePlot([2000,2000],900,[0,0],0,"~/Parallel/ParallelProjects/graphs/fractal/",bData,tData)
	makePlot([2000,2000],900,[0,0],1,"~/Parallel/ParallelProjects/graphs/fractal/",bData,tData)
	makePlot([2000,2000],900,[0,0],2,"~/Parallel/ParallelProjects/graphs/fractal/",bData,tData)

runComparison()