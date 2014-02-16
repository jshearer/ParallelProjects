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


def callCUDA(position, zoom, dimensions, name, iterations=100,scale=1,save=True,block=(5,5,1),thread=(1,1,1)):
	# http://docs.nvidia.com/cuda/parallel-thread-execution/index.html
	result,time = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,scale=scale,block=block,thread=thread)
	if save:
		render.SaveToPngThread(result,"cuda_"+name,render.colors['default'],silent=False)
	return time

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"

def cudaCollect(position,zoom,dimensions,blockData,threadData):
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
					time = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=True)
					times[(x,y,t_x,t_y)] = (time,x,y,t_x,t_y)
					print "\t"+str(block)+", "+str(thread)+": "+str(time)
	return times

#Test:

bData = {
		0: #x
			range(1,160,1),
		1: #y
			range(1,160,1)
}

tData = {
		0: #x
			#[1,2,4,8,16,32,64,128,256,512,1024],
			range(1,100),
		1: #y
			range(1,2,1)
}

#cudaCollect([0,0],450/2.0,[400,400],bData,tData)
	
def makePlot():
	dimensions = [2000,2000]
	zoom = 450*2
	recData = cudaCollect([0,0],zoom,dimensions,bData,tData)
	
	cores = [xy[0]*xy[1] for xy in recData.keys()]
	times = [time[0] for time in recData.values()]
	colors = [float(xy[2]*xy[3]) for xy in recData.keys()]

	colors_max = max(colors)
	colors_min = min(colors)
###############################################
	colors = np.array(colors,dtype=np.float)
	cores = np.array(cores,dtype=np.float)
	times = np.array(times,dtype=np.float)
###############################################
	colors = colors / colors.max()
	colors = np.log10(colors)
###############################################
	times = times / times.max()
	times = np.log10(times)
###############################################
	print colors_max

	#colors = [(x,x,x) for x in colors]
	print colors
	print cores
	print times

	#plt.plot(x_coords,y_coords,'ro')

	plt.scatter(cores,times,c=colors,marker="+")

	plt.ylabel("Time to compute (seconds,log10)")
	plt.xlabel("Number of CUDA cores")
	plt.title("Fractal generation write.\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom))
		#("+str(bData[0][0])+","+str(bData[0][1])+")-("+str(bData[1][0])+","+str(bData[1][1])+") cores, stride ("+str(bData[0][2])+","+str(bData[1][2])+")\n"
		#"("+str(tData[0][0])+","+str(tData[0][1])+")-("+str(tData[1][0])+","+str(tData[1][1])+") threads, stride ("+str(tData[0][2])+","+str(tData[1][2])+")")
	#plt.axis([0,len(x_coords),0,len(y_coords)])
	plt.show()

makePlot()
