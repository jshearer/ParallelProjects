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
	result,time = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,scale=scale,block=block,thread=thread)
	if save:
		render.SaveToPngThread(result,"cuda_"+name,render.colors['default'],silent=True)
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
	for x in range(blockData[0][0],blockData[0][1],blockData[0][2]):
		for y in range(blockData[1][0],blockData[1][1],blockData[1][2]):

			block = (x,y,1)

			for t_x in range(threadData[0][0],threadData[0][1],threadData[0][2]):
				for t_y in range(threadData[1][0],threadData[1][1],threadData[1][2]):
					
					thread = (t_x,t_y,1)
					time = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=True)
					times[(x,y,t_x,t_y)] = (time,x,y,t_x,t_y)
					print "\t"+str(block)+", "+str(thread)+": "+str(time)
	return times

#Test:

bData = {
		0: #x
			{
				0: 1,
				1: 90,
				2: 2
			},
		1: #y
			{
				0: 1,
				1: 2,
				2: 1
			}
}

tData = {
		0: #x
			{
				0: 1,
				1: 30,
				2: 5
			},
		1: #y
			{
				0: 1,
				1: 30,
				2: 5
			}
}

#cudaCollect([0,0],450/2.0,[400,400],bData,tData)

def makePlot():
	times = cudaCollect([0,0],450	,[1500,1500],bData,tData)
	
	x_coords = [xy[0]*xy[1] for xy in times.keys()]
	y_coords = [time[0] for time in times.values()]
	colors = [float(xy[2]*xy[3]) for xy in times.keys()]

	colors_max = max(colors)
	colors_min = min(colors)

	colors = np.array(colors,dtype=np.float)

	colors = colors - colors.min()
	colors = colors / colors.max()

	colors = np.log10(colors)

	print colors_max

	#colors = [(x,x,x) for x in colors]

	print colors
	
	#print x_coords
	#print y_coords

	#plt.plot(x_coords,y_coords,'ro')

	plt.scatter(x_coords,y_coords,c=colors,marker="+")

	plt.ylabel("Time to compute (seconds)")
	plt.xlabel("Number of CUDA cores")
	plt.title("Fractal generation.")
		#("+str(bData[0][0])+","+str(bData[0][1])+")-("+str(bData[1][0])+","+str(bData[1][1])+") cores, stride ("+str(bData[0][2])+","+str(bData[1][2])+")\n"
		#"("+str(tData[0][0])+","+str(tData[0][1])+")-("+str(tData[1][0])+","+str(tData[1][1])+") threads, stride ("+str(tData[0][2])+","+str(tData[1][2])+")")
	#plt.axis([0,len(x_coords),0,len(y_coords)])
	plt.show()

makePlot()