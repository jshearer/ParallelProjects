import matplotlib.pyplot as plt
import time

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
	start = time.time()
	result = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,scale=scale,block=block,thread=thread)
	elapsed = time.time()-start
	if save:
		render.SaveToPngThread(result,"cuda_"+name,render.colors['default'],silent=True)
	return elapsed

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"

def cudaCollect(position,zoom,dimensions,blockData,threadData):
	#First run, block checking only
	blockTimes = {}
	threadTimes = {}
	#blockTimes[block] = time

	#[0] = start,
	#[1] = end,
	#[2] = stride
	for x in range(blockData[0][0],blockData[0][1],blockData[0][2]):
		for y in range(blockData[1][0],blockData[1][1],blockData[1][2]):

			block = (x,y,1)
			blockTimes[block] = callCUDA(position,zoom,dimensions,str(block),block=block,save=True)
			print str(block)+": "+str(blockTimes[block])
			threadTimes[block] = {}
			for t_x in range(threadData[0][0],threadData[0][1],threadData[0][2]):
				for t_y in range(threadData[1][0],threadData[1][1],threadData[1][2]):
					
					thread = (t_x,t_y,1)
					threadTimes[block][thread] = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=True)
					print "\t"+str(block)+", "+str(thread)+": "+str(threadTimes[block][thread])
	return [blockTimes,threadTimes]

#Test:

bData = {
		0: #x
			{
				0: 1,
				1: 20,
				2: 1
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
				1: 2,
				2: 1
			},
		1: #y
			{
				0: 1,
				1: 2,
				2: 1
			}
}

#cudaCollect([0,0],450/2.0,[400,400],bData,tData)

def makePlot():
	dat = cudaCollect([0,0],450/2.0,[400,400],bData,tData)
	blockTimes = dat[0]
	threadTimes = dat[1]

	print "Lol hi! "+str(threadTimes)

	x_coords = [x[0]*x[1] for x in threadTimes[threadTimes.keys()[0]].keys()]
	y_coords = [threadTimes[threadTimes.keys()[0]][y] for y in threadTimes[threadTimes.keys()[0]]]

	for loop_x in xrange(1,len(threadTimes.keys())):
		x_coords.extend([x[0]*x[1]*loop_x for x in threadTimes[threadTimes.keys()[loop_x]].keys()])

	for loop_y in xrange(1,len(threadTimes.keys())):
		y_coords.extend([threadTimes[threadTimes.keys()[loop_y]][y]*loop_y for y in threadTimes[threadTimes.keys()[loop_y]]])

	print x_coords
	print y_coords

	plt.plot(x_coords,y_coords,'ro')
	#plt.axis([0,len(x_coords),0,len(y_coords)])
	plt.show()

makePlot()