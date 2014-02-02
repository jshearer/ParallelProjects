import time

from cpu_fractal import fractal_2 as cpu_f
from cuda_fractal.pycuda import fractal as cuda_f
import render

def callCPU(position, zoom, dimensions, name, iterations=100,scale=1,save=True):
	start = time.time()
	result = cpu_f.gen(position,zoom,dimensions,iterations=iterations,silent=True,scale=scale)
	elapsed = time.time()-start
	if save:
		render.SaveToPng(result,"cpu_"+name,render.colors['default'],silent=True)
	return elapsed


def callCUDA(position, zoom, dimensions, name, iterations=100,scale=1,save=True):
	start = time.time()
	result = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,scale=scale)
	elapsed = time.time()-start
	if save:
		render.SaveToPng(result,"cuda_"+name,render.colors['default'],silent=True)
	return elapsed

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"
