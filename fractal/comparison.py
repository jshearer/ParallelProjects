import time

from cpu_fractal import fractal_2 as cpu_f
from cuda_fractal.pycuda import fractal as cuda_f
import render

def callCPU(position, zoom, dimensions, name, iterations=100):
	start = time.time()
	result = cpu_f.gen(position,zoom,dimensions,iterations=iterations,silent=True)
	elapsed = time.time()-start
	render.SaveToPng(result,"cpu_"+name,render.colors['default'],silent=True)
	return elapsed


def callCUDA(position, zoom, dimensions, name, iterations=100):
	start = time.time()
	result = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True)
	elapsed = time.time()-start
	render.SaveToPng(result,"cuda_"+name,render.colors['default'],silent=True)
	return elapsed

def compareParams(position, zoom, dimensions, name, iterations=100):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = callCPU(position,zoom,dimensions,name,iterations)
	cudaTime = callCUDA(position,zoom,dimensions,name,iterations)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"
