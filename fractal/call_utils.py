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
	result,time,err1,err2 = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,debug=False,action=mode,scale=scale,block=block,thread=thread)
	if save:
		render.SaveToPngThread(result,"cuda_mode"+str(mode)+"-"+name,render.colors['default'],silent=False)
	return (result,time,err1,err2)