import call_utils
import fractal_data
import plot_data

def compareParams(position, zoom, dimensions, name, iterations=100,save=True):
	print "Comparing paramaters for ("+str(name)+"): ( ("+str(position[0])+", "+str(position[1])+"), "+str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"

	cpuTime = call_utils.callCPU(position,zoom,dimensions,name,iterations,save=save)
	cudaTime = call_utils.callCUDA(position,zoom,dimensions,name,iterations,save=save)

	print "CPU ran in "+str(cpuTime)+"s"
	print "CUDA ran in "+str(cudaTime)+"s"

def runComparison():

	execData = {'blocks':range(1,2048),
		    'threads':range(1,1024)}
	
	position = [-1.3,0]
	dimensions = [2000,1000]
	zoom = 900

	for mode in range(0,5):
		print "Mode "+str(mode)+":"
		fractal_data.cudaCollect(position,zoom,dimensions,execData,mode=mode)

#	print "Inserted into index: "+str(index)
#	data = fractal_data.extractCols(index)
#	print "len cores,times,threads ("+str(len(cores))+", "+str(len(times))+", "+str(len(threads))+")."
#	plot_data.makePlot(data,"/home/jshearer/ParallelProjects/graphs/fractal/")

runComparison()
