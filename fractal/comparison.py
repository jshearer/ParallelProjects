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

	execData = {'blocks':range(1,10)+range(10,100,10)+range(100,500,50)+[512],
		    'threads':range(1,10)+range(10,100,10)+range(100,500,50)+[512]}

	print "Mode 0:"
	fractal_data.cudaCollect([0,0],250,[512,512],execData,mode=0)
	print "Mode 1:"
	fractal_data.cudaCollect([0,0],250,[512,512],execData,mode=1)
	print "Mode 2:"
	fractal_data.cudaCollect([0,0],250,[512,512],execData,mode=2)
	print "Mode 3:"
	fractal_data.cudaCollect([0,0],250,[512,512],execData,mode=3)

	print "Overlap (mode 4):"
	fractal_data.cudaCollect([0,0],250,[512,512],execData,mode=4)

#	print "Inserted into index: "+str(index)
#	data = fractal_data.extractCols(index)
#	print "len cores,times,threads ("+str(len(cores))+", "+str(len(times))+", "+str(len(threads))+")."
#	plot_data.makePlot(data,"/home/jshearer/ParallelProjects/graphs/fractal/")

runComparison()
