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

	bData = {
			0: #x
				[1,2,3,4,5,6,7,8,9,10,15,20,25,30,35,40,45,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,300,350,400,450,500,600,700,800,900,1000,1100,1200,1300,1400,1500,1600,1700,1800,1900,2000,2100],
				#[50,100,200],
			1: #y
				[1]
	}

	tData = {
			0: #x
				[1,2,3,4,5,6,7,8,16,20,25,40,32,64,128,256,512,1024],
				#range(1,100),
			1: #y
				[1]
	}

	print "Mode 0:"
	fractal_data.cudaCollect([0,0],900,[2000,2000],bData,tData,mode=0)
	print "Mode 1:"
	fractal_data.cudaCollect([0,0],900,[2000,2000],bData,tData,mode=1)
	print "Mode 2:"
	fractal_data.cudaCollect([0,0],900,[2000,2000],bData,tData,mode=2)
	print "Mode 3:"
	fractal_data.cudaCollect([0,0],900,[2000,2000],bData,tData,mode=3)

#	print "Inserted into index: "+str(index)
#	data = fractal_data.extractCols(index)
#	print "len cores,times,threads ("+str(len(cores))+", "+str(len(times))+", "+str(len(threads))+")."
#	plot_data.makePlot(data,"/home/jshearer/ParallelProjects/graphs/fractal/")

runComparison()
