import matplotlib
matplotlib.use('Agg') #to use savefig without DISPLAY set
import matplotlib.pyplot as plt
import matplotlib as pltlib
import time
import numpy as np

def makePlot(data,directory):
	cores      = data[0]
	times      = data[1]
	threads    = data[2]
	zoom       = data[3]
	mode       = data[4]
	dimensions = data[5]
	iterations = data[6]
	index 	   = data[7]

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
	#times = times / times.max()
	#times = np.log10(times)
###############################################
	cores = np.log10(cores)
##############################################
	print threads_max

	#threads = [(x,x,x) for x in threads]
	print threads
	print cores
	print times

	#plt.plot(x_coords,y_coords,'ro')

	plt.scatter(cores,times,c=threads,marker="+")

	plt.ylabel("Time to compute (seconds)")
	plt.xlabel("Number of CUDA cores, log10")

	plt.legend('bottom right', scatterpoints=5)

	title_identifier = {0:'write',1:'read and write',2:'raw compute',3:'atomicAdd test + regular write'}[mode]

	plt.title("Fractal generation ["+title_identifier+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom)+)
	plt.tight_layout()
	
	with open(directory+"mode_"+str(mode)+"-id"+str(index)+".png",'w') as f:
		plt.savefig(f)
	plt.clf()
