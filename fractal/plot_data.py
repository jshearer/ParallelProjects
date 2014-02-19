import matplotlib
matplotlib.use('Agg') #to use savefig without DISPLAY set
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogFormatter
import matplotlib as pltlib
import time
import numpy as np

def makePlot(data,directory,xlog=True,ylog=False):
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
#	threads = threads / threads.max()
#	threads = np.log10(threads)
###############################################
	#times = times / times.max()
#	times = np.log10(times)
###############################################
#	cores = np.log10(cores)
##############################################
	print threads_max

	#threads = [(x,x,x) for x in threads]
	print threads
	print cores
	print times

	#plt.plot(x_coords,y_coords,'ro')

	x_axis = cores
	y_axis = times
	
	fig = plt.figure(figsize = (12,7),dpi=350)

	ax = fig.add_subplot(111) 

	ax.set_xlim([x_axis.min()-(x_axis.min()/10.0),x_axis.max()+(x_axis.max()/10.0)])
	ax.set_ylim([y_axis.min()-(y_axis.min()/10.0),y_axis.max()+(y_axis.max()/10.0)])

	sc = ax.scatter(x_axis,y_axis,c=threads,marker="+",norm=pltlib.colors.LogNorm())	

	cbar = plt.colorbar(sc,use_gridspec=True)
	cbar.set_label("Number of threads per core")

	ax.set_ylabel("Time to compute "+("log10" if ylog else "")+"(seconds)")
	ax.set_xlabel("Number of CUDA cores"+(", log10" if xlog else ""))

	if xlog:
		ax.set_xscale('log')
	if ylog:
		ax.set_yscale('log')

	ax.grid(True)

	title_identifier = {0:'write',1:'read and write',2:'raw compute',3:'atomicAdd test + regular write'}[mode]

	plt.title("Fractal generation ["+title_identifier+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom))
	plt.tight_layout()
	print "Saving file now."	
	with open(directory+"mode_"+str(mode)+"-id"+str(index)+".png",'w') as f:
		plt.savefig(f)
	plt.clf()
