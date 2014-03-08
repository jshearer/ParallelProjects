import matplotlib
matplotlib.use('Agg') #to use savefig without DISPLAY set
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogFormatter
from matplotlib import gridspec
import matplotlib as pltlib
import time
import numpy as np

def makePlot(data,directory,xlog=True,ylog=False,ovlog=False, show=False):
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
	if mode==4:
		overlap = np.array(data[8],dtype=np.float)
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

	gspec = gridspec.GridSpec(1,3,width_ratios=[30,4,1])

	ax = plt.subplot(gspec[0])

	#ax.set_xlim([-(x_axis.min()/10.0),x_axis.max()+(x_axis.max()/10.0)])
	#ax.set_ylim([-(y_axis.min()/10.0),y_axis.max()+(y_axis.max()/10.0)])

	sc = ax.scatter(x_axis,y_axis,c=threads,marker="+",norm=pltlib.colors.LogNorm())	
	cb_ax = plt.subplot(gspec[2])

	cbar = fig.colorbar(sc, cax=cb_ax)
	cbar.set_label("Number of threads per core")

	if mode==4:
		ax2 = ax.twinx()
		ax2.ticklabel_format(style="sci",scilimits=(0,0),useOffset=False)
		print overlap.min(), overlap.max()
		ov = ax2.scatter(x_axis,overlap,c="k",marker="o",alpha=0.2)
		ax2.set_ylabel("Overlap (# of pixels)")
		ax2.set_ylim(overlap.min(),overlap.max())
		ax.legend((sc,ov),('Time','Overlap'),scatterpoints=4,loc="upper right")
		pass
	else:
		ax.legend((sc,), ('Time',),scatterpoints=4,loc="upper right")
	
	ax.set_ylabel("Time to compute "+("log10" if ylog else "")+"(seconds)")
	ax.set_xlabel("Number of CUDA cores"+(", log10" if xlog else ""))

	if xlog:
		ax.set_xscale('log')
		if mode==4 and ovlog:
			ax2.set_xscale('log')
	if ylog:
		ax.set_yscale('log')
		if mode==4 and ovlog:
			ax2.set_yscale('log')
	ax.grid(True)
	ax.axis('tight')
	title_identifier = {0:'write',1:'read and write',2:'raw compute',3:'atomicAdd test + regular write',4:'Overlap'}[mode]

	fig.suptitle("Fractal generation ["+title_identifier+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom))
	plt.tight_layout()
	
	plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
	if show: plt.show()  # doesn't work with Agg???
	
	print "Saving file now."
	fn = directory+"mode_"+str(mode)+"-id"+str(index)+".png"
	with open(fn,'w') as f:
		print 'Saving to ', fn
		plt.savefig(f)
	plt.clf()

if __name__ == '__main__':
	import fractal_data
	from sys import  argv
#	print "Inserted into index: "+str(index)
	if len(argv) == 2:
		num = argv[1]
	else:
		num = 1

	print 'Displaying index = %s'%num
	data = fractal_data.extractCols(num)
#	print "len cores,times,threads ("+str(len(cores))+", "+str(len(times))+", "+str(len(threads))+")."
	makePlot(data,"results/",ylog=True, show=True)
	
