from os import environ
if environ.has_key('DISPLAY'):
	ren='WXAgg'
else:
	ren = 'Agg'  # Agg to use savefig without DISPLAY set

import matplotlib
matplotlib.use(ren)
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, LogFormatter
from matplotlib import gridspec
import matplotlib as pltlib
import time
import numpy as np

_VERBOSE = False

def makePlot(data,directory,xaxis='cores',xlog=True,ylog=False,ovlog=False,show=False,save=True):
    cores      = data[0]
    times      = data[1]
    threads    = data[2]
    zoom       = data[3]
    mode       = data[4]
    dimensions = data[5]
    iterations = data[6]
    index      = data[7]

    threads_max = max(threads)
    threads_min = min(threads)
###############################################
    threads = np.array(threads,dtype=np.float)
    cores = np.array(cores,dtype=np.float)
    times = np.array(times,dtype=np.float)
    dimensions = np.array(dimensions, dtype=np.int)
    if mode==4:
        overlap = np.array(data[8],dtype=np.float)
###############################################
#   threads = threads / threads.max()
#   threads = np.log10(threads)
###############################################
    #times = times / times.max()
#   times = np.log10(times)
###############################################
#   cores = np.log10(cores)
##############################################
    if _VERBOSE:
        print threads_max
        # threads = [(x,x,x) for x in threads]
        print threads
        print cores
        print times

    # plt.plot(x_coords,y_coords,'ro')

    if xaxis == 'cores':
        x_axis = cores
        c_axis = threads
        clabel = 'Threads per core'
    elif xaxis == 'threads_per_core':
        x_axis = threads
        c_axis = cores
        clabel = 'Blocks'
    elif xaxis == 'total_threads':
        x_axis = cores*threads
        c_axis = cores
        clabel = 'Blocks'
    elif xaxis == 'px_per_thread':
        x_axis = dimensions.prod() / cores / threads
        c_axis = cores
        clabel = 'Blocks'
        
    y_axis = times
    
    fig = plt.figure(figsize = (12,7)) # dpi=350

    gspec = gridspec.GridSpec(1,3,width_ratios=[30,4,1])

    ax = plt.subplot(gspec[0])

    #ax.set_xlim([-(x_axis.min()/10.0),x_axis.max()+(x_axis.max()/10.0)])
    #ax.set_ylim([-(y_axis.min()/10.0),y_axis.max()+(y_axis.max()/10.0)])

    sc = ax.scatter(x_axis,y_axis,c=c_axis,marker="+",norm=pltlib.colors.LogNorm())    
    cb_ax = plt.subplot(gspec[2])

    cbar = fig.colorbar(sc, cax=cb_ax)
    cbar.set_label(clabel)

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
        ax.legend((sc,), ('Time',),scatterpoints=4,loc="upper center")
    
    ax.set_ylabel("Time to compute "+("log10" if ylog else "")+"(seconds)")
    ax.set_xlabel("Number of CUDA %s"%xaxis+(", log10" if xlog else ""))

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
    title_identifier = {0:'write',1:'read and write',2:'raw compute',
                        3:'atomicAdd test + regular write',4:'Overlap'}[mode]

    fig.suptitle("Fractal generation ["+title_identifier+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom))
    plt.tight_layout()
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if save:
        print "Saving file now."
        fn = directory+"mode_"+str(mode)+"-id"+str(index)+".png"
        with open(fn,'w') as f:
            print 'Saving to ', fn
            plt.savefig(f, figsize = (12,7)) 

    # http://stackoverflow.com/questions/9012487/matplotlib-pyplot-savefig-outputs-blank-image
    if show: plt.show()  # do nothing if Agg

    plt.clf()
     
