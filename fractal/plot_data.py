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
_modeD = {0:'write',1:'read/write',2:'raw compute',3:'atomicAdd test + write',4:'overlap'}

def _create_plot(axisT, mode, title, legendT, show, save, fn):
    """2D plot, with right hand colorbar, and optional overlay for another curve"""
    x_axis, xlabel, xlog = axisT[0]
    y_axis, ylabel, ylog = axisT[1]
    c_axis, clabel = axisT[2]
    if mode==4:
        o_axis, olabel, olog = axisT[3]

    fig = plt.figure(figsize = (12,7)) # dpi=350

    gspec = gridspec.GridSpec(1,3,width_ratios=[30,4,1])

    ax = plt.subplot(gspec[0])

    sc = ax.scatter(x_axis,y_axis,c=c_axis,marker="+",norm=pltlib.colors.LogNorm())    
    cb_ax = plt.subplot(gspec[2])

    cbar = fig.colorbar(sc, cax=cb_ax)
    cbar.set_label(clabel)

    if mode==4:
        ax2 = ax.twinx()
        ax2.ticklabel_format(style="sci",scilimits=(0,0),useOffset=False)
        ov = ax2.scatter(x_axis,o_axis,c="k",marker="o",alpha=0.2)
        ax2.set_ylabel(olabel)
        ax2.set_ylim(o_axis.min(),o_axis.max())
        ax.legend((sc,ov),legenT,scatterpoints=4,loc="upper right")
    else:
        ax.legend((sc,), legendT,scatterpoints=4,loc="upper center")

    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    
    if xlog:
        ax.set_xscale('log')
        if mode==4 and olog:
            ax2.set_xscale('log')
    if ylog:
        ax.set_yscale('log')
        if mode==4 and olog:
            ax2.set_yscale('log')
    ax.grid(True)
    ax.axis('tight')

    fig.suptitle(title)
    plt.tight_layout()
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

    if save:
        with open(fn,'w') as f:
            print 'Saving to ', fn
            plt.savefig(f, figsize = (12,7)) 

    # http://stackoverflow.com/questions/9012487/matplotlib-pyplot-savefig-outputs-blank-image
    if show: plt.show()  # do nothing if Agg

    plt.clf()


def loop_queue_plot(timeL, blocksL, pptL, dimx, dimy, threads, mode, xaxis, idS='X', save=False, show=True):
    y_axis = np.array(timeL,dtype=np.float)
    blocks = np.array(blocksL,dtype=np.float)
    ppt = np.array(pptL,dtype=np.float)

    if xaxis == 'px_per_thread':
        x_axis = ppt
        c_axis = blocks
        clabel = 'Blocks'
    elif xaxis == 'blocks':
        x_axis = blocks
        c_axis = ppt
        clabel = 'px_per_thread'

    xlog=True
    ylog=False

    xlabel="Number of CUDA %s"%xaxis+(", log10" if xlog else "")
    ylabel="Time to compute "+("log10" if ylog else "")+"(seconds)"

    title = "Queue vs looping: pixels=(%d, %d) #threads = %d  mode=%s"%(dimx, dimy, threads, _modeD[mode])
    fn = 'loop-queue-%s-%s'%(xaxis, idS)

    axisT = ( (x_axis,xlabel,xlog), (y_axis, ylabel, ylog), (c_axis, clabel) )
    
    _create_plot(axisT, mode, title, ('Time', ), show=show, save=save, fn=fn)


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
    y_axis = np.array(times,dtype=np.float)
    dimensions = np.array(dimensions, dtype=np.int)
    o_axis=None
    if mode==4:
        o_axis = np.array(data[8],dtype=np.float)
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

    olabel = "Overlap (# of pixels)"
    legendT = ('Time', )
    if mode == 4:
        legendT.append('Overlap')
        
    xlabel="Number of CUDA %s"%xaxis+(", log10" if xlog else "")
    ylabel="Time to compute "+("log10" if ylog else "")+"(seconds)"

    modeS = _modeD[mode]

    title="Fractal generation ["+modeS+"]\nDimensions: "+str(dimensions)+"\nZoom: "+str(zoom) 
    fn = directory+"mode_"+modeS+"-id"+str(index)+".png"

    axisT = ( (x_axis,xlabel,xlog), (y_axis, ylabel, ylog), (c_axis, clabel), (o_axis, olabel, ovlog) )

    _create_plot(axisT, mode, title, legendT, show=show, save=save, fn=fn)
    
def _graph(arguments):

    if not (arguments.show or arguments.save):
        parser.error("You must choose --save or --show, otherwise whadya-want-from-me???")

    if not arguments.index:
        dataD = extractMetaData()
        dataL = dataD.keys()
        dataL.sort()
        for dataA in dataL:
            print '%s: %s'%(dataA, dataD[dataA])
        chc = str( raw_input('\nEnter id (RET to exit): ') )
        if len(chc)==0:
            print 'exiting ... '
            exit(0)
        chc = int(chc)
        print '\nYou chose %d: %s'%(chc, dataD[chc])
        arguments.index = chc

    xaxisD = {'cores': 'cores', 'tpc': 'threads_per_core', 'threads': 'total_threads', 'ppt': 'px_per_thread'}
    if arguments.xaxis not in xaxisD.keys():
        parser.error("--axis=XXX, XXX must be one of cores, tpc, threads, ppt")
    
    print 'You selected index = %s'%arguments.index
    try:
        data = extractCols(arguments.index)
    except NoSuchNodeError, e:
        print e
        exit(0)
        
    makePlot(data,"results/",xlog=arguments.xlog, ylog=arguments.ylog, ovlog=arguments.ovlog,
             show=arguments.show, save=arguments.save, xaxis=xaxisD[arguments.xaxis])
     
if __name__ == '__main__':
    from argparse import ArgumentParser

    from fractal_data import NoSuchNodeError, extractMetaData, extractCols, init

    # example: python -O plot_data.py  --save &/| --store [OPTIONS]

    parser = ArgumentParser(description="Plot stored results for fractal simulations")

    parser.add_argument("--xaxis",  action="store",       dest="xaxis", default="cores", help="Choose X axis: cores (default), tpc (threads_per_core), threads (total_threads), ppt (px_per_thread)")
    parser.add_argument("--ylog",   action="store_true",  dest="ylog",  default=True,    help="use Log10 y (default)")
    parser.add_argument("--noylog", action="store_false", dest="ylog",                   help="do not use log10 y")
    parser.add_argument("--xlog",   action="store_true",  dest="xlog",  default=True,    help="use Log10 x (default)")
    parser.add_argument("--noxlog", action="store_false", dest="xlog",                   help="do not use log10 x")
    parser.add_argument("--ovlog",  action="store_true",  dest="ovlog", default=False,   help="use Log10 overlap")
    
    parser.add_argument("--datasrc",action="store",       dest="h5file",default="fractalData.h5",   help="hdf5 data file")
        
    parser.add_argument("--show",   action="store_true",  dest="show",  default=False,   help="Display plot if DISPLAY is set")
    parser.add_argument("--save",   action="store_true",  dest="save",  default=False,   help="Store plot as PNG")
    parser.add_argument("--index",  action="store",       dest="index", default=None,    help="Data set identifier (index); if none, you will be prompted with a list of existing data")

    args = parser.parse_args()

    init(filename=args.h5file)
    _graph(args)

