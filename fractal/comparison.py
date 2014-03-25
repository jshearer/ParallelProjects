from math import sqrt

import call_utils
import plot_data
import fractal_data 
from utils import allocate_cores

def runComparison(arguments):
    print "Comparing parameters for (" +(str(arguments.name) if arguments.save else "")+ "): ( (" +str(arguments.pos[0])+","+str(arguments.pos[1])+"), ",
    print str(arguments.zoom)+", ("+str(arguments.dim[0])+", "+str(arguments.dim[1])+") )"
        
    cpuTime = call_utils.callCPU(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,save=arguments.save)
    try:
        results, cudaTime, retblock, retthread = call_utils.callCUDA(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,
            block=arguments.blocks,thread=arguments.threads,save=arguments.save,mode=arguments.mode)
    except Exception, e:
        print e
        cudaTime = 'NA'
        pass

    print "CPU ran in "+str(cpuTime)+"s"
    print "CUDA ran in "+str(cudaTime)+"s"

def runTiming(arguments):
    execData = {'blocks':arguments.blocksL,
                'threads':arguments.threadsL}

    print("Doing timing run, execData: "+str(execData)[0:64])+'...'
    fractal_data.init()
    for mode in arguments.modesL:
        print "Mode "+str(mode)+":"
        index=None
        try:
            index = fractal_data.cudaCollect(arguments.pos,arguments.zoom,arguments.dim,execData,mode=mode)
        except Exception, e:
            print e
            raise
        
        if arguments.show and index:
            data = fractal_data.extractCols(index)
            plot_data.makePlot(data,"results/", ylog=True, show=True, save=False)
    

def runGeneration(arguments):
    if arguments.procCuda:
        print("Doing generation ("+arguments.name+") using CUDA")

        result, time, blocks, threads = call_utils.callCUDA(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,
            block=arguments.blocks,thread=arguments.threads,save=arguments.save)

    elif arguments.procCpu:
        print("Doing generation ("+arguments.name+") using the CPU")
        time = call_utils.callCPU(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,save=arguments.save)

    print "("+("CUDA" if arguments.procCuda else "CPU")+") run took "+str(time)+"s."        

def runQueueLoopComparison(args):
    """1. lets take pixels_per_thread for queueing case greater than 1, so
    lets say 10, 50

    2. blocks*threads = dimx*dimy/ px_per_thread, so if we take
    threads = 512, px_per_thread = 50, and blocks = 4096, then that
    defines dimx*dimy

    dimx * dimy = 4096*512 * px_per_thread

    3. pick a dimx and a dimy

    4. run that dimx, dimy, px_per_thread=10, or 50, say, 4096 blocks, 512 threads.

    5, then run the same but with blocks = 512

    zoom=900
    iterations = 100
                            px
                           per
     dimx  dimy blks thds  thd      time
    -----------------------------------------
    19968 19968 4992 1024   78  0.408787567139
    19968 19968 2496 1024  156  0.447927703857

    19968 19968 4992 1024   78  0.4087394104
    19968 19968 1248 1024  312  0.464446563721
 
    29952 29952 4992  512  351  1.17091418457
    29952 29952 2496  512  702  1.18278723145

    29952 29952 4992  512  351  1.17207836914
    29952 29952 1248  512 1404  1.19653564453

    even with 4:1 W::H:
    Doing (runQueueLoopComparison) run. Position: [-1.3, 0.0], Dimensions: [1024, 1024], Zoom: 900.0, Iterations: 100
    59904 14976 4992 512 351 1.16753649902
    59904 14976 2496 512 702 1.18160949707


    """
    dimx,dimy=args.dim
    resultL = allocate_cores(dimx, dimy, args.threads, silent=True)
    timeL = []
    pptL = []
    blocksL = []
    for ppt, threads, blocks in resultL:
        # eventually call 3-5 times and average
        result, time,blocksA, threadsA = call_utils.callCUDA(args.pos,args.zoom, (dimx,dimy),args.name,iterations=args.iter,
                                                              block=blocks,thread=threads,save=args.save, 
                                                              mode=args.mode) 
        
        timeL.append(time)
        blocksL.append(blocks)
        pptL.append(ppt)
        print '%6d,%6d:   %6d (%14s)  %4d (%12s) %5d   %f'%(dimx, dimy, blocks, blocksA, threads, threadsA, ppt, time)
        
    plot_data.loop_queue_plot(timeL, blocksL, pptL, dimx, dimy, threads, args.mode, xaxis='px_per_thread')
    plot_data.loop_queue_plot(timeL, blocksL, pptL, dimx, dimy, threads, args.mode, xaxis='blocks')
    

if __name__ == '__main__':

    from argparse import ArgumentParser

    # example: python -O comparison.py --dimensions 2048 1024  --zoom 1800.  --position -1.5  0.0  --iterations 100  --save  generate  --cuda

    parser = ArgumentParser(description="Run fractal simulation, for comparison or timing purposes.")
    subparsers = parser.add_subparsers(title="Available actions")

    parser.add_argument("--position","-p",    action="store",      dest="pos",  default=[0,0],       help="Fractal position offset",        nargs=2, metavar=('x','y'), type=float)
    parser.add_argument("--dimensions","-d",  action="store",      dest="dim",  default=[1024,1024], help="Dimensions of resulting image.", nargs=2, metavar=('w','h'), type=int)
    parser.add_argument("--zoom","-z",        action="store",      dest="zoom", default=400,         help="Fractal dilation. Scales around positoin.", type=float)
    parser.add_argument("--iterations",'-i',  action="store",      dest="iter", default=200,         help="Maximum number of iterations for a specific pixel.", type=int)

    parser.add_argument("--blocks","-b",   action="store",      dest="blocks",   default=1,          help="Number of blocks to use for the run.", type=int)
    parser.add_argument("--threads","-t",  action="store",      dest="threads",  default=1,          help="Number of threads to use for the run.", type=int)
    parser.add_argument("--mode","-m",     action="store",      dest="mode",     default=0,          help="Action ID to use for the run.", type=int)
    parser.add_argument("--name","-n",     action="store",      dest="name",     default="untitled", help="The name of the comparison for use in saving to flies.", type=str)
    parser.add_argument("--save","-s",     action="store_true", dest="save",     default=False,      help="If set, will save to file specified by --name")
    parser.add_argument("--show",          action="store_true", dest="show",     default=False,      help="Display timing plot if DISPLAY is set.")

    timing_parser = subparsers.add_parser("timing",    help="Do timing run using cudaCollect.")
    comp_parser   = subparsers.add_parser("comparison",help="Compare times between CUDA and multithreaded CPU runs.")
    gen_parser    = subparsers.add_parser("generate",  help="Do one generation, and either display or save the generated fractal.")
    queue_parser  = subparsers.add_parser("queue",  help="Compare queueing and looping.")
    
    timing_parser.set_defaults(func=runTiming)
    comp_parser.set_defaults(func=runComparison) #So that I can easily call their respective functions later
    gen_parser.set_defaults(func=runGeneration)
    queue_parser.set_defaults(func=runQueueLoopComparison)

    timing_parser.add_argument("--blocksL","-b", action="store",      dest="blocksL", default=range(1,2048), help="List of block counts to use, seperated by spaces.",  nargs="+", type=int)
    timing_parser.add_argument("--threadsL","-t",action="store",      dest="threadsL",default=range(1,1024), help="List of thread counts to use, seperated by spaces.", nargs="+", type=int)
    timing_parser.add_argument("--modesL","-m",  action="store",      dest="modesL",  default=range(0,3),    help="List of modes to use, seperated by spaces.",         nargs="+", type=int)

    queue_parser.add_argument("--pxperthread",   action="store",       dest="px_per_thread",   default=20,   help="Number of pixels per thread minimum", type=int)

    group = gen_parser.add_argument_group(title="Processor to use").add_mutually_exclusive_group(required=True)
    group.add_argument("--cuda","--gpu","-g",  action="store_true", dest="procCuda", default=False,      help="Use the GPGPU/CUDA processor to do the run.")
    group.add_argument("--cpu","-c",           action="store_true", dest="procCpu",  default=False,      help="Use the multithreaded CPU generator for the run.")

    args = parser.parse_args()

    print("Doing ("+args.func.__name__+") run. Position: "+str(args.pos)+", Dimensions: "+str(args.dim)+", Zoom: "+str(args.zoom)+", Iterations: "+str(args.iter))

    args.func(args)

