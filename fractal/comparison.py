import call_utils
import fractal_data #Put these here to avoid the pesky "Compiled and got function Gen" lag from compiling the kernel
import plot_data

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
    execData = {'blocks':arguments.blocks,
                'threads':arguments.threads}

    print("Doing timing run, execData: "+str(execData))

    for mode in arguments.modes:
        print "Mode "+str(mode)+":"
        nExec=None
        try:
            nExec = fractal_data.cudaCollect(arguments.pos,arguments.zoom,arguments.dim,execData,mode=mode)
        except Exception, e:
            print e

def runGeneration(arguments):
    if arguments.procCuda:
        print("Doing generation ("+arguments.name+") using CUDA")

        result, time, blocks, threads = call_utils.callCUDA(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,
            block=arguments.blocks,thread=arguments.threads,save=arguments.save)

    elif arguments.procCpu:
        print("Doing generation ("+arguments.name+") using the CPU")
        time = call_utils.callCPU(arguments.pos,arguments.zoom,arguments.dim,arguments.name,iterations=arguments.iter,save=arguments.save)

    print "("+("CUDA" if arguments.procCuda else "CPU")+") run took "+str(time)+"s."        

if __name__ == '__main__':

    from argparse import ArgumentParser

    # example: python -O comparison.py --dimensions 2048 1024  --zoom 1800.  --position -1.5  0.0  --iterations 100  generate --save  --cuda

    parser = ArgumentParser(description="Run fractal simulation, for comparison or timing purposes.")
    subparsers = parser.add_subparsers(title="Available actions")

    parser.add_argument("--position","-p",    action="store",      dest="pos",  default=[0,0],       help="Fractal position offset",        nargs=2, metavar=('x','y'), type=float)
    parser.add_argument("--dimensions","-d",  action="store",      dest="dim",  default=[1024,1024], help="Dimensions of resulting image.", nargs=2, metavar=('w','h'), type=int)
    parser.add_argument("--zoom","-z",        action="store",      dest="zoom", default=400,         help="Fractal dilation. Scales around positoin.", type=float)
    parser.add_argument("--iterations",'-i',  action="store",      dest="iter", default=200,         help="Maximum number of iterations for a specific pixel.", type=int)
    
    timing_parser = subparsers.add_parser("timing",    help="Do timing run using cudaCollect.")
    comp_parser   = subparsers.add_parser("comparison",help="Compare times between CUDA and multithreaded CPU runs.")
    gen_parser    = subparsers.add_parser("generate",  help="Do one generation, and either display or save the generated fractal.")
    
    timing_parser.set_defaults(func=runTiming)
    comp_parser.set_defaults(func=runComparison) #So that I can easily call their respective functions later
    gen_parser.set_defaults(func=runGeneration)

    timing_parser.add_argument("--blocks","-b", action="store", dest="blocks", default=range(1,2048), help="List of block counts to use, seperated by spaces.",  nargs="+", type=int)
    timing_parser.add_argument("--threads","-t",action="store", dest="threads",default=range(1,1024), help="List of thread counts to use, seperated by spaces.", nargs="+", type=int)
    timing_parser.add_argument("--modes","-m",  action="store", dest="modes",  default=range(0,3),    help="List of modes to use, seperated by spaces.",         nargs="+", type=int)

    comp_parser.add_argument("--name","-n",     action="store",      dest="name",     default="untitled", help="The name of the comparison for use in saving to flies.", type=str)
    comp_parser.add_argument("--save","-s",     action="store_true", dest="save",     default=False,      help="If set, will save to file specified by --name")
    comp_parser.add_argument("--blocks","-b",   action="store",      dest="blocks",   default=1,          help="Number of blocks to use for the run.", type=int)
    comp_parser.add_argument("--threads","-t",  action="store",      dest="threads",  default=1,          help="Number of threads to use for the run.", type=int)
    comp_parser.add_argument("--mode","-m",     action="store",      dest="mode",     default=0,          help="Action ID to use for the run.", type=int)

    group = gen_parser.add_argument_group(title="Processor to use").add_mutually_exclusive_group(required=True)
    group.add_argument("--cuda","--gpu","-g",  action="store_true", dest="procCuda", default=False,      help="Use the GPGPU/CUDA processor to do the run.")
    group.add_argument("--cpu","-c",           action="store_true", dest="procCpu",  default=False,      help="Use the multithreaded CPU generator for the run.")
    gen_parser.add_argument("--blocks","-b",   action="store",      dest="blocks",   default=1,          help="Number of blocks to use for the run.", type=int)
    gen_parser.add_argument("--threads","-t",  action="store",      dest="threads",  default=1,          help="Number of threads to use for the run.", type=int)
    gen_parser.add_argument("--mode","-m",     action="store",      dest="mode",     default=0,          help="Mode ID to use for the run.", type=int)
    gen_parser.add_argument("--name","-n",     action="store",      dest="name",     default="untitled", help="The name of the comparison for use in saving to flies.", type=str)
    gen_parser.add_argument("--save","-s",     action="store_true", dest="save",     default=False,      help="If set, will save to file specified by --name")

    args = parser.parse_args()

    print("Doing ("+args.func.__name__+") run. Position: "+str(args.pos)+", Dimensions: "+str(args.dim)+", Zoom: "+str(args.zoom)+", Iterations: "+str(args.iter))

    args.func(args)

