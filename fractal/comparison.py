import call_utils
import fractal_data
import plot_data

def runComparison(name,iterations=100,save=True):
    print "Comparing parameters for (" +str(name)+ "): ( (" +str(position[0])+","+str(position[1])+"), ",
    print str(zoom)+", ("+str(dimensions[0])+", "+str(dimensions[1])+") )"
        
    cpuTime = call_utils.callCPU(position,zoom,dimensions,name,iterations,save=save)
    try:
        cudaTime = call_utils.callCUDA(position,zoom,dimensions,name,iterations,block=block,thread=thread,save=save)
    except:
        cudaTime = 'NA'
        pass

    print "CPU ran in "+str(cpuTime)+"s"
    print "CUDA ran in "+str(cudaTime)+"s"

def runTiming():
    execData = {'blocks':range(1,2049),
                'threads':range(1,1025)}

    for mode in modeL:
        print "Mode "+str(mode)+":"
        nExec=None
        try:
            nExec = fractal_data.cudaCollect(position,zoom,dimensions,execData,mode=mode)
        except Exception, e:
            print e

        if options.show and nExec:
            data = fractal_data.extractCols(nExec)
            plot_data.makePlot(data,"results/", ylog=True, show=True, save=False)
    
if __name__ == '__main__':

    from optparse import OptionParser
    usage="python [-O] comparison.py [--timing | --comparison] [other OPTIONS] "+\
      " Run fractal simulation, for comparison or timing purposes."
    parser = OptionParser(version="1.0", usage=usage)
    parser.add_option("--timing", action="store_true", dest="runT", default=False, help="Run the timing mode")
    parser.add_option("--comparison", action="store_true", dest="runC", default=False, help="Run the comparison mode")    
    parser.add_option("--show", action="store_true", dest="show", default=False, help="Display plot if DISPLAY is set")
    parser.add_option("--save", action="store_true", dest="save", default=False, help="Store plot as PNG")
    (options, args) = parser.parse_args()

    # TODO: can convert these to options as well
    position = [-1.3,0]
    dimensions = [2000,1000]
    zoom = 900
    modeL = (0,1) # range(0,5)
    
    if options.runT:
        runTiming()

    if options.runC:
        block=(5,5,1)
        thread=(1,1,1)
        runComparison('Check1', save=options.save)
