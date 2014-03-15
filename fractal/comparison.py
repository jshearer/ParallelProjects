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

    ## Each pixel (x,y) in the generated image is
    ## ((((x*zoom)-(dimensions.x/2))/zoom) + position.x) + ((((y*zoom)-(dimensions.y/2))/zoom) + position.y) i
    ## 
    ## So, for example. Position = [-1,0] Dimensions = [2000,1000], zoom = 1000.
    ## Top left corner: x=0, y=0.
    ## (Real part:) 0*1000 - (2000/2)/1000 + -1 = -1 - 1 = -2
    ## (Imag part:) 0*1000 - (1000/2)/1000 + 0 = -500/1000 = - 1/2i
    ## So in the complex plane, the top left corner of the image would map to -2 - 1/2i


    position = [0,0]           # centers the view????
    dimensions = [2048, 2048]   # H, W !!!!  this affects the area covered
    zoom = 500                  # some kind of zoom???
    modeL = (0,1) # range(0,5)
    
    if options.runT:
        runTiming()

    if options.runC:
        block=(5,5,1)
        thread=(1,1,1)
        runComparison('Check1', save=options.save)
