from cpu_fractal import fractal_2 as cpu_f
try:
    from cuda_fractal.pycuda import fractal as cuda_f
except ImportError, e:
    print e
    print "you are running w/o CUDA hardware/software present?"
    
import render
import time

# TODO: pass in the function to be run and its args and kwargs, to make these generic run and time functions.
# the RETURN is likewise passed back out, perhaps with the time inserted into the tuple.__add__

# TODO: move all function specific code (like render) back up to caller.
def callCPU(position, zoom, dimensions, name, iterations=100,scale=1,save=True):
    start = time.time()
    result = cpu_f.gen(position,zoom,dimensions,iterations=iterations,silent=True,scale=scale)
    elapsed = time.time()-start
    if save:
        print 'Saving image ... '
        # TODO: SaveToPngThread produces nothing if caller exits before this new thread is done.
        render.SaveToPng(result,"cpu_"+name,render.colors['default'],silent=False)
    return elapsed


def callCUDA(position, zoom, dimensions, name, iterations=100,scale=1,save=True,block=(5,5,1),thread=(1,1,1),mode=0):
    result,time,block,thread = cuda_f.GenerateFractal(dimensions,position,zoom,iterations,silent=True,
                                                      debug=False,action=mode,scale=scale,block=block,thread=thread)
    if save:
        render.SaveToPngThread(result,"cuda_mode"+str(mode)+"-"+name,render.colors['default'],silent=False)
    return (result,time,block,thread)
