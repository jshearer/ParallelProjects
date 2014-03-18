import tables as tab

NoSuchNodeError = tab.exceptions.NoSuchNodeError

_data_file = None  # module private

class Execution(tab.IsDescription):
    index             = tab.Int32Col()
    block_x           = tab.Int32Col()
    block_y           = tab.Int32Col()
    blocks            = tab.Int32Col()
    thread_x          = tab.Int32Col()
    thread_y          = tab.Int32Col()
    threads           = tab.Int32Col()
    time              = tab.Float64Col()

class OverlapExecution(tab.IsDescription):
    index             = tab.Int32Col()
    block_x           = tab.Int32Col()
    block_y           = tab.Int32Col()
    blocks            = tab.Int32Col()
    thread_x          = tab.Int32Col()
    thread_y          = tab.Int32Col()
    threads           = tab.Int32Col()
    overlap           = tab.Int64Col()
    time              = tab.Float64Col()

mode_identifier = {0:'write',1:'read+write',2:'no_rw', 3:'atomicAdd+write',4:'Overlap'}
    
class MetaData(tab.IsDescription):
    pos_x             = tab.Float64Col()
    pos_y             = tab.Float64Col()
    zoom              = tab.Float32Col()
    dimensions_x      = tab.Int32Col()
    dimensions_y      = tab.Int32Col()
    mode              = tab.UInt8Col()  # should be a string for easier comprehension. 
    iterations        = tab.Int32Col()

def cudaCollect(position,zoom,dimensions,execData,mode=0,iterations=100):
    """
    Run callCUDA over a range of block and thread shapes and sizes, and collect data on time spent. 
    """
    from call_utils import callCUDA
    global _data_file

    _init()

    #Only compile the function when we need to
    overlap = (mode==4)

    nExec = len(_data_file.listNodes(getGroup()))

    if overlap:
        grp = _data_file.createGroup(getGroup(),"Overlap"+str(nExec), "Overlap run "+str(nExec+1))
    else:
        grp = _data_file.createGroup(getGroup(),str(nExec), "Execution number "+str(nExec+1))
        
    meta = _data_file.createTable(grp,"meta",MetaData,"Metadata")

    meta.row['pos_x'] = position[0]
    meta.row['pos_y'] = position[1]
    meta.row['dimensions_x'] = dimensions[0]
    meta.row['dimensions_y'] = dimensions[1]
    meta.row['zoom'] = zoom
    meta.row['mode'] = mode
    meta.row['iterations'] = iterations
    # TODO: add these!!!
    # git commit/hash
    # version info
    # frozen library data
    # os?

    meta.row.append()
    meta.flush()
    if overlap:
        data = _data_file.createTable(grp,"data",OverlapExecution,"Real data")   
    else:
        data = _data_file.createTable(grp,"data",Execution,"Real data")

    for block in execData['blocks']:
        for thread in execData['threads']:              
            try:
                name=str(block)+", "+str(thread)
                result,time,block_dim,thread_dim = callCUDA(position,zoom,dimensions,name,
                                                            block=block,thread=thread,save=False,mode=mode)
            except ValueError:
                continue

            print "GOOD \t"+str(block)+", "+str(thread)+": "+str(time)
            
            if overlap:     
                data.row['overlap'] = calculateOverlap(result)
        
            data.row['time'] = time
            data.row['index'] = len(data)
            data.row['block_x'] = block_dim[0]
            data.row['block_y'] = block_dim[1]
            data.row['blocks'] = block
            data.row['thread_x'] = thread_dim[0]
            data.row['thread_y'] = thread_dim[1]
            data.row['threads'] = thread
            

            data.row.append()
            data.flush()
    return nExec

def calculateOverlap(result):
    import numpy as np
    return np.sum(result)-(result.shape[0]*result.shape[1])

def alreadyRan(position,dimensions,zoom,mode):
    global _data_file
    _init()

    # TODO: replace this with a tables search
    for node in _data_file.walkNodes('/execSets'):
        pos_x = (node.meta['pos_x']==position[0])
        pos_y = (node.meta['pos_y']==position[1])
        pos = pos_x and pos_y

        dim_x = (node.meta['dimensions_x']==dimensions[0])
        dim_y = (node.meta['dimensions_y']==dimensions[1])
        dim = dim_x and dim_y

        zoom_check = (node.meta['zoom']==zoom)
        mode_check = (node.meta['mode']==mode)
    
        if pos and dim and zoom_check and mode_check:
            return True

    return False

def extractCols(nExec):
    global _data_file
    _init()

    meta = _data_file.getNode("/execSets/"+str(nExec)+"/meta")

    blocks  = []
    times   = []
    threads = []
    overlap = []
    
    data = _data_file.getNode("/execSets/"+str(nExec)+"/data")

    for execution in data:
        blocks.append(execution['blocks'])
        threads.append(execution['threads'])
        times.append(execution['time'])
        if meta[0]['mode']==4:
            overlap.append(execution['overlap'])

    iters = 0
    if 'iterations' in meta[0]:
        iters = meta[0]['iterations']

    return (blocks,times,threads,meta[0]['zoom'],meta[0]['mode'],
            (meta[0]['dimensions_x'],meta[0]['dimensions_y']),iters,nExec,overlap)

def extractMetaData():
    global _data_file    
    _init()

    nameL = [ e._v_name for e in _data_file.root.execSets ]

    metaD={}
    for name in nameL:
        meta = _data_file.getNode("/execSets/"+name+"/meta")
        metaD[name] = { key:meta[0][key] for key in meta.colnames}

    return metaD
    
def getGroup():
    global _data_file
    _init()
    return _data_file.root.execSets

def _init():
    global _data_file

    if _data_file == None:
        filename = "fractalData.h5"
        _data_file = tab.openFile(filename,mode='a',title="Fractal timing data")
        if not ("/execSets" in _data_file):
            _data_file.createGroup("/","execSets","Sets of execution with varying position,zoom,dimensions,blockData, or threadData")

if __name__ == '__main__':
    
    print extractMetaData()
