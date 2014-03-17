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
    overlap           = tab.Int64Col()
    time              = tab.Float64Col()

mode_identifier = {0:'write',1:'read+write',2:'no_rw', 3:'atomicAdd+write',4:'Overlap'}
    
class MetaData(tab.IsDescription):
    index             = tab.Int32Col()    
    pos_x             = tab.Float64Col()
    pos_y             = tab.Float64Col()
    zoom              = tab.Float32Col()
    dimensions_x      = tab.Int32Col()
    dimensions_y      = tab.Int32Col()
    mode              = tab.UInt8Col()  # should be a string for easier comprehension. 
    iterations        = tab.Int32Col()

# TODO: we should supply data as dicts, or allied, so we can automagically populate rows
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

    # TODO: get rid of the explicit Overlap N stuff, mode tells us all
    # we need to know.
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

    foundL = []
    # isnt there a public func for _f_list_nodes()???
    # TODO: can go straight to the right node eventually and search ROWS of meta data
    nodeL = _data_file.root.execSets._f_list_nodes()
    
    tupL = (dimensions[0], dimensions[1], mode, position[0], position[1], zoom)
    # NOTE!!!!: the condx string very sensitive. when i replaced a '&' with 'and', all rows were taken!!!
    condS = "(dimensions_x==%d) & (dimensions_y==%d) & (mode == %d) & (pos_x == %f) & (pos_y == %f) & (zoom == %f)"%tupL
    print condS
    for node in nodeL:
        # node.meta is a table, search it
        for row in node.meta.where(condS):
            print row['dimensions_x'], row['dimensions_y'], row['mode'],  row['pos_x'], row['pos_y'],  row['zoom']
            foundL.append(node)
            # we need only know that at least one row in the metadata meets condx
            # TODO: later we can clean this up when we clean up the table structure
            break 
        
    return foundL

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
            desc = "Sets of execution with varying position,zoom,dimensions,blockData, or threadData"
            _data_file.createGroup("/","execSets",desc)

if __name__ == '__main__':

    if 0:
        print extractMetaData()

    if 1:
        newFile = tab.openFile('newFractalData.h5', mode='w', title='New Fractal timing data')
        grp = newFile.create_group("/", 'TimingData', 'Timing data for parallel execution')
        table = newFile.create_table('/TimingData', 'meta', MetaData, 'Fractal timing meta data')
        table = newFile.create_table('/TimingData', 'data', Execution, 'Fractal timing data')        

    if 1:
        _newFile = tab.openFile('newFractalData.h5', mode='a', title='New Fractal timing data')        
        theMetaTable = _newFile.root.TimingData.meta
        theDataTable = _newFile.root.TimingData.data   
        row = theMetaTable.row
        drow = theDataTable.row
        for i in range(2):
            row['dimensions_x'] = 20+i
            row['dimensions_y'] = 30+i
            row['iterations'] = 100+i
            row['mode'] = 2+i
            row['pos_x']= 2.1
            row['pos_y']= 1.7
            row['zoom']=900.
            row['index'] = i
            row.append()
            for j in range(3):
                drow['index'] = i
                drow['block_x'] = 1000+j
                drow['block_y'] = 1000+j
                drow['blocks'] = 1000+j
                drow['thread_x'] = 1000+j
                drow['thread_y'] = 1000+j
                drow['threads'] = 1000+j
                drow['time'] = 2.1+float(j)/3
                drow.append()


        theMetaTable.flush()
        theDataTable.flush()        
        
        
    if 0:
        
        res= alreadyRan( (-1.3,0.0), (2000, 1000), 900., 4)
        print res

