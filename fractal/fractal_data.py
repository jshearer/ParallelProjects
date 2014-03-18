import tables as tab
import numpy as np

NoSuchNodeError = tab.exceptions.NoSuchNodeError

_data_file = None  # module private

class ExecutionData(tab.IsDescription):
    metaIndexFK       = tab.Int32Col()
    block_x           = tab.Int32Col()
    block_y           = tab.Int32Col()
    blocks            = tab.Int32Col()
    thread_x          = tab.Int32Col()
    thread_y          = tab.Int32Col()
    threads           = tab.Int32Col()
    overlap           = tab.Int64Col()
    time              = tab.Float64Col()


# TODO: add these!!!
# git commit/hash
# version info
# frozen library data
# os?
class VersionInfo(tab.IsDescription):
    code_git           = tab.StringCol(64)
    nvidia             = tab.StringCol(64)
    cuda               = tab.StringCol(64)
    os                 = tab.StringCol(64)
    gcc_python         = tab.StringCol(64)

    
class MetaData(tab.IsDescription):
    index             = tab.Int32Col()    
    pos_x             = tab.Float64Col()
    pos_y             = tab.Float64Col()
    zoom              = tab.Float32Col()
    dimensions_x      = tab.Int32Col()
    dimensions_y      = tab.Int32Col()
    mode              = tab.UInt8Col()  # should be a string for easier comprehension. 
    iterations        = tab.Int32Col()

    versioninfo = VersionInfo()

    # TODO: what kind of attributes can we add here. when i tried to
    # add mode_identifier it balked when building the table
    
mode_identifier = {0:'write',1:'read+write',2:'no_rw', 3:'atomicAdd+write',4:'Overlap'}

class PairedDataStorage(object):
    @classmethod
    def cudaCollect(cls, kwdD):
        pass

    @classmethod
    def init(cls):
        pass

    @classmethod
    def alreadyRan(cls, kwdD):
        pass

    @classmethod
    def extractCols(cls):
        pass

    @classmethod
    def extractMetaData(cls):
        pass

def _get_new_meta_index():
    meta = _data_file.root.TimingData.meta
    if meta.nrows == 0:
        return 1
    # TODO: is there a get column operation???
    indexL = [row['index'] for row in meta]
    indexL.sort()
    newIndex = indexL[-1]+1
    return newIndex
    
    
# TODO: we should supply data as dicts, or allied, so we can automagically populate rows
def cudaCollect(position,zoom,dimensions,execData,mode=0,iterations=100):
    """
    Run callCUDA over a range of block and thread shapes and sizes, and collect data on time spent. 
    """
    # keep here so we only compile as needed
    from call_utils import callCUDA
    
    global _data_file
    data = _data_file.root.TimingData.data
    meta = _data_file.root.TimingData.meta

    index = alreadyRan(position, dimensions, zoom, mode)
    if not index:
        # need a new entry
        index = _get_new_meta_index()
        meta.row['pos_x'] = position[0]
        meta.row['pos_y'] = position[1]
        meta.row['dimensions_x'] = dimensions[0]
        meta.row['dimensions_y'] = dimensions[1]
        meta.row['zoom'] = zoom
        meta.row['mode'] = mode
        meta.row['iterations'] = iterations
        meta.row.append()
        meta.flush()

    for block in execData['blocks']:
        for thread in execData['threads']:
            # TODO: check to see if we have done this combo already for the given metadata      
            try:
                name=str(block)+", "+str(thread)
                result,time,block_dim,thread_dim = callCUDA(position,zoom,dimensions,name,
                                                            block=block,thread=thread,save=False,mode=mode)
            except ValueError:
                continue

            print "GOOD \t"+str(block)+", "+str(thread)+": "+str(time)
            
            data.row['time'] = time
            data.row['index'] = len(data)
            data.row['block_x'] = block_dim[0]
            data.row['block_y'] = block_dim[1]
            data.row['blocks'] = block
            data.row['thread_x'] = thread_dim[0]
            data.row['thread_y'] = thread_dim[1]
            data.row['threads'] = thread
            if mode==4:     
                data.row['overlap'] = np.sum(result)-(result.shape[0]*result.shape[1])
            else:
                data.row['overlap']=0
            data.row['metaIndexFK'] = index

            data.row.append()
            data.flush()
            
    return index

def alreadyRan(position,dimensions,zoom,mode):
    global _data_file

    mtable = _data_file.root.TimingData.meta
    
    tupL = (dimensions[0], dimensions[1], mode, position[0], position[1], zoom)
    # NOTE!!!!: the condx string very sensitive. when i replaced a '&' with 'and', all rows were taken!!!
    condS = "(dimensions_x==%d) & (dimensions_y==%d) & (mode == %d) & (pos_x == %f) & (pos_y == %f) & (zoom == %f)"%tupL
    print condS
    rowL = [row for row in mtable.where(condS)]
    if len(rowL) == 0:
        return None
    if 0:
        row = rowL[0]
        print row['dimensions_x'], row['dimensions_y'], row['mode'],  row['pos_x'], row['pos_y'],  row['zoom']
        
    return rowL[0]['index']

def extractCols(nExec):
    global _data_file

    meta = _data_file.root.TimingData.meta
    data = _data_file.root.TimingData.data

    
    rowL = [row for row in  meta.where('index == %d'%nExec)]
    if len(rowL)==0:
        raise ValueError('no such data items: %d'%nExec)
    
    # grab first row data
    row=rowL[0]

    blocks  = []
    times   = []
    threads = []
    overlap = []
    
    for execution in data.where('metaIndexFK == %d'%nExec):
        blocks.append(execution['blocks'])
        threads.append(execution['threads'])
        times.append(execution['time'])
        overlap.append(execution['overlap'])

    # TODO: should accept, and return dicts!
    return (blocks,times,threads,row['zoom'],row['mode'],
            (row['dimensions_x'],row['dimensions_y']), row['iterations'],nExec,overlap)

def extractMetaData():
    global _data_file    

    metaD={}
    colnames = _data_file.root.TimingData.meta.colnames
    for meta in _data_file.root.TimingData.meta:
        fk = meta['index']
        metaD[fk] = { key:meta[key] for key in colnames}

    return metaD
    
def init(filename='fractalData.h5'):
    global _data_file

    if _data_file == None:
        _data_file = tab.openFile(filename,mode='a',title="Fractal timing data")

    if not ("/TimingData" in _data_file):
        print 'Creating top level folder'
        grp = _data_file.create_group("/", 'TimingData', 'Timing data for parallel execution')

    if not ("/TimingData/data" in _data_file.root):
        print 'Creating data folder'        
        _data_file.create_table('/TimingData', 'data', ExecutionData, 'Fractal timing data')
        
    if not ("/TimingData/meta" in _data_file.root):
        print 'Creating meta folder'
        _data_file.create_table('/TimingData', 'meta', MetaData, 'Fractal timing meta data')

def build_table(tableN, populate=True):
    global _data_file
    
    init(tableN)
    
    if not populate: return 
        
    theMetaTable = _data_file.root.TimingData.meta
    theDataTable = _data_file.root.TimingData.data   
    row = theMetaTable.row
    drow = theDataTable.row
    for ii in range(3):
        i = _get_new_meta_index()
        row['dimensions_x'] = 20+i
        row['dimensions_y'] = 30+i
        row['iterations'] = 100+i
        row['mode'] = 2+i
        row['pos_x']= 2.1
        row['pos_y']= 1.7
        row['zoom']=900.
        row['versioninfo/code_git'] = 'brand new'
        row['versioninfo/nvidia'] = 'brand new'
        row['versioninfo/cuda'] = 'brand new'
        row['versioninfo/os'] = 'brand new'
        row['versioninfo/gcc_python'] = 'brand new'              
        row['index'] = i
        row.append()
        theMetaTable.flush()
        for j in range(3):
            drow['metaIndexFK'] = i
            drow['block_x'] = 1000+j+i
            drow['block_y'] = 1000+j+i
            drow['blocks'] = 1000+j+i
            drow['thread_x'] = 1000+j+i
            drow['thread_y'] = 1000+j+i
            drow['threads'] = 1000+j+i
            drow['time'] = 2.1+float(j+i)/3
            drow.append()
            theDataTable.flush()        
    

if __name__ == '__main__':

    if 1:
        build_table('fractalData.h5', populate=True)
        
    if 0:
        init()
        res= alreadyRan( (2.1,1.7), (20, 30), 900., 2)
        if res: print extractCols(res)

        print  extractCols(0)
        print  extractCols(1)
        print extractMetaData()

    if 1:
        init()
        print _get_new_meta_index()
