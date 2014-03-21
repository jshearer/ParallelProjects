import tables as tab
import numpy as np

NoSuchNodeError = tab.exceptions.NoSuchNodeError

_VERBOSE = False

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

# godzilla@K20cGPU:~$ cat /proc/driver/nvidia/gpus/0/information 
# Model: 		 Tesla K20c
# godzilla@TMA-1:~/Desktop/Science/ParallelProjects/fractal$ cat /proc/driver/nvidia/gpus/0/information 
# Model: 		 GeForce GT 555M
class VersionInfo(tab.IsDescription):
    os                 = tab.StringCol(64)
    gcc                = tab.StringCol(64)
    python             = tab.StringCol(64)
    nvidia_driver      = tab.StringCol(64)
    cuda_device        = tab.StringCol(32)
    cuda_api           = tab.StringCol(64)
    pycuda             = tab.StringCol(64)
    pytables           = tab.StringCol(64)
    code_git           = tab.StringCol(64)

class MetaData(tab.IsDescription):
    index             = tab.Int32Col()    
    pos_x             = tab.Float64Col()
    pos_y             = tab.Float64Col()
    zoom              = tab.Float32Col()
    dimensions_x      = tab.Int32Col()
    dimensions_y      = tab.Int32Col()
    mode              = tab.UInt8Col()  # TODO: should be a string for easier comprehension. 
    iterations        = tab.Int32Col()
    versioninfo = VersionInfo()

    # TODO: what kind of attributes can we add here. when i tried to
    # add mode_identifier it balked when building the table
    
mode_identifier = {0:'write',1:'read+write',2:'no_rw', 3:'atomicAdd+write',4:'Overlap'}

# TODO: how to make generic versions of cudaCollect, alreadyRan,
# extractCols, extractMetaData, etc. i.e., not dependent on any
# particular problem, like fractal, etc.
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
    # TODO: if index, check versions, and WARN if mismatch
    if not index:
        # need a new entry
        index = _get_new_meta_index()
        meta.row['index'] = index
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
            
            data.row['metaIndexFK'] = index
            data.row['time'] = time
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

            data.row.append()
            data.flush()
            
    return index

def alreadyRan(position,dimensions,zoom,mode):
    global _data_file

    mtable = _data_file.root.TimingData.meta
    
    tupL = (dimensions[0], dimensions[1], mode, position[0], position[1], zoom)
    # NOTE!!!!: the condx string very sensitive. when i replaced a '&' with 'and', all rows were taken!!!
    condS = '''(dimensions_x==%d) & (dimensions_y==%d) & (mode==%d) & (pos_x==%f) & (pos_y==%f) & (zoom==%f)'''%tupL
    if _VERBOSE: print condS
    rowL = [row['index'] for row in mtable.where(condS)]  
    if len(rowL) == 0:
        return None

    if len(rowL)>1:
        raise Exception('!!!!WARNING!!!!!: more than one row found in alreadyRan')
    
    return rowL[0]

def extractCols(index):
    global _data_file

    if _VERBOSE: print 'index = ', index

    meta = _data_file.root.TimingData.meta
    data = _data_file.root.TimingData.data

    rowL = [{'zoom':row['zoom'],'mode':row['mode'],'dimensions_x':row['dimensions_x'],'dimensions_y':row['dimensions_y'],'iterations':row['iterations'] }\
            for row in meta.where('index==%d'%index)] 
    if len(rowL)==0:
        raise ValueError('''no such metadata index: %d'''%index)
    
    if len(rowL)>1:
        raise Exception('!!!!!Warning!!!!!\n!!!!!more than one row of metadata with index %d'%index)

    # grab first row data
    row=rowL[0]
    if _VERBOSE: print 'mode = ', row['mode']

    blocks  = []
    times   = []
    threads = []
    overlap = []
    
    for execution in data.where('''metaIndexFK == %d'''%index):  
        blocks.append(execution['blocks'])
        threads.append(execution['threads'])
        times.append(execution['time'])
        overlap.append(execution['overlap'])

    if len(blocks)==0:
        raise ValueError("No data associated with index = %d"%index)

    # TODO: should accept, and return dicts!
    return (blocks,times,threads,row['zoom'],row['mode'],
            (row['dimensions_x'],row['dimensions_y']), row['iterations'],index,overlap)

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
        _data_file = tab.openFile(filename, mode='a',title="Fractal timing data")

    if not ("/TimingData" in _data_file):
        print 'Creating top level folder'
        grp = _data_file.createGroup("/", 'TimingData', 'Timing data for parallel execution')

    if not ("/TimingData/data" in _data_file.root):
        print 'Creating data table'        
        _data_file.createTable('/TimingData', 'data', ExecutionData, 'Fractal timing data')
        
    if not ("/TimingData/meta" in _data_file.root):
        print 'Creating meta table'
        _data_file.createTable('/TimingData', 'meta', MetaData, 'Fractal timing meta data')

def _build_table(tableN, populate=True):
    global _data_file
    
    init(filename=tableN)
    
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
    
def _delete_row(tableT, rowN):
    global _data_file
    
    if tableT == 'meta':
        theTab =_data_file.root.TimingData.meta
    else:
        theTab =_data_file.root.TimingData.data

    theTab.remove_row(0)
    theTab.flush()
    
def _test_where(index):
    global _data_file

    print 'searching for index = %d'%index
    
    mtable = _data_file.root.TimingData.meta
    data = _data_file.root.TimingData.data    

    for row in mtable.where('index==%d'%index):
        print '1: ', row['index']
    
    rowL = [row[:] for row in  mtable.where('index==%d'%index)]
    print '2: ', rowL[0][:]

    rowS = mtable.where('index==%d'%index)
    print '3: ', rowS.next().fetch_all_fields()

    for row in mtable.where('''(dimensions_x==21) & (dimensions_y==31) & (mode==3) & (pos_x==2.100000) & (pos_y==1.700000) & (zoom==900.000000)'''):
        print '4: ', row['index']

    rowL = [row for row in mtable.where('''(dimensions_x==21) & (dimensions_y==31) & (mode==3) & (pos_x==2.100000) & (pos_y==1.700000) & (zoom==900.000000)''')]
    print '5: ', rowL[0].fetch_all_fields()

    print
    
    for row in data.where('metaIndexFK==%d'%index):
        print '6: ', row['metaIndexFK']

    
if __name__ == '__main__':
    from argparse import ArgumentParser

    parser = ArgumentParser(description="Plot stored results for fractal simulations")
    parser.add_argument("--datasrc",action="store", dest="h5file",default="fractalData.h5",   help="hdf5 data file")
    args = parser.parse_args()

    init(filename=args.h5file)

    if 0:
        _delete_row('meta',0)

    if 0:
        _build_table('fractalData.h5', populate=True)

    if 0:
        print _get_new_meta_index()

    if 0:
        _test_where(2)

    if 1:
        md = extractMetaData()
        for index in md.keys():
            print  index, md[index]['index'], 
            print md[index]
            print extractCols(index)
        print
        
        for dim, mode in zip (  ( (21,31), (22,32), (23,33) , (24, 34)   ), (3, 4, 5, 6)  ):
            print 'seaching for ', dim, mode
            res= alreadyRan( (2.1, 1.7), dim, 900., mode)
            if res: 
                print 'res = ', res
                print extractCols(res)
            else:
                print 'no result for ', dim, mode

