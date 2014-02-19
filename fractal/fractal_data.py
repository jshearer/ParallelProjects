import tables as tab
from call_utils import callCUDA

data_file = None

class Execution(tab.IsDescription):
	index		  = tab.Int32Col()
	block_x		  = tab.Int32Col()
	block_y		  = tab.Int32Col()
	thread_x	  = tab.Int32Col()
	thread_y	  = tab.Int32Col()
	time		  = tab.Float64Col()

class MetaData(tab.IsDescription):
	pos_x 		  = tab.Float64Col()
	pos_y 		  = tab.Float64Col()
	zoom  		  = tab.Float32Col()
	dimensions_x  = tab.Int32Col()
	dimensions_y  = tab.Int32Col()
	mode		  = tab.UInt8Col()

def cudaCollect(position,zoom,dimensions,blockData,threadData,mode=0):
	#First run, block checking only
	init()
	global data_file

	nExec = len(data_file.listNodes(getGroup()))

	grp = data_file.createGroup(getGroup(),str(nExec), "Execution number "+str(nExec+1))
	meta = data_file.createTable(grp,"meta",MetaData,"Metadata")

	meta.row['pos_x'] = position[0]
	meta.row['pos_y'] = position[1]
	meta.row['dimensions_x'] = dimensions[0]
	meta.row['dimensions_y'] = dimensions[1]
	meta.row['zoom'] = zoom
	meta.row['mode'] = mode
	meta.row.append()
	meta.flush()

	data = data_file.createTable(grp,"data",Execution,"Real data")

	#[0] = start,
	#[1] = end,
	#[2] = stride
	for x in blockData[0]:
		for y in blockData[1]:

			block = (x,y,1)

			for t_x in threadData[0]:
				for t_y in threadData[1]:
					
					thread = (t_x,t_y,1)
					time = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=False,mode=mode)
					data.row['time'] = time
					data.row['index'] = len(data)
					data.row['block_x'] = x
					data.row['block_y'] = y
					data.row['thread_x'] = t_x
					data.row['thread_y'] = t_y
					data.row.append()
					data.flush()
					print "\t"+str(block)+", "+str(thread)+": "+str(time)
	return nExec

def alreadyRan(position,dimensions,zoom,mode):
	init()
	global data_file

	for node in data_file.walkNodes('/execSets'):
		pos_x = (node.meta['pos_x']==position[0])
		pos_y = (node.meta['pos_y']==position[1])
		pos = pos_x and pos_y

		dim_x = (node.meta['dimensions_x']==dimensions[0])
		dim_y = (node.meta['dimensions_y']==dimensions[1])
		dim = dim_x and dim_y

		zoom_check = (node.meta['zoom']==zoom)
		mode_check = (node.meta['mode']==mode)

		if pos and dim and zoom_check and mode_check:
			return true

	return false		

def extractCols(nExec):
	init()
	global data_file

	cores   = []
	times   = []
	threads = []

	data = data_file.getNode("/execSets/"+str(nExec)+"/data")

	for execution in data:
		cores.append(execution['block_x']*execution['block_y'])
		threads.append(execution['thread_x']*execution['thread_y'])
		times.append(execution['time'])

	return cores,times,threads


def getGroup():
	init()
	global data_file

	return data_file.root.execSets

def init():
	global data_file

	if data_file == None:
		filename = "fractalData.h5"
		data_file = tab.openFile(filename,mode='a',title="Fractal timing data")
		if not ("/execSets" in data_file):
			data_file.createGroup("/","execSets","Sets of execution with varying position,zoom,dimensions,blockData, or threadData")