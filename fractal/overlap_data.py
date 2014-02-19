from fractal_data import *

def overlapCollect(position,zoom,dimensions,blockData,threadData,iterations=100):
	from call_utils import callCUDA
	#Only compile the function when we need to
	
	mode = 4 #overlap mode

	class Execution(tab.IsDescription):
		index		  = tab.Int32Col()
		block_x		  = tab.Int32Col()
		block_y		  = tab.Int32Col()
		thread_x	  = tab.Int32Col()
		thread_y	  = tab.Int32Col()
		overlap		  = tab.Int64Col()
		time		  = tab.Float64Col()


	class MetaData(tab.IsDescription):
		pos_x 		  = tab.Float64Col()
		pos_y 		  = tab.Float64Col()
		zoom  		  = tab.Float32Col()
		dimensions_x  = tab.Int32Col()
		dimensions_y  = tab.Int32Col()
		mode		  = tab.UInt8Col()
		iterations    = tab.Int32Col()

	init()
	global data_file

	nExec = len(data_file.listNodes(getGroup()))

	grp = data_file.createGroup(getGroup(),"Overlap "+str(nExec), "Overlap run number "+str(nExec+1))
	meta = data_file.createTable(grp,"meta",MetaData,"Metadata")

	meta.row['pos_x'] = position[0]
	meta.row['pos_y'] = position[1]
	meta.row['dimensions_x'] = dimensions[0]
	meta.row['dimensions_y'] = dimensions[1]
	meta.row['zoom'] = zoom
	meta.row['mode'] = mode
	meta.row['iterations'] = iterations

	meta.row.append()
	meta.flush()

	data = data_file.createTable(grp,"data",Execution,"Real data")

	for x in blockData[0]:
		for y in blockData[1]:

			block = (x,y,1)

			for t_x in threadData[0]:
				for t_y in threadData[1]:
					
					thread = (t_x,t_y,1)
					result,time = callCUDA(position,zoom,dimensions,str(block)+", "+str(thread),block=block,thread=thread,save=False,mode=mode,returnResult=True)

					data.row['time'] = time
					data.row['overlap'] = calculateOverlap(result)
					data.row['index'] = len(data)
					data.row['block_x'] = x
					data.row['block_y'] = y
					data.row['thread_x'] = t_x
					data.row['thread_y'] = t_y
					data.row.append()
					data.flush()
					print "\t"+str(block)+", "+str(thread)+": "+str(time)

def calculateOverlap(result):
	overlap = 0

	for x in range(0,result.shape[0]):
		for y in range(0,result.shape[1]):
			overlap = overlap + result[x][y]

	return overlap