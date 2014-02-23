import numpy as np

def smallestPerim(val):
	factors = [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]

	best = None #1*a

	for factor in factors:
		if best == None:
			best = factor
		elif (factor[0]+factor[1])<(best[0]+best[1]):
			best = factor

	return np.array(best, dtype=np.int32)

def calcParameters(blocks,threads,dimensions,silent=False):
	bestBlocks = smallestPerim(blocks)
	bestThreads = smallestPerim(threads)[::-1]
	dimensions = np.array(dimensions)

	px_per_block = dimensions/bestBlocks
	px_per_thread = (dimensions/bestBlocks)/bestThreads
	px_total = (px_per_thread.prod()*bestThreads.prod()*bestBlocks.prod())

	too_much = False #too many blocks or threads
	wrong = False	 #bad configuration of blocks and threads
	if not silent:
		print "Block dimensions are "+str(bestBlocks)+", each block contains "+str(px_per_block)+" pixels."
		print "Thread dimensions per block are "+str(bestThreads)
		print "Each thread will calculate "+str(px_per_thread) + " pixels."
		print "Total number of pixels calculated: "+str(px_total)
		print "Total number of threads to be launched "+str(bestBlocks*bestThreads)+" = "+str((bestBlocks*bestThreads).prod())+" total threads."

	if (1-(px_per_block!=0).astype(np.int)).sum()>=1 or (1-(px_per_thread!=0).astype(np.int)).sum()>=1:
		too_much = "WARNING: Current configuration of blocks and threads result in pixels/thread of less than 1. Revise downwards."
		if not silent:
			print too_much
	if px_total < dimensions.prod():
		wrong = "WARNING: Current configuration of blocks and threads results in "+str(px_total)+"/"+str(dimensions.prod())+" pixels calculated."
		if not silent:
			print wrong
	
	return bestBlocks,bestThreads,px_per_block,px_per_thread,too_much,wrong


