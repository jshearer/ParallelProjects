import numpy as np

def getFactors(val):
	return np.array([(i, val / i) for i in range(1, int(val+1)) if val % i == 0])

def smallestPerim(val):
	factors = [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]

	best = None #1*a

	for factor in factors:
		if best == None:
			best = factor
		elif (factor[0]+factor[1])<(best[0]+best[1]):
			best = factor

	return np.array(best, dtype=np.int32)

def genParameters(blocks,threads,dimensions,silent=False):
	dimensions = np.array(dimensions)

	block_factors = getFactors(blocks)
	thread_factors = getFactors(threads)

	bestBlocks = None
	bestThreads = None

	for b_factor in block_factors:
		for t_factor in thread_factors:
			err1,err2 = checkParameters(b_factor,t_factor,dimensions,silent=True)
			if not err1 and not err2:
				bestBlocks = b_factor
				bestThreads = t_factor

	if bestBlocks == None:
		raise ValueError("Combination of blocks, threads, and dimensions does not result in any block, thread dimensions that will reach every pixel. Reconsider.")

	px_per_block = dimensions/bestBlocks
	px_per_thread = (dimensions/bestBlocks)/bestThreads
	px_total = (px_per_thread.prod()*bestThreads.prod()*bestBlocks.prod())
	
	return (bestBlocks,bestThreads,px_per_block,px_per_thread)

def checkParameters(bestBlocks,bestThreads,dimensions,silent=False):
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

	return too_much,wrong


def bestPerim(val,ratio): #Was an idea, doesn't quite fit yet.
	factors = [(i, val / i) for i in range(1, int(val)+1) if val % i == 0]

	best = None
	bestRatio = None

	for factor in factors:
		if best == None:
			best = factor
			bestRatio = (float(best[0])/float(best[1]))
		elif abs((float(factor[0])/float(factor[1]))-ratio) < abs(bestRatio-ratio):
			best = factor
			bestRatio = float(factor[0])/float(factor[1])

	return np.array(best)