import numpy as np

# we should have a modular way of characterizing GPU resources.
K20C_BLOCKS = 2048
K20C_THREADS = 1024

def getFactors(val):
	return np.array([(i, val / i) for i in range(1, int(val+1)) if val % i == 0])

def genParameters(blocks,threads,dimensions,silent=False):
    # should check that blocks <= blocks available on GPU
    # should check that threads <= threads available on block on GPU
    if blocks > K20C_BLOCKS:
        raise ValueError("GPU does not have enough blocks")
    if threads > K20C_THREADS:
         raise ValueError("GPU does not have enough threads per block")

	dimensions = np.array(dimensions)
	px_total_needed =  dimensions.prod()

    px_per_thread = px_total_needed / (blocks*threads)
    px_total = px_per_thread  * blocks * threads
	if px_total != px_total_needed:
        if not silent:
            print "WARNING: Current configuration of blocks and threads results in ",
            print str(px_total) + "/" + str(px_total_needed) + " pixels calculated."
        raise ValueError("Total pixels does not divide evenly across total threads.")

    # we might want to try a shape in the middle of each first
	for bestBlocks in getFactors(blocks):
        px_per_block = dimensions/bestBlocks
        if (1-(px_per_block!=0).astype(np.int)).sum()>=1:
            continue
		for bestThreads in getFactors(threads):
            px_per_thread = px_per_block/bestThreads
            if (1-(px_per_thread!=0).astype(np.int)).sum() < 1: # ==0???, anyway, we found a case, use it!
                if not silent:
                    print "Block dimensions are " + str(bestBlocks)+", each block contains " + str(px_per_block)+" pixels."
                    print "Thread dimensions per block are " + str(bestThreads)
                    print "Each thread will calculate " + str(px_per_thread) + " pixels."
                    print "Total number of pixels calculated: " + str(px_total)
                    print "Total number of threads to be launched " + str(bestBlocks*bestThreads),
                    print " = " + str((bestBlocks*bestThreads).prod()) + " total threads."
                    
                return (bestBlocks,bestThreads,px_per_block,px_per_thread)


    if not silent:
        print "Combination of blocks, threads, and dimensions does not result in any block, ",
        print "thread dimensions that will reach every pixel. Reconsider."
    raise ValueError("No valid parameters found")


def smallestPerim(val):
	factors = [(i, val / i) for i in range(1, int(val**0.5)+1) if val % i == 0]

	best = None #1*a

	for factor in factors:
		if best == None:
			best = factor
		elif (factor[0]+factor[1])<(best[0]+best[1]):
			best = factor

	return np.array(best, dtype=np.int32)

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
