#!/usr/bin/python2
import os
from multiprocessing import Pool
import utils
from PIL import Image
import numpy
import argparse
from Vector import Vector
import sys
import time
import traceback

def genChunk(position,size,zoom,iterations):
	try:
		zoom = float(zoom)
		
		if type(size) is not int:
			result = numpy.zeros((size[0],size[1]))
			dimensions = (size[0],size[1])
		else:
			result = numpy.zeros((size,size))
			dimensions = (size,size)
		
		for x in range(dimensions[0]):
			for y in range(dimensions[1]):
				t_x = (x+position[0])/float(zoom)
				t_y = (y+position[1])/float(zoom)
				comp = complex(t_x,t_y)
				z = comp
				for i in range(1,iterations+1):
					z = z*z + comp
					if((z.real*z.real + z.imag*z.imag)>4):
						result[x,y] = i
						break

		return (result,position,size)
	except Exception as e:
		print("ERROR::::::::: "+str(e))
		traceback.print_exc(file=sys.stdout)

#position,zoom,resolution,scale
#position is obvious
#zoom is the fractal's divisor
#dimensions is the size, as a 2-tuple, of the output image
#scale is a number that allows you to reduce the zoom and dimensions simultaneously 
#in order to render a lower-res version of the same fractal

def gen(position,zoom,dimensions,scale=1,squaresize=50,processes=4,silent=False, iterations=100):
	scale = 1.0/scale
	procPool = Pool(processes=processes)

	#to correct, not sure why the problem occurs in the first place.
	position.reverse()
	dimensions.reverse()
	zoom = zoom/scale
	dimensions = (int(dimensions[0]/scale),int(dimensions[1]/scale))

	position = Vector(position[0]/scale,position[1]/scale)
	position = position - (Vector(dimensions[0],dimensions[1])/2)
	position = (int(position.x),int(position.y))

	startime = time.time()

	grad = {	0:(0,0,0,255), #Color data
				iterations:(255,255,255,255),
				99999999999:(0,0,0,255)}
	
	result = numpy.zeros((dimensions[0],dimensions[1],3),dtype=numpy.uint8)
	
	lookup = []
	for i in range(0,255):
		lookup.append(utils.getGradCol(i,grad)) #precache gradient values in order to avoid repetitive function calls

	num_completed = {'done':0,'total':0} #This hack is nessecary because python 2 doesn't have the nonlocal keyword, so callback() couldn't modify num_completed if it were just an int.
						#Apparently, single element arrays work as a replacement...

	def callback(data): #should find a better numpy-way to do this...

		#data is the return of genChunk, which is:
		#return (result,position,size)

		pos = (data[1][0]-position[0],data[1][1]-position[1])
		size = data[2]
		subgrid = data[0]

		num_completed['done'] = num_completed['done'] + 1

		if not silent: #report progress to stdout, using FancyProgressBar(tm)
			pct = int((num_completed['done']/float(num_completed['total']))*100.0*0.3)
			sys.stdout.write("\r["+("#"*pct)+(" "*(30-pct))+"] "+str(int((num_completed['done']/float(num_completed['total']))*100.0))+"% <"+str(int(time.time()-startime))+"s> ("+str(num_completed['done'])+")")

		try:
			for x in range(pos[0],size[0]+pos[0]):
				for y in range(pos[1],size[1]+pos[1]):
					clamped = min(int(subgrid[x-pos[0],y-pos[1]]),100)
					result[x,y,0] = lookup[clamped][0] #R
					result[x,y,1] = lookup[clamped][1] #G
					result[x,y,2] = lookup[clamped][2] #B
		except Exception as e:
			if not silent:
				print("Error:::::: "+str(e))


	for x in xrange(position[0],dimensions[0]+position[0],squaresize):
		for y in xrange(position[1],dimensions[1]+position[1],squaresize):
			procPool.apply_async(genChunk,[(x,y),(squaresize,squaresize),zoom,iterations],callback=callback,)
			num_completed['total'] += 1

	procPool.close()
	procPool.join()
	return result

def savepng(result,name,silent=False):
	result = numpy.rot90(result)

	Image.fromarray(result,"RGB").save(name+".png")