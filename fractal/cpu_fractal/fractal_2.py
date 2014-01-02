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

def gen(position,zoom,dimensions,name,scale=1,squaresize=50,processes=4,silent=False, iterations=100):
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
				50:(255,100,255,255),
				100:(255,100,200,255),
				99999999:(255,255,255,255)}
	
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

	result = numpy.rot90(result)

	Image.fromarray(result,"RGB").save(name)


parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter, description="A program to generate a fractal image. \nIn the case of 2-argument flags (such as --position), the arguments should be space-seperated. As in: --position 10 10", epilog='''Example usage:
	./fractal.py -p 0 0 -z 50 -d 200 200 -- This will render a 200x200 image, with the fractal centered, and zoomed to 50.

	./fractal.py -p 100 0  -d 200 200 -z 50 -s 2 -- This will render a 200x200 image, zoomed to 50, and then scaled up 2x. 
	Without the -m argument this effectively means it will zoom in on the center of the image 2x.

	If you have a case such as this: 
	./fractal_2.py -p 0 -29.65 -d 400 1500 -z 20 -s 230 fractal.png
	And you want to render it at 2x resolution, you can't simply add \'-s 2 -m\' because you already have scale defined and by adding -m, 
	you would be rendering it at 230x resolution.
	So this is where the --calculate option is useful. This will calculate the modied position and zoom arguments, 
	and print out a command that will output the same as your current setup, without the scale argument like so:
	./fractal_2.py -p 0 -6819 -d 400 1500 -z 4600.0
	You can then add the scale and -m arguments as you like to scale your render by whatever you want.''', prog="fractal")
parser.add_argument('--position','-p', metavar="pos", nargs=2, type=float, required=True, help="The offset of the rendered fractal. If set to 0 0, the fractal will be centered in the output image.")
parser.add_argument('--dimensions','-d', metavar="dim", nargs=2, type=int, required=True, help="The rendered image dimensions, in pixels. This may be modified by the use of the scale argument.")
parser.add_argument('--zoom','-z', type=float, required=True, help="The zoom of the fractal. This may be modified by the scale argument.")
parser.add_argument('--scale','-s', type=float, default=1, help="The scale multiplier. 1=default, use provided position and zoom. 2=2x zoom and position. This can be used to make a thumbnail image before running the full render, to see if your coordinates are correct.\n")
parser.add_argument('--processes','-procs', type=int, default=4, help="The number of processes to use for multi-process rendering. This is usually the number of CPU cores you have. Defaults to 4")
parser.add_argument('--iterations','-i', dest="iters", default=50, type=int, help="The maximum number of iterations the generator will go through, per pixel. The higher this is, the more accurate the fractal will be, and the slower the generation will be. It will be especially slow in places where the pixel is inside the shape, in which case the loop would go on for ever without a limit.")
parser.add_argument('--calculate', action="store_true", dest="calc", help="If this flag is set, no fractal will be rendered. Instead, the parameters will be outputed to stdout, calculated such that they can be applied with scale at 1. ")
parser.add_argument('--blocksize', dest="bsize", type=int, default=50, help="Set the block generation size. Default 50.")
parser.add_argument('name', help="The name of the output file. This can have any image extension supported by PIL.")
args = vars(parser.parse_args())

if args['calc']:
	position = args['position']
	zoom = args['zoom']
	scale = 1.0/args['scale']
	dimensions = args['dimensions']
	
	dimensions.reverse()
	position.reverse()
	
	zoom = zoom/scale
	dimensions = (int(dimensions[0]/scale),int(dimensions[1]/scale))

	position = Vector(position[0]/scale,position[1]/scale)
	position = (int(position.x),int(position.y))
	print("./fractal.py -p "+str(position[1])+" "+str(position[0])+" -d "+str(dimensions[1])+" "+str(dimensions[0])+" -z "+str(zoom))
else:
	try:
		gen(args['position'],args['zoom'],args['dimensions'],args['name'], processes=args['processes'], scale=args['scale'], iterations=args['iters'], squaresize=args['bsize'])
	except Exception as exc:
		os.system("setterm -cursor on")
		traceback.print_exc(file=sys.stdout)