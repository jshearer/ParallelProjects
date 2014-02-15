import numpy
import pdb
from PIL import Image

from multiprocessing import Pool

pool = Pool(10)

def SaveToPngThread(result,name,gradient,silent=False):
	pool.apply_async(SaveToPng,[result,name,gradient,True])

def SaveToPng(result,name,gradient,silent=False):
	lookup = {}

	gradient = remapGradient(gradient,0,256)

	for i in range(0,256):
		lookup[i] = getGradCol(i,gradient)

	if not silent: 
		print("Resizing result to be in range 0-255")

	result = (result.astype(numpy.float32)*(255.0/result.max())).astype(numpy.uint8)
	if not silent:
		print("Done resizing. Now generating image array.")

	result = result.reshape((result.shape[1],result.shape[0]))
	if not silent:
		print("Done generating image array. Writing image file.")

	#colors!
	#First, make a WxHx3 array
	colorArr = numpy.zeros([result.shape[0],result.shape[1],3],dtype=numpy.uint8)

	x=y=0

	while y < len(result):
		while x < len(result[y]):
			a = lookup[result[y,x]]
			colorArr[y,x] = a
			#print str(colorArr[y,x])+", "+str((x,y))
			x = x + 1
		y = y + 1
		pct = (float(y)/len(result))*50
		if not silent:
			print "\r["+("#"*int(pct))+(" "*(50-int(pct)))+"] "+str(pct*2)+"% ("+str(y)+"/"+str(len(result))+")",
		x = 0

	#colors!
	Image.fromarray(colorArr).convert("RGB").save(name+".png")
	if not silent:
		print("Image file written.")

def renderGradient(gradient,name):
	keys = sorted(gradient.keys())

	scale = 1
	
	leng = 0
	oldi = 0
	for k in keys:
		leng = (leng + (k-oldi)*scale)

	leng = int(leng)

	arr = numpy.zeros([leng/2,20,3],dtype=numpy.uint8)

	for i in range(leng/2):
		for j in range(20):
			arr[i,j] = getGradCol(float(i)/scale,gradient)

	Image.fromarray(arr).convert("RGB").save(name+".png")

def remapGradient(gradient,newlower,newupper):
	newgrad = {}

	oldlower = min(gradient.keys())
	oldupper = max(gradient.keys())

	for oldId in gradient.keys():
		newid = remap(oldId,oldlower,oldupper,newlower,newupper)
		newgrad[newid] = gradient[oldId]
	return newgrad

def remap(X,A,B,C,D):
	#If your number X falls between A and B, and you would like Y to fall between C and D, you can apply the following linear transform:

	return (float(X)-A)/(B-A) * (float(D)-C) + float(C)



#colors = {
#			'default' : 
#			{
#		  		4:(0,0,0),
#		  		3:(128,255,20),
#		  		2:(64,255,128),
#		  		1:(255,64,255),
#		  		0:(182,64,200)
#		  	}
#		 }

colors = {
			'default' : 
			{
		  		0:(0,0,0),
		  		20:(128,64,128),
		  		100:(64,128,64),
		  		75:(30,0,200)
		  	},
		  	'reds' :
		  	{
		  		0:(0,0,0),
		  		5:(255,128,10),
		  		30:(200,10,180),
		  		75:(75,0,200)
		  	}
		 }

def getGradCol(number,colors):	
	start = -999999999999999
	end = 999999999999999
	for found in colors.keys():
		if found <= number:
			if found > start:
				start = found
		else:
			if found < end:
				end = found

	if start == -999999999999999:
		start = 0
	if end == 999999999999999:
		end = 0

	dist = end-start
	pctage = remap(number,start,end,0,1)

	def lerp(a,b,t):
		return a+(b-a)*t

	endcol = (lerp(colors[start][0],colors[end][0],pctage),
					  lerp(colors[start][1],colors[end][1],pctage),
					  lerp(colors[start][2],colors[end][2],pctage))
	return endcol
