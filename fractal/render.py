import numpy
import pdb
from PIL import Image

def SaveToPng(result,name,gradient,silent=False):
	lookup = {}

	gradient = remapGradient(gradient,0,255)

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
		x = 0

	#colors!
	Image.fromarray(colorArr).convert("RGB").save(name+".png")
	if not silent:
		print("Image file written.")

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

colors = {
			'default' : 
			{
		  		0:(0,0,0),
		  		5:(128,255,20),
		  		10:(64,255,128),
		  		15:(0,128,255),
		  		20:(182,64,200)
		  	}
		 }

def getGradCol(number,colors):	
	start = min(colors.keys())
	end = max(colors.keys())

	dist = end-start
	pctage = remap(number,start,end,0,1)

	def lerp(a,b,t):
		return a+(b-a)*t

	endcol = (lerp(colors[start][0],colors[end][0],pctage),
					  lerp(colors[start][1],colors[end][1],pctage),
					  lerp(colors[start][2],colors[end][2],pctage))
	return endcol
