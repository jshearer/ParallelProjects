import Vector
import math

def CallIfExists(func,*args):
	if type(func) is types.FunctionType or type(func) is types.MethodType:
		return func(*args)

def remap(thing,lowerbound,newrange,oldrange):
	return(((thing-lowerbound) * newrange) / oldrange)

def getGradCol(number,colors):	
	start = -99999999999999
	end = 999999999999999
	for found in colors.keys():
		if found <= number:
			if found > start:
				start = found
		else:
			if found < end:
				end = found
	dist = end-start
	pctage = remap(number,float(start),float(1),dist)

	def lerp(a,b,t):
		return a+(b-a)*t

	endcol = (lerp(colors[start][0],colors[end][0],pctage),
					  lerp(colors[start][1],colors[end][1],pctage),
					  lerp(colors[start][2],colors[end][2],pctage),
					  lerp(colors[start][3],colors[end][3],pctage))
	return endcol

