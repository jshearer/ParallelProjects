import numpy
#Slope function:
def slope(y):
	return (y*2.0)

delta_x = 0.01
start = (0,1) #starting condition

def regularIntegrator(func,step,initial,distance,plot):
	values = []
	y = initial[1]
	x = initial[0]

	while(x<distance):
		m = slope(y) #Slope at previous y value
		y = y + (m*step) #Move y by slope.
		values.append((x,y))
		print (x,y)

		x = x + step

	values = numpy.array(values,dtype=numpy.float)
	
	if plot:
		xs = values[:,0]
		ys = values[:,1]
		from pylab import plot,show
		plot(xs,ys)
		show()

	return values

