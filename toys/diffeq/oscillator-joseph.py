#   |
#   |
#   |
# v |
#   |
#   |
#   |
#   |_____________________
#   			  x
		
#dx/dt=v
#dv/dt=-x
#x_0 = 1
#v_0 = 0


#n: 0
#(1,0)

#n: 1
#(1,-1)

#n: 2
#(0, -2)

#n: 3
#(-2, -2)

#n: 4
#(-4, 0)

#n: 5

from pylab import plot,show
from numpy import array, pi

def motion():
    x = float(1)
    v = float(0)
    t = float(0)
    dt = 0.2
    max_t = float(10*2*pi)
    
    results = [(t,x,v), ]
    while t<max_t:
        
        dx = dt*v
        x = x + dx
        dv = -(x)*dt
        
        v = v + dv
        t = t + dt
        results.append((t,x,v))


    return array(results)
    

res = motion()
plot(res[:, 1], res[:, 2])
        
show()

