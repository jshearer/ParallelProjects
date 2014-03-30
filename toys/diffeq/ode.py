#Plotting ODEs, Isoclines using Python
#x,y = var("x y")
#eq = y^3-3*y-x
#p = implicit_plot(eq==0,(x,-4,4),(y,-4,4))
#p += plot_slope_field(eq, (x,-4,4),(y,-4,4), headlength=1e-8)
#p.show(aspect_ratio=1)

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

       
from pylab import *
xmax = 2.0
xmin = 0.0
D = 20
ymax = xmax/2
ymin = -ymax
x = linspace(xmin, xmax, D+1)
y = linspace(ymin, ymax, 2*D+1)
X, Y = meshgrid(x, y)
deg = arctan(-Y)
QP = quiver(X,Y,cos(deg),sin(deg))
xlabel('$x$')
ylabel('$y$')
title(r'$\frac{dy}{dx} = -y$')
yexp = exp(-x)
y1exp = 0.5*exp(-x)
ymexp = -exp(-x)
ym1exp = -0.5*exp(-x)

for ys in (yexp, y1exp, ymexp, ym1exp):
    plot(x, ys)
show()
