from pylab import *

from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
## for Palatino and other serif fonts use:
#rc('font',**{'family':'serif','serif':['Palatino']})
rc('text', usetex=True)

def _isocline_plot(x,y, func, curveL, theTitle, xlab, ylab):
    X, Y = meshgrid(x, y)
    deg = func(X,Y)
    QP = quiver(X,Y,cos(deg),sin(deg))
    xlabel(xlab)
    ylabel(ylab)

    for ys in curveL:
        plot(x, ys)

    title(theTitle)
    show()
    
    

def isocline_exponential():
    xmax = 2.0
    xmin = 0.0
    D = 20
    ymax = xmax/2
    ymin = -ymax
    x = linspace(xmin, xmax, D+1)
    y = linspace(ymin, ymax, 2*D+1)
    yexp = exp(-x)
    y1exp = 0.5*exp(-x)
    ymexp = -exp(-x)
    ym1exp = -0.5*exp(-x)

    def slope(x,y):
        return arctan( -y )
    
    _isocline_plot(x,y, slope, (yexp, y1exp, ymexp, ym1exp), r'$\frac{dy}{dx} = -y$', 'x', 'y')


def isocline_oscillator():
    xmax = 2.0
    xmin = -xmax
    ymax = xmax
    ymin = -ymax
    D = 20
    x = linspace(xmin, xmax, D+1)
    y = linspace(ymin, ymax, 2*D+1)

    def slope(x,y):
        return arctan2( -x, y )
    
    _isocline_plot(x,y, slope, (), r'$\frac{dv}{dx} = -x/v$', 'x', 'v')
    
if __name__ == '__main__':
    if 0: isocline_exponential()
    if 1: isocline_oscillator()
