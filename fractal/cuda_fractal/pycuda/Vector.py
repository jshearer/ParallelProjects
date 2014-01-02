import math
class Vector:
    'Represents a 2D vector.'
    def __init__(self, x = 0, y = 0):
        self.x = x
        self.y = y
        
    def __add__(self, val):
        return Vector( self[0] + val[0], self[1] + val[1] )
    
    def __sub__(self,val):
        return Vector( self[0] - val[0], self[1] - val[1] )
    
    def __iadd__(self, val):
        self.x = val[0] + self.x
        self.y = val[1] + self.y
        return self
        
    def __isub__(self, val):
        self.x = self.x - val[0]
        self.y = self.y - val[1]
        return self
    
    def __div__(self, val):
        return Vector( self[0] / val, self[1] / val )
    
    def __mul__(self, val):
        return Vector( self[0] * val, self[1] * val )
    
    def __idiv__(self, val):
        self[0] = self[0] / val
        self[1] = self[1] / val
        return self
        
    def __imul__(self, val):
        self[0] = self[0] * val
        self[1] = self[1] * val
        return self
                
    def __getitem__(self, key):
        if( key == 0):
            return self.x
        elif( key == 1):
            return self.y
        else:
            raise Exception("Invalid key to Vector")
        
    def __setitem__(self, key, value):
        if( key == 0):
            self.x = value
        elif( key == 1):
            self.y = value
        else:
            raise Exception("Invalid key to Vector")
        
    def asint(self):
        return Vector(self.x,self.y)
        
    def __str__(self):
        return "(" + str(self.x) + "," + str(self.y) + ")"

    @staticmethod    
    def DistanceSqrd( Vector1, Vector2 ):
        'Returns the distance between two Vectors squared. Marginally faster than Distance()'
        return ( (Vector1[0]-Vector2[0])**2 + (Vector1[1]-Vector2[1])**2)
    
    @staticmethod
    def Distance( Vector1, Vector2 ):
        'Returns the distance between two Vectors'
        return math.sqrt( Vector.DistanceSqrd(Vector1,Vector2) )
    
    @staticmethod
    def LengthSqrd( vec ):
        'Returns the length of a vector sqaured. Faster than Length(), but only marginally'
        return vec[0]**2 + vec[1]**2
    
    @staticmethod
    def Length( vec ):
        'Returns the length of a vector'
        return math.sqrt( Vector.LengthSqrd(vec) )
    
    @staticmethod
    def Normalize( vec ):
        'Returns a new vector that has the same direction as vec, but has a length of one.'
        if( vec[0] == 0. and vec[1] == 0. ):
            return Vector(0.,0.)
        return vec / Vector.Length(vec)
    
    @staticmethod
    def Dot( a,b ):
        'Computes the dot product of a and b'
        return a[0]*b[0] + a[1]*b[1]
    
    @staticmethod
    def ProjectOnto( w,v ):
        'Projects w onto v.'
        return v * Dot(w,v) / Vector.LengthSqrd(v)