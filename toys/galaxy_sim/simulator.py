import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

CUDASimulate = SourceModule("""
#define pos_x stars[star_id*7]
#define pos_y stars[(star_id*7)+1]
#define pos_z stars[(star_id*7)+2]
#define vel_x stars[(star_id*7)+3]
#define vel_y stars[(star_id*7)+4]
#define vel_z stars[(star_id*7)+5]
#define mass stars[(star_id*7)+6]

__global__ void sim(float* stars, int numstars, int stride)
{	
	//A star is every 7 floats. 
	//stars[id+0]: pos.x
	//stars[id+1]: pos.y
	//stars[id+2]: pos.z
	//stars[id+3]: vel.x
	//stars[id+4]: vel.y
	//stars[id+5]: vel.z
	//stars[id+6]: mass
	//float posx, posy, posz, velx, vely, velz, accx, accy, accz, mass;

	for(int star_id = blockIdx.x; star_id<numstars; star_id += (stride){

		posx = pos_x;
		posy = pos_y;
		posz = pos_z;
		velx = vel_x;
		vely = vel_y;
		velz = vel_z;

		//This way, every thread is operating on the same, old set of data instead of some operations changing the data
		//therefore making some threads work on partially new data, which would eventually cause instability
		thread_sync();

		//Actually calculate acceleration, new position, velocity, and store all the data. Make sure to use the
		//local variables (those without the underscore) for the calculating, in order to not contaminate with new data.
		
		//Calculate acceleration
		accx = 0;
		accy = 0;
		accz = 1;

		//Calculate velocity
		vel_x = vel_x + accx;
		vel_y = vel_y + accy;
		vel_z = vel_z + accz;

		//Calculate position
		pos_x = pos_x + vel_x;
		pos_y = pos_y + vel_y;
		pos_z = pos_z + vel_z;
	}

	//Make sure every thread is done calculating before going on to the next timestep.
	thread_sync();
}
""").get_function("sim")
print("Compiled and got function gen")

def In(thing):
	thing_pointer = cuda.mem_alloc(thing.nbytes)
	cuda.memcpy_htod(thing_pointer, thing)
	return thing_pointer

def GenerateFractal():