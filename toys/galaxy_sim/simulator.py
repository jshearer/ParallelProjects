import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import os

KernelCode = """
#define getMass(id) stars[id+6]

__global__ void sim(float* stars, int numstars, int stride, float timestep, int* stars_complete, int steps)
{
	float mass,len,force;
	int this_id, that_id;

	float ThisPos_x,ThisPos_y,ThisPos_z;
	float NewVelocity_x,NewVelocity_y,NewVelocity_z;
	float dir_x,dir_y,dir_z;
	int step;
	for(step = 0; step<steps; step++)
	{
		for(this_id = blockIdx.x; this_id<numstars; this_id += stride*7){
			NewVelocity_x = stars[this_id+3];
			NewVelocity_y = stars[this_id+4];
			NewVelocity_z = stars[this_id+5];

			ThisPos_x = stars[this_id];
			ThisPos_y = stars[this_id+1];
			ThisPos_z = stars[this_id+2];

			for(that_id = 0; that_id<numstars; that_id+=7)
			{
				if(that_id != this_id)
				{
					//F = G*((this.mass*that.mass)/
					//			distance^2)

					//subtract this.pos-that.pos


					dir_x = ThisPos_x-stars[that_id];
					dir_y = ThisPos_y-stars[that_id+1];
					dir_z = ThisPos_z-stars[that_id+2];

					//Calculate force magnitude
					mass = getMass(this_id)*getMass(that_id);
					len = sqrtf(
								powf(dir_x,2)+
								powf(dir_y,2)+
								powf(dir_z,2));

					force = mass/powf(len,2);

					//vec3fNormalize(thisMinusThat); //Calculate direction of influence
					dir_x = dir_x/len;
					dir_y = dir_y/len;
					dir_z = dir_z/len;


					//vec3fMul(thisMinusThat,forceVec,thisMinusThat); //multiply by force magnitude
					dir_x = dir_x * force;
					dir_y = dir_y * force;
					dir_z = dir_z * force;

					//vec3fMul(thisMinusThat,stepVector,thisMinusThat); // *delta-t
					dir_x = dir_x * timestep;
					dir_y = dir_y * timestep;
					dir_z = dir_z* timestep;

					//vec3fAdd(NewVelocity,thisMinusThat,NewVelocity); //Incorporate the calculated force into the new velocity

					NewVelocity_x = NewVelocity_x + dir_x;
					NewVelocity_y = NewVelocity_y + dir_y;
					NewVelocity_z = NewVelocity_z + dir_z;
				}

				//store NewVelocity
				stars[this_id+3] = NewVelocity_x;	
				stars[this_id+4] = NewVelocity_y;
				stars[this_id+5] = NewVelocity_z;

				//store NewPosition
				stars[this_id] = stars[this_id] + NewVelocity_x;
				stars[this_id+1] = stars[this_id+1] + NewVelocity_y;
				stars[this_id+2] = stars[this_id+2] + NewVelocity_z;
			}

			//I would like to wait until *all* blocks have reached this point, but that's not possible in CUDA. 
			//Must find a better way. Currenly with no synchronization scheme in place, 
			//I would expect to see a slow but steady declie in accuracy 
			//as a certain percentage of blocks finish looping sooner than others, thus getting slightly ahead. 
			//Though possibly they will then get slightly behind, thus canceling everything out. 
			//Accuracey wise, this is bad. Performance wise, this is good.
		}
		(*stars_complete)++;
		while((*stars_complete)<numstars){

		}
	}
}
"""

SimKernel = SourceModule(KernelCode)
SimFunc = SimKernel.get_function("sim")

print("Compiled and got function gen")

def In(thing):
	thing_pointer = cuda.mem_alloc(thing.nbytes)
	cuda.memcpy_htod(thing_pointer, thing)
	return thing_pointer


