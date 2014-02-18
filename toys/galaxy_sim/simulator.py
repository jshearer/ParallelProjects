import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule
import sys
import os

KernelCode = """
#define getMass(id) stars[id+6]
#define vec3fSub(a,b,out) out.x = a.x-b.x; out.y = a.y-b.y; out.z = a.z-b.z; 
#define vec3fAdd(a,b,out) out.x = a.x+b.x; out.y = a.y+b.y; out.z = a.z+b.z;
#define vec3fDiv(a,b,out) out.x = a.x/b.x; out.y = a.y/b.y; out.z = a.z/b.z;
#define vec3fDivConstant(a,b,out) out.x = a.x/b; out.y = a.y/b; out.z = a.z/b;
#define vec3fMul(a,b,out) out.x = a.x*b.x; out.y = a.y*b.y; out.z = a.z*b.z;
#define vec3fMulConstant(a,b,out) out.x = a.x*b; out.y = a.y*b; out.z = a.z*b;
#define vec3fLen(a) sqrtf(powf(a.x,2)+powf(a.y,2)+powf(a.z,2));
#define getPosition(star_id,struct) struct.x = stars[star_id]; struct.y = stars[star_id+1]; struct.z = stars[star_id+2];
#define getVelocity(star_id,struct) struct.x = stars[star_id+3]; struct.y = stars[star_id+4]; struct.z = stars[star_id+5];

struct vec3f{
	float x;
	float y;
	float z;
};
typedef struct vec3f vec3f;

__global__ void sim(float* stars, int numstars, int stride, float timestep, int* stars_complete, int steps)
{
	float mass,len,force;
	int this_id, that_id;

	vec3f ThisPos,ThatPos,NewVelocity,dir;
	int step;

	int leap = 

	for(step = 0; step<steps; step++)
	{
		for(this_id = blockIdx.x; this_id<numstars; this_id += stride*7){
			getVelocity(this_id,NewVelocity);
			getPosition(this_id,ThisPos);

			for(that_id = 0; that_id<numstars; that_id+=7)
			{
				if(that_id != this_id)
				{
					//F = G*((this.mass*that.mass)/
					//			distance^2)

					//subtract this.pos-that.pos

					getPosition(that_id,ThatPos);

					vec3fSub(ThisPos,ThatPos,dir);

					//Calculate force magnitude
					mass = getMass(this_id)*getMass(that_id);
					len = vec3fLen(dir);

					force = mass/powf(len,2);

					//vec3fNormalize(thisMinusThat); //Calculate direction of influence
					vec3fDivConstant(dir,len,dir);


					//vec3fMul(thisMinusThat,forceVec,thisMinusThat); //multiply by force magnitude
					vec3fMulConstant(dir,force,dir);

					//vec3fMul(thisMinusThat,stepVector,thisMinusThat); // *delta-t
					vec3fMulConstant(dir,timestep,dir);

					//vec3fAdd(NewVelocity,thisMinusThat,NewVelocity); //Incorporate the calculated force into the new velocity

					vec3fAdd(NewVelocity,dir,NewVelocity);
				}

				//store NewVelocity
				stars[this_id+3] = NewVelocity.x;
				stars[this_id+4] = NewVelocity.y;
				stars[this_id+5] = NewVelocity.z;

				//store NewPosition
				stars[this_id] = stars[this_id] + NewVelocity.x;
				stars[this_id+1] = stars[this_id+1] + NewVelocity.y;
				stars[this_id+2] = stars[this_id+2] + NewVelocity.z;
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

#Stars = Array 	[
#	['position'] = Array[
#		[0] = x,
#		[1] = y,
#		[2] = z
#	],

#	['velocity'] = Array[
#		[0] = x,
#		[1] = y,
#		[2] = z
#	],
#	
#	['mass'] = 9
#]

num_cores = 10
threads_per_core = 10

def StarSim(stars,timestep,steps):
	#Create star array
	star_array = numpy.zeros(len(stars)*7, dtype=numpy.float32)

	star_id = 0
	for star in stars:
		star_array[star_id] = star['position'][0]
		star_array[star_id+1] = star['position'][1]
		star_array[star_id+2] = star['position'][2]

		star_array[star_id+3] = star['velocity'][0]
		star_array[star_id+4] = star['velocity'][1]
		star_array[star_id+5] = star['velocity'][2]

		star_array[star_id+6] = star['mass']
		star_id = star_id + 7

	
	star_array_pointer = In(star_array)
	numstars = In(numpy.int32(len(stars)))

	
