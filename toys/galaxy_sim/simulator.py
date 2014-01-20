import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

CUDASimulate = SourceModule("""
#define .x [0]
#define .y [1]
#define .z [2]
#define getMass(id) stars[id+6]
#define getPosition(star_id) vec3fCreate(stars[star_id],stars[star_id+1],stars[star_id+2])
#define getVelocity(star_id) vec3fCreate(stars[star_id+3],stars[star_id+4],stars[star_id+5])

__device__ float* vec3fCreate(float x, float y, float z)
{
	static float arr[3];
	arr.x = x;
	arr.y = y;
	arr.z = z;
	return arr;
}

__device__ float* vec3fZeros()
{
	return vec3fCreate(0,0,0);
}

__device__ void vec3fAdd(float* vec1, float* vec2, float* res)
{
	res.x = vec1.x + vec2.x;
	res.y = vec1.y + vec2.y;
	res.z = vec1.z + vec2.z;
}

__device__ void vec3fSub(float* vec1, float* vec2, float* res)
{
	res.x = vec1.x - vec2.x;
	res.y = vec1.y - vec2.y;
	res.z = vec1.z - vec2.z;
}

__device__ void vec3fMul(float* vec1, float* vec2, float* res)
{
	res.x = vec1.x * vec2.x;
	res.y = vec1.y * vec2.y;
	res.z = vec1.z * vec2.z;
}

__device__ void vec3fDiv(float* vec1, float* vec2, float* res)
{
	res.x = vec1.x / vec2.x;
	res.y = vec1.y / vec2.y;
	res.z = vec1.z / vec2.z;
}

__device__ float vec3fLen(float* vec)
{
	return sqrtf(powf(vec.x,2)+
				 powf(vec.y,2)+
				 powf(vec.z,2));
}

__device__ float vec3fDistance(float* vec1, float* vec2)
{
	float subt = 0;
	vec3fSub(vec1, vec2, subt);
	return vec3fLen(subt);
}

__device__ void vec3fNormalize(float* vec)
{
	float len = vec3fLen(vec);
	float* div = vec3fCreate(len,len,len);
	vec3fDiv(vec,div,vec);
	free(len);
	free(div);
}

__global__ void sim(float* stars, int numstars, int stride, float timestep)
{
	float* NewVelocity, NewPosition;
	float* stepVector = vec3fCreate(timestep,timestep,timestep);
	float mass,len,force;
	for(int star_id = blockIdx.x; star_id<numstars; star_id += stride*7){

		NewVelocity = getVelocity(star_id);

		for(int that_id = 0; that_id<numstars; that_id+=7)
		{
			if(that_id != star_id)
			{
				float* thisPosition = getPosition(star_id);
				float* thatPosition = getPosition(that_id);
				float* thisMinusThat = float[3];
				vec3fSub(thisPosition,thatPosition,thisMinusThat);

				//Calculate force magnitude
				mass = getMass(star_id)*getMass(that_id);
				len = vec3fLen(thisMinusThat);
				force = mass/powf(len,2);
				float* forceVec = vec3fCreate(force,force,force);

				vec3fNormalize(thisMinusThat); //Calculate direction of influence
				vec3fMul(thisMinusThat,forceVec,thisMinusThat); //multiply by force magnitude
				vec3fMul(thisMinusThat,stepVector,thisMinusThat);

				vec3fAdd(NewVelocity,thisMinusThat,NewVelocity)

				free(thisPosition);
				free(thatPosition);
				free(forceVec);
				free(thisMinusThat);
			}
		}

		//I would like to wait until *all* blocks have reached this point, but that's not possible in CUDA. Must find a better way. Currenly with no synchronization scheme in place, I would expect to see a slow but steady declie in accuracy as a certain percentage of blocks finish looping sooner than others, thus getting slightly ahead. Though possibly they will then get slightly behind, thus canceling everything out. Accuracey wise, this is bad. Performance wise, this is good.

		//store NewVelocity
		stars[star_id+3] = NewVelocity.x
		stars[star_id+4] = NewVelocity.y
		stars[star_id+5] = NewVelocity.z

		NewPosition = getPosition(star_id);
		vec3fAdd(NewPosition,NewVelocity,NewPosition);

		//store NewPosition
		stars[star_id] = NewPosition.x
		stars[star_id+1] = NewPosition.y
		stars[star_id+2] = NewPosition.z

		free(NewPosition);
		free(NewVelocity);

		
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