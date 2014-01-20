import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

CUDASimulate = SourceModule("""
#DEFINE .x [0]
#DEFINE .y [1]
#DEFINE .z [2]

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

__global__ void sim(float* stars, int numstars, int stride)
{	
	for(int star_id = blockIdx.x; star_id<numstars; star_id += (stride){

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