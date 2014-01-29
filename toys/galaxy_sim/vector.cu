#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <vector.cuh>

#define getMass(id) stars[id+6]
#define getPosition(star_id) vec3fCreate(stars[star_id],stars[star_id+1],stars[star_id+2])
#define getVelocity(star_id) vec3fCreate(stars[star_id+3],stars[star_id+4],stars[star_id+5])

__device__ float* vec3fCreate(float x, float y, float z)
{
	float* arr;
	arr = (float*)malloc(sizeof(float)*3);
	arr[0] = x;
	arr[1] = y;
	arr[2] = z;
	return arr;
}

__device__ float* vec3fZeros()
{
	return vec3fCreate(0,0,0);
}

__device__ void vec3fAdd(float* vec1, float* vec2, float* res)
{
	res[0] = vec1[0] + vec2[0];
	res[1] = vec1[1] + vec2[1];
	res[2] = vec1[2] + vec2[2];
}

__device__ void vec3fSub(float* vec1, float* vec2, float* res)
{
	res[0] = vec1[0] - vec2[0];
	res[1] = vec1[1] - vec2[1];
	res[2] = vec1[2] - vec2[2];
}

__device__ void vec3fMul(float* vec1, float* vec2, float* res)
{
	res[0] = vec1[0] * vec2[0];
	res[1] = vec1[1] * vec2[1];
	res[2] = vec1[2] * vec2[2];
}

__device__ void vec3fDiv(float* vec1, float* vec2, float* res)
{
	res[0] = vec1[0] / vec2[0];
	res[1] = vec1[1] / vec2[1];
	res[2] = vec1[2] / vec2[2];
}

__device__ float vec3fLen(float* vec)
{
	return sqrtf(powf(vec[0],2)+
				 powf(vec[1],2)+
				 powf(vec[2],2));
}

__device__ float vec3fDistance(float* vec1, float* vec2)
{
	float* subt = vec3fZeros();
	vec3fSub(vec1, vec2, subt);
	return vec3fLen(subt);
}

__device__ void vec3fNormalize(float* vec)
{
	float len = vec3fLen(vec);
	float* div = vec3fCreate(len,len,len);
	vec3fDiv(vec,div,vec);
	free(div);
}