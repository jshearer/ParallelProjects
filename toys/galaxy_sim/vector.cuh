#ifndef VECTOR_H_INCLUDE
#define VECTOR_H_INCLUDE

__device__ float* vec3fCreate(float x, float y, float z);
__device__ float* vec3fZeros();
__device__ void vec3fAdd(float* vec1, float* vec2, float* res);
__device__ void vec3fSub(float* vec1, float* vec2, float* res);
__device__ void vec3fMul(float* vec1, float* vec2, float* res);
__device__ void vec3fDiv(float* vec1, float* vec2, float* res);
__device__ float vec3fLen(float* vec);
__device__ float vec3fDistance(float* vec1, float* vec2);
__device__ void vec3fNormalize(float* vec);

#endif