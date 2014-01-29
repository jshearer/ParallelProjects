#include "vector.cuh"
#define getMass(id) stars[id+6]
#define getPosition(star_id) vec3fCreate(stars[star_id],stars[star_id+1],stars[star_id+2])
#define getVelocity(star_id) vec3fCreate(stars[star_id+3],stars[star_id+4],stars[star_id+5])

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
				//F = G*((this.mass*that.mass)/
				//			distance^2)

				float* thisPosition = getPosition(star_id);
				float* thatPosition = getPosition(that_id);
				float* thisMinusThat = vec3fZeros();
				vec3fSub(thisPosition,thatPosition,thisMinusThat);

				//Calculate force magnitude
				mass = getMass(star_id)*getMass(that_id);
				len = vec3fLen(thisMinusThat);
				force = mass/powf(len,2);
				float* forceVec = vec3fCreate(force,force,force);

				vec3fNormalize(thisMinusThat); //Calculate direction of influence
				vec3fMul(thisMinusThat,forceVec,thisMinusThat); //multiply by force magnitude
				vec3fMul(thisMinusThat,stepVector,thisMinusThat); // *delta-t

				vec3fAdd(NewVelocity,thisMinusThat,NewVelocity); //Incorporate the calculated force into the new velocity

				free(*thisPosition);
				free(*thatPosition);
				free(*forceVec);
				free(*thisMinusThat);
			}

			free(*NewVelocity);
		}

		//I would like to wait until *all* blocks have reached this point, but that's not possible in CUDA. Must find a better way. Currenly with no synchronization scheme in place, I would expect to see a slow but steady declie in accuracy as a certain percentage of blocks finish looping sooner than others, thus getting slightly ahead. Though possibly they will then get slightly behind, thus canceling everything out. Accuracey wise, this is bad. Performance wise, this is good.

		//store NewVelocity
		stars[star_id+3] = NewVelocity[0];
		stars[star_id+4] = NewVelocity[1];
		stars[star_id+5] = NewVelocity[2];

		NewPosition = getPosition(star_id);
		vec3fAdd(NewPosition,NewVelocity,NewPosition);

		//store NewPosition
		stars[star_id] = NewPosition[0];
		stars[star_id+1] = NewPosition[1];
		stars[star_id+2] = NewPosition[2];

		free(*NewPosition);
		//free(NewVelocity); //Already free it in the loop for every particle

		
	}

	//Make sure every thread is done calculating before going on to the next timestep.
	thread_sync();
}