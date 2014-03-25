#include <cuComplex.h>

__global__ void gen(int px_per_block[2],int px_per_thread[2],int size[2],float position[2],float *zoom,
		    int *iterations,int *result, int* progress,int action)
{ 	
  //blockDim = size of threads per block
  //gridDim = size of blocks
  
  //int size[2] argument is just to make sure we don't fall off the edge and crash the entire machine...
  
  //actions: 0 = write
  //	     1 = read+write
  //         2 = none	
  //	     3 = atomicAddTest
  //	     4 = overlapMap
  
  int startx = (blockIdx.x*px_per_block[0])+(threadIdx.x*px_per_thread[0]);
  int starty = (blockIdx.y*px_per_block[1])+(threadIdx.y*px_per_thread[1]);
  
  float t_x, t_y;
  int i, x, y;
  int pixelVal;  // long int needed???

  cuFloatComplex z = cuFloatComplex();
  cuFloatComplex z_unchanging = cuFloatComplex();
  
  float z_real, z_imag;
  
  for(x = startx; x < startx+px_per_thread[0]; x++){
    for(y = starty; y < starty+px_per_thread[1]; y++){
      pixelVal = x + (y*size[0]); //   map2Dto1D(x,y,size[0]);
      if(action==4) //generate overlap map
	{
	  result[pixelVal] = result[pixelVal] + 1;
	  continue;
	} 
      if(action==3)
	{
	  atomicAdd(progress,1);
	}

      t_x = (x+position[0])/(*zoom);
      t_y = (y+position[1])/(*zoom);
      
      z.x = t_x;
      z.y = t_y;
      z_unchanging.x = t_x;
      z_unchanging.y = t_y; //optomize this with pointer magic?
      
      for(i = 0; i<(*iterations) + 1; i++) {
	z = cuCmulf(z,z);
	z = cuCaddf(z,z_unchanging); //z = z^2 + z_orig
	z_real = cuCrealf(z);
	z_imag = cuCimagf(z);
	if((z_real*z_real + z_imag*z_imag)>4){
	  if(action==0)//act cool, do the default
	    {
	      result[pixelVal] = i;
	    } else if(action==1)// read+write test
	    {
	      result[pixelVal] = result[pixelVal] + 1;
	    }//else if action==2, do nothing
	  break;
	}
      }
    }
  }
}

/* Local Variables:  */
/* mode: c           */
/* comment-column: 0 */
/* End:              */
