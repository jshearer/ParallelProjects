//Inputs:
//    Number of bodies 
//    Max sphere size
//    AVG temperature
#define MAX_THREADS_PER_BLOCK 1024
#define MAX_BLOCKS 2496
#define SQUARED(a) a*a
typedef struct Particles{
  float * x;
  float * y;
  float * z;
  float * v_x;
  float * v_y;
  float * v_z;
  short * intersects; //Number of times intersected the dust grain
  unsigned char * pType;
  int numBodies;
  void Particles(int nBodies){
    x =          cudaMalloc(nBodies * sizeof(float));
    y =          cudaMalloc(nBodies * sizeof(float));
    z =          cudaMalloc(nBodies * sizeof(float));
    v_x =        cudaMalloc(nBodies * sizeof(float));
    v_y =        cudaMalloc(nBodies * sizeof(float));
    v_z =        cudaMalloc(nBodies * sizeof(float));
    pType =      cudaMalloc(nBodies * sizeof(unsigned char));
    intersects = cudaMalloc(nBodies * sizeof(short));
    numBodies = nBodies;
  };

  void free(){
    cudaFree(x);
    cudaFree(y);
    cudaFree(z);
    cudaFree(v_x);
    cudaFree(v_y);
    cudaFree(v_z);
    cudaFree(intersects);
    cudaFree(pType);
  }
} Particles;

enum DiagnosticsType{
  InitTime = 0;
  StepTime = 1;
}

enum ParticleType{
  Electron = 0;
  Ion = 1;
  Grain = 2;
}

enum DiagnosticsType diag_id;

//Potentially pass into command whether or not to reallocate memory for particles
__device__ Particles * ParticleCollection = NULL; 

// Diagnostics:
// 1. science charging, debye size and shape, plasma condx, etc
// 2. simulation: energy conservation, etc
// 3. cuda: time, memory, etc 

// TODO:
// 1. only prepare if asked for.

__device__ void command(int nBodies, float radius, float temp, bool reallocate, int* diag_id, bool* diag_dataAvailable, int* diag_length, void* diag_data)
{
  cudaEvent_t start,stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  if (reallocate && (ParticleCollection!=NULL) )
  {
    ParticleCollection.free();   
    cudaFree(ParticleCollection);
    ParticleCollection=NULL;
  }

  if(ParticleCollection==NULL)
  {
    ParticleCollection = Particles(nBodies); //dynamically allocate the required memory for n bodies.
  }

  int blocks,threads; 

  if(nBodies<MAX_BLOCKS)
  {
    blocks = nBodies;
    threads = 1;
  } else
  {
    blocks = MAX_BLOCKS
    threads = (nBodies/MAX_BLOCKS)+1; //truncate and add one just to be sure. Okay if falls off, that is checked.
  }

  cudaEventRecord(start,0);
  //Generate initial plasma, and time it.
  prepare<<blocks,threads,1>>(ParticleCollection,radius,tepp);
  cudaEventRecord(end,0);

  float seconds;
  cudaEventElapsedTime(&seconds);
  //Submit plasma generation phase timing results.
  pushDiagnostics(DiagnosticsType.InitTime, seconds, sizeof(float), diag_id, diag_dataAvailable, diag_length, diag_data)

  /*for (i=0, i<numSteps, i++)
  { 

    forceAccum();
    // time diagnostic on force accum
    advanceOrbit();
    // time diagnostic on advance orbit
    boundarCheck();
    // time diagnostic on aboundary check       
    grainColelct();
    // time diagnostic on ocllctChecke       
    pushDiagnostics(DiagnosticsType.StepTime, ..., sizeof(float), diag_id, diag_dataAvailable, diag_length, diag_data)
  } */     
}

// TODO: Make equal parts ions and electrons, don't forget to make the dust grain
__device__ void prepare(Particles ParticleCollection, float radius, float temp) //Generate inital plasma
{
  //(https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialMultidimensionalKernelLaunch)
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  if(index<ParticleCollection.numBodies)
  {
    float x,y,z,v_x,v_y,v_z;
    //thermalize(&x,&y,&z,&v_x,&v_y,&v_z);
    int sqareroot = sqrt(ParticleCollection.numBodies)+1;
    randomize(&x,&y,&z,&v_x,&v_y,&v_z,index,sqareroot);
    ParticleCollection.x[index] = x;
    ParticleCollection.y[index] = y;
    ParticleCollection.z[index] = z;
    ParticleCollection.v_x[index] = v_x;
    ParticleCollection.v_y[index] = v_y;
    ParticleCollection.v_z[index] = v_z;
    ParticleCollection.intersects[index] = 0;
    if(index==0){
      ParticleCollection.pType[index] = ParticleType.Grain;
    } else if(index%2==0){
      ParticleCollection.pType[index] = ParticleType.Electron;  
    } else{
      ParticleCollection.pType[index] = ParticleType.Ion;
    }
  }
}

__device__ void randomize(float * x, float * y, float * z, float * v_x, float * v_y, float * v_z, int index, int sqareroot)
{
  *z = index % sqareroot;
  *y = (index/sqareroot) % sqareroot;
  *x = index/(sqareroot*sqareroot);
  *v_x = 0;
  *v_y = 0;
  *v_z = 0;
}

//Watch out when block queueing happens, might be off by max one time step.
__device__ void collectAndIntegrate(Particles ParticleCollection)
{
  //(https://code.google.com/p/stanford-cs193g-sp2010/wiki/TutorialMultidimensionalKernelLaunch)
  int index_x = blockIdx.x * blockDim.x + threadIdx.x;
  int index_y = blockIdx.y * blockDim.y + threadIdx.y;

  // map the two 2D indices to a single linear, 1D index
  int grid_width = gridDim.x * blockDim.x;
  int index = index_y * grid_width + index_x;

  float fx=0,fy=0,fz=0;

  float k = 3.1415926; // TODO: cmon, fix this

  if(index<ParticleCollection.numBodies)
  {
    float i_x,i_y,i_z,i_type,i_charge,i_v_x,i_v_y,i_v_z;
    float j_x,j_y,j_z,j_type,j_charge;//,j_v_x,j_v_y,j_v_z;
    float rsq,r;
    i_x = ParticleCollection.x[index];
    i_y = ParticleCollection.y[index];
    i_z = ParticleCollection.z[index];
    i_type = ParticleCollection.pType[index];
    i_v_x = ParticleCollection.v_x[index];
    i_v_y = ParticleCollection.v_y[index];
    i_v_z = ParticleCollection.v_z[index];

    // TODO: dimensional scaling will come later
    i_charge = 0;
    if(i_type==ParticleType.Electron){ //electron
      i_charge = -1.0; 
    }
    else if(i_type==ParticleType.Ion){ //ion
      i_charge = 1.0;
    }

    for(j = 0;j<ParticleCollection.numBodies;j++)
    {
      if(j!=index)
      {
        //TODO: Potentially memcpy 6 floats and work directly etc.
        j_x = ParticleCollection.x[j];
        j_y = ParticleCollection.y[j];
        j_z = ParticleCollection.z[j];
        j_type = ParticleCollection.pType[j];
        //j_v_x = ParticleCollection.v_x[index];
        //j_v_y = ParticleCollection.v_y[index];
        //j_v_z = ParticleCollection.v_z[index];

        j_charge = 0;

        if(j_type==0){ //electron
          j_charge = -1.0;
        }
        else if(j_type==1){ //ion
          j_charge = 1.0;
        }
        
        rsq = ((i_x*i_x) - (2*i_x*j_x) + (j_x*j_x))+
              ((i_y*i_y) - (2*i_y*j_y) + (j_y*j_y))+
              ((i_z*i_z) - (2*i_z*j_z) + (j_z*j_z));
        
        r = __sqrt(rsq);
        rcb = rsq*r;

        fx += - j_charge/rcb * (j_x-i_x);
        fy += - j_charge/rcb * (j_y-i_y);
        fz += - j_charge/rcb * (j_z-i_z);
      }
      fx = k * i_charge* fx;
      fy = k * i_charge* fy;
      fz = k * i_charge* fz;

    }
    //INTEGRATE MEEEEEEE
  }
}


__device__ void gatherBlaDiagnostics(Particles ParticleCollection)
{

}

__device__ void pushDiagnostics(DiagnosticsType type, void * data, int length, int* diag_id, bool* diag_dataAvailable, int* diag_length, void* diag_data)
{
  while(!*diag_dataAvailable){
    *diag_id = type;
    *diag_length = length;
    cudaMemcpy(diag_data,data,length,cudaMemcpyDefault); //cudaMemcpyDeviceToHost??
    *diag_dataAvailable = true;
  }
}

//__device__ void queuePush();
//__device__ void handlePushQueue();

//Particles parts = Particles(nBodies);

//float x,y,z,v_x,v_y,v_z;


  















