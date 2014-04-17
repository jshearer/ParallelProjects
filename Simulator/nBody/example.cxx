using namespace std; 

typedef struct ThermalSpeedSelector{
private:
  int unique_vel;
  int redundant_vel;
  double * pool; 
  double bin_width;
  
public:
  double Tkelv;
  double avg_vel;
  
  ThermalSpeedSelector(double TeV, double m1, int numSlots){
    // construction of weighted velocity array
    
    unsigned long seed=time(0);
    double boltz=1.381e-23;
    double * trial_vel = new double[numSlots];
    double * P = new double[numSlots];
    double * Area = new double[numSlots];
    double totalArea=0;
    double minimum=100.;
    int abundance;
    int scale_up;
    int h=0;
    int s=0;
    int counter=0;

    unique_vel=numSlots;

    srandom(seed);  
    Tkelv=TeV*(2./3.)/8.617e-5; // Kelvin temp
    avg_vel=sqrt((8*boltz*Tkelv)/(M_PI*m1));
    double vel_low=avg_vel/20.;
    double vel_high=avg_vel*2.6; 
    
    
    bin_width=(vel_high-vel_low)/(double)unique_vel;
    trial_vel[0]=vel_low;
    
    for(int i=1;i<unique_vel;i++){
      trial_vel[i]=trial_vel[i-1]+bin_width;  
      Area[i]=FIND_AREA (m1,boltz,Tkelv,trial_vel[i],bin_width);
      totalArea=totalArea+Area[i];
    }
    
    for (int i=0;i<unique_vel;i++){
      P[i]=   (Area[i]/totalArea)*100;
      if (P[i]<minimum)minimum=P[i];
    }
    
    // check for size of array needed
    redundant_vel=0;
    if (minimum<0.0001)scale_up=unique_vel;
    if (minimum>=0.0001)scale_up=int(1./minimum);
    
    for (int v=0;v<unique_vel;v++){ // to determine size of array for pool
      abundance=   int(P[v]*scale_up);
      redundant_vel=redundant_vel+(int)abundance;
    }

    pool = new double[redundant_vel];
    
    while (h<unique_vel && s<redundant_vel){
      pool[s]=trial_vel[h];
      s++;
      counter++;
      abundance=   int(P[h]*scale_up);
      if(counter>=abundance){   
    counter=0;
    h++; // start next velocity
      }
    }
  }

  double PickSpeed(){
    return pool[  int(random()/2147483647.0*redundant_vel) ];
  }

  void  CheckSpeedDistribution(){
    //Check Speed Distribution Curve
    
    double * speed = new double[unique_vel];
    double * G = new double[unique_vel];
    int l=0;
    int bins=0;
    int i;

    for (i=0;i<unique_vel;i++){
      G[i]=0;
      speed[i]=0;
    }
    
    for (i=1;i<redundant_vel;i++){
      speed[l]=pool[i];
      if(speed[l]>speed[l-1]) bins=l;
      if(fabs(pool[i]-pool[i-1])<bin_width*0.5) 
    G[l]=G[l]+1; // speed same
      else 
    {
      l++; // speed changed
    }
      
    }
    
    TCanvas *cs=new TCanvas("cs","Speed Distribution",800,800);
    cs->cd(0);
    TH1F *histo1=new TH1F("Histogram 1","Speed Distribution",bins,speed[0],speed[bins]); 
    for (int i=0;i<unique_vel;i++){
      histo1-> SetBinContent(i,G[i]);
    }
    histo1->Draw("Al");
  }
  
} ThermalSpeedSelector;

typedef struct DataSet{
  int numPts;
  double* x;
  double* y;
  double* z;
  char * description;
  char * xLabel;
  char * yLabel;
  char * zLabel;
  DataSet(int _numPts, double* _x, double* _y, double*_z, char*_description, char*_xLabel, char*_yLabel, char*_zLabel): numPts(_numPts), x(_x), y(_y), z(_z), description(_description), xLabel(_xLabel), yLabel(_yLabel), zLabel(_zLabel) {}
  DataSet(int _numPts, double* _x, double* _y, char*_description, char*_xLabel, char*_yLabel): numPts(_numPts), x(_x), y(_y), z(NULL), description(_description), xLabel(_xLabel), yLabel(_yLabel), zLabel(NULL) {}
} DataSet;

typedef struct DataSetPair{
  DataSet* dSet[2];
  DataSetPair(DataSet* dSet0, DataSet* dSet1) 
  {
    dSet[0]=dSet0; 
    dSet[1]=dSet1;
  }
  DataSet* get(int i)
  {
    return dSet[i];
  }
} DataSetPair;