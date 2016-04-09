#include "lab2.h"
#define TIMESTEP 0.04
#define DIFF 0.0001
#define VISC 0.000015
static const unsigned NFRAME = 240;
static const unsigned W = 640;
static const unsigned H = 480;

void SWAP(float* A, float* B) {
    float *d_temp;
    
    cudaMalloc((void **) &d_temp      , W*H*sizeof(float)); 
   
    cudaMemcpy(d_temp, A     , W*H*sizeof(float),cudaMemcpyDeviceToDevice); 
    cudaMemcpy(A     , B     , W*H*sizeof(float),cudaMemcpyDeviceToDevice); 
    cudaMemcpy(B     , d_temp, W*H*sizeof(float),cudaMemcpyDeviceToDevice); 
  
    cudaFree(d_temp); 
}


float Lab2VideoGenerator::h_dens[640*480] = {};
float Lab2VideoGenerator::h_vel_x[640*480] = {};
float Lab2VideoGenerator::h_vel_y[640*480] = {};
struct Lab2VideoGenerator::Impl {
	int t = 0;
};

Lab2VideoGenerator::Lab2VideoGenerator(): impl(new Impl) {
}

Lab2VideoGenerator::~Lab2VideoGenerator() {}

void Lab2VideoGenerator::get_info(Lab2VideoInfo &info) {
	info.w = W;
	info.h = H;
	info.n_frame = NFRAME;
	// fps = 24/1 = 24
	info.fps_n = 24;
	info.fps_d = 1;
};

__global__ void init_vel(float* d_vel_x ,float* d_vel_y, float* d_vel_x_old,float* d_vel_y_old,int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx>=(W/2-100) && idx<(W/2+100) && idy>=(H/2-100) && idy<(H/2+100)){
    if(t==0){
      d_vel_x[idy*W+idx] = 64;
      d_vel_y[idy*W+idx] = 64;
    }
    else{
      d_vel_x[idy*W+idx] = d_vel_x_old[idy*W+idx];    
      d_vel_y[idy*W+idx] = d_vel_y_old[idy*W+idx];    
    }
  }
  else{
    if(t==0){
      d_vel_x[idy*W+idx] = 0;
      d_vel_y[idy*W+idx] = 0;
    }
    else{
      d_vel_x[idy*W+idx] = d_vel_x_old[idy*W+idx];  
      d_vel_y[idy*W+idx] = d_vel_y_old[idy*W+idx];      
    }
  }
}

__global__ void add_force(float* d_vel_x ,float* d_vel_y, float* d_vel_x_old,float* d_vel_y_old,float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
      d_vel_x[idy*W+idx] += (d_vel_x_old[idy*W+idx]*dt) ;
      d_vel_y[idy*W+idx] += (d_vel_y_old[idy*W+idx]*dt) ;
}

__global__ void diff_vel(float* d_vel_x ,float* d_vel_y, float* d_vel_x_old,float* d_vel_y_old, float diff, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int a = dt*diff*W*H;
    if (idx>=1 && idx<(W-1) && idy>=1 && idy<(H-1)){
      for(int k=0; k < 20; k++){
        d_vel_x[idy*W+idx] = (d_vel_x_old[idy*W+idx] + a*(d_vel_x[(idy-1)*W+idx] + d_vel_x[(idy+1)*W+idx] + d_vel_x[idy*W+(idx-1)] + d_vel_x[idy*W+(idx+1)]  ))/(1+4*a);
        d_vel_y[idy*W+idx] = (d_vel_y_old[idy*W+idx] + a*(d_vel_y[(idy-1)*W+idx] + d_vel_y[(idy+1)*W+idx] + d_vel_y[idy*W+(idx-1)] + d_vel_y[idy*W+(idx+1)]  ))/(1+4*a);
      }
    }
}

__global__ void init_dens(float* d_dens , float* d_dens_old,int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx>=(W/2-50) && idx<(W/2+50) && idy>=(H/2-50) && idy<(H/2+50)){
    if(t==0)
      d_dens[idy*W+idx] = 64;
    else
      d_dens[idy*W+idx] = d_dens_old[idy*W+idx];    
  }
  else{
    if(t==0)
      d_dens[idy*W+idx] = 0;
    else
      d_dens[idy*W+idx] = d_dens_old[idy*W+idx];    
  }
}

__global__ void add_source(float* d_dens,float* d_dens_old ,float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
      d_dens[idy*W+idx] += (d_dens_old[idy*W+idx]*dt) ;
}

__global__ void diff_dens(float* d_dens,float* d_dens_old , float diff, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int a = dt*diff*W*H;
  
    if (idx>=1 && idx<(W-1) && idy>=1 && idy<(H-1)){
      for(int k=0; k < 20; k++){
        d_dens[idy*W+idx] = (d_dens_old[idy*W+idx] + a*(d_dens[(idy-1)*W+idx] + d_dens[(idy+1)*W+idx] + d_dens[idy*W+(idx-1)] + d_dens[idy*W+(idx+1)]  ))/(1+4*a);
      }
    }
}

__global__ void advec_dens(float* d_dens,float* d_dens_old , float* d_vel_x,float* d_vel_y, float dt) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
   
    float pre_x = idx - dt*W*d_vel_x[idy*W+idx];
    float pre_y = idy - dt*H*d_vel_y[idy*W+idx];
    
    if (pre_x<0.5) pre_x = 0.5; 
    if (pre_x>(W-1-0.5)) pre_x = (W-1-0.5);
    if (pre_y<0.5) pre_y = 0.5; 
    if (pre_y>(H-1-0.5)) pre_y = (H-1-0.5);

    int left = (int)pre_x;
    int top = (int)pre_y;
    
    d_dens[idy*W+idx] = (pre_x-left)*(pre_y-top)*d_dens_old[(top+1)*W+left+1]+
                        (left+1-pre_x)*(pre_y-top)*d_dens_old[(top+1)*W+left]+
                        (pre_x-left)*(top+1-pre_y)*d_dens_old[(top)*W+left+1]+
                        (left+1-pre_x)*(top+1-pre_y)*d_dens_old[(top)*W+left];
}

__global__ void output_yuv(uint8_t* yuv , float* result, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(result[idx] >=255)
      yuv[idx] =  255;
    else if (result[idx] < 0)
      yuv[idx] = 0;
    else
      yuv[idx] = result[idx];
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
  
  float *d_dens, *d_dens_old;
  float *d_vel_x,*d_vel_y,*d_vel_x_old,*d_vel_y_old;
  
  cudaMalloc((void **) &d_dens      , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_dens_old  , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_x     , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_x_old , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_y     , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_y_old , W*H*sizeof(float)); 
  
  cudaMemcpy(d_dens_old, h_dens, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  cudaMemcpy(d_vel_x_old, h_vel_x, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  cudaMemcpy(d_vel_y_old, h_vel_y, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  
  dim3 blocks(W/16, H/16);
  dim3 threads(16, 16);
  
  //velocity step
  init_vel<<<blocks, threads>>>(d_vel_x,d_vel_y, d_vel_x_old,d_vel_y_old, impl->t);
  add_force<<<blocks, threads>>>(d_vel_x,d_vel_y, d_vel_x_old,d_vel_y_old,TIMESTEP);
  SWAP(d_vel_x, d_vel_x_old);
  SWAP(d_vel_y, d_vel_y_old);
  diff_vel<<<blocks, threads>>>(d_vel_x,d_vel_y, d_vel_x_old,d_vel_y_old,DIFF,TIMESTEP);
  
  //density  step
  init_dens<<<blocks, threads>>>(d_dens, d_dens_old, impl->t);
  add_source<<<blocks, threads>>>(d_dens,d_dens_old,TIMESTEP);
  SWAP(d_dens, d_dens_old);
  diff_dens<<<blocks, threads>>>(d_dens,d_dens_old,DIFF,TIMESTEP);
  SWAP(d_dens, d_dens_old);
  advec_dens<<<blocks, threads>>>(d_dens,d_dens_old,d_vel_x,d_vel_y,TIMESTEP);

  //output
  output_yuv<<<W*H/512, 512>>>(yuv, d_dens,impl->t);
  
  cudaMemcpy(h_dens, d_dens, W*H*sizeof(float),cudaMemcpyDeviceToHost); 
  cudaMemcpy(h_vel_x, d_vel_x_old, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  cudaMemcpy(h_vel_y, d_vel_y_old, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  
  //cudaMemset(yuv, (impl->t)*255/NFRAME, W*H);
	cudaMemset(yuv+W*H, 128, W*H/2);
  cudaFree(d_dens); 
  cudaFree(d_dens_old ); 
  cudaFree(d_vel_x    ); 
  cudaFree(d_vel_x_old); 
  cudaFree(d_vel_y    ); 
  cudaFree(d_vel_y_old); 
  
	++(impl->t);
}
