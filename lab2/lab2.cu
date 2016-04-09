#include "lab2.h"
#define TIMESTEP 0.2
#define DIFF 0.00001
#define VISC 0.000015
static const unsigned NFRAME = 240;
static const unsigned W = 640;
static const unsigned H = 480;


float Lab2VideoGenerator::h_dens[640*480] = {};
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

__global__ void init_dens(float* d_dens ,int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
  //use share maybe TODO
  if (idx>=(W/2-100) && idx<(W/2+100) && idy>=(H/2-100) && idy<(H/2+100)){
    if(t==0)
      d_dens[idy*W+idx] = 128;
    else
      d_dens[idy*W+idx] = d_dens[idy*W+idx]+1;    
  }
  else
    d_dens[idy*W+idx] = 0;
}

__global__ void output_yuv(uint8_t* yuv , float* result, int t) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    yuv[idx] = result[idx];
}

void Lab2VideoGenerator::Generate(uint8_t *yuv) {
  
  float* d_dens,d_dens_old,d_vel_x,d_vel_y,d_vel_x_old,d_vel_y_old;
  
  cudaMalloc((void **) &d_dens      , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_dens_old  , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_x     , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_x_old , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_y     , W*H*sizeof(float)); 
  cudaMalloc((void **) &d_vel_y_old , W*H*sizeof(float)); 
  
  cudaMemcpy(d_dens_old, h_dens, W*H*sizeof(float),cudaMemcpyHostToDevice); 
  
  dim3 blocks(W/16, H/16);
  dim3 threads(16, 16);
  
  init_dens<<<blocks, threads>>>(d_dens_old, impl->t);
  
  output_yuv<<<W*H/512, 512>>>(yuv, d_dens,impl->t);
  
  cudaMemcpy(h_dens, d_dens, W*H*sizeof(float),cudaMemcpyDeviceToHost); 
  
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
