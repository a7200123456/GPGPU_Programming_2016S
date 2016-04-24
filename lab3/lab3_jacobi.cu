#include "lab3.h"
#include <cstdio>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void SimpleClone(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	const int yt = blockIdx.y * blockDim.y + threadIdx.y;
	const int xt = blockIdx.x * blockDim.x + threadIdx.x;
	const int curt = wt*yt+xt;
	if (yt < ht and xt < wt and mask[curt] > 127.0f) {
		const int yb = oy+yt, xb = ox+xt;
		const int curb = wb*yb+xb;
		if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
			output[curb*3+0] = target[curt*3+0];
			output[curb*3+1] = target[curt*3+1];
			output[curb*3+2] = target[curt*3+2];
            // let see github work
		}
	}
}

__global__ void JacobiIteration(
    const float *pre_output,
    const float *target,
    const float *mask,
    float *output,
    const int wb, const int hb, const int wt, const int ht,
    const int oy, const int ox
)
{
    const int yt = blockIdx.y * blockDim.y + threadIdx.y;
    const int xt = blockIdx.x * blockDim.x + threadIdx.x;
    const int curt = wt*yt+xt;
    if ((yt == ht-1 or xt == wt-1 or yt == 0 or xt == 0) and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = (//4*target[curt*3+0]-(0+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+target[(curt+1)*3+0])+
                                (pre_output[(curb-wb)*3+0]+pre_output[(curb+wb)*3+0]+pre_output[(curb-1)*3+0]+pre_output[(curb+1)*3+0]))/4;
            output[curb*3+1] = (//4*target[curt*3+1]-(0+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+target[(curt+1)*3+1])+
                                (pre_output[(curb-wb)*3+1]+pre_output[(curb+wb)*3+1]+pre_output[(curb-1)*3+1]+pre_output[(curb+1)*3+1]))/4;
            output[curb*3+2] = (//4*target[curt*3+2]-(0+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+target[(curt+1)*3+2])+
                                (pre_output[(curb-wb)*3+2]+pre_output[(curb+wb)*3+2]+pre_output[(curb-1)*3+2]+pre_output[(curb+1)*3+2]))/4;
        }
    }/*
    else if (yt < ht-1 and xt < wt-1 and yt > 0 and xt == 0 and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = (4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+0+target[(curt+1)*3+0])
                                +(pre_output[(curb-wb)*3+0]+pre_output[(curb+wb)*3+0]+pre_output[(curb-1)*3+0]+pre_output[(curb+1)*3+0]))/4;
            output[curb*3+1] = (4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+0+target[(curt+1)*3+1])
                                +(pre_output[(curb-wb)*3+1]+pre_output[(curb+wb)*3+1]+pre_output[(curb-1)*3+1]+pre_output[(curb+1)*3+1]))/4;
            output[curb*3+2] = (4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+0+target[(curt+1)*3+2])
                                +(pre_output[(curb-wb)*3+2]+pre_output[(curb+wb)*3+2]+pre_output[(curb-1)*3+2]+pre_output[(curb+1)*3+2]))/4;
        }
    }
    else if (yt == ht-1 and xt < wt-1 and yt > 0 and xt > 0 and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = (4*target[curt*3+0]-(target[(curt-wt)*3+0]+0+target[(curt-1)*3+0]+target[(curt+1)*3+0])
                                +(pre_output[(curb-wb)*3+0]+pre_output[(curb+wb)*3+0]+pre_output[(curb-1)*3+0]+pre_output[(curb+1)*3+0]))/4;
            output[curb*3+1] = (4*target[curt*3+1]-(target[(curt-wt)*3+1]+0+target[(curt-1)*3+1]+target[(curt+1)*3+1])
                                +(pre_output[(curb-wb)*3+1]+pre_output[(curb+wb)*3+1]+pre_output[(curb-1)*3+1]+pre_output[(curb+1)*3+1]))/4;
            output[curb*3+2] = (4*target[curt*3+2]-(target[(curt-wt)*3+2]+0+target[(curt-1)*3+2]+target[(curt+1)*3+2])
                                +(pre_output[(curb-wb)*3+2]+pre_output[(curb+wb)*3+2]+pre_output[(curb-1)*3+2]+pre_output[(curb+1)*3+2]))/4;
        }
    }
    else if (yt < ht-1 and xt == wt-1 and yt > 0 and xt > 0 and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = (4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+0)
                                +(pre_output[(curb-wb)*3+0]+pre_output[(curb+wb)*3+0]+pre_output[(curb-1)*3+0]+pre_output[(curb+1)*3+0]))/4;
            output[curb*3+1] = (4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+0)
                                +(pre_output[(curb-wb)*3+1]+pre_output[(curb+wb)*3+1]+pre_output[(curb-1)*3+1]+pre_output[(curb+1)*3+1]))/4;
            output[curb*3+2] = (4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+0)
                                +(pre_output[(curb-wb)*3+2]+pre_output[(curb+wb)*3+2]+pre_output[(curb-1)*3+2]+pre_output[(curb+1)*3+2]))/4;
        }
    }*/
    else if (yt < ht-1 and xt < wt-1 and yt > 0 and xt > 0 and mask[curt] > 127.0f) {
        const int yb = oy+yt, xb = ox+xt;
        const int curb = wb*yb+xb;
        if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
            output[curb*3+0] = (4*target[curt*3+0]-(target[(curt-wt)*3+0]+target[(curt+wt)*3+0]+target[(curt-1)*3+0]+target[(curt+1)*3+0])
                                +(pre_output[(curb-wb)*3+0]+pre_output[(curb+wb)*3+0]+pre_output[(curb-1)*3+0]+pre_output[(curb+1)*3+0]))/4;
            output[curb*3+1] = (4*target[curt*3+1]-(target[(curt-wt)*3+1]+target[(curt+wt)*3+1]+target[(curt-1)*3+1]+target[(curt+1)*3+1])
                                +(pre_output[(curb-wb)*3+1]+pre_output[(curb+wb)*3+1]+pre_output[(curb-1)*3+1]+pre_output[(curb+1)*3+1]))/4;
            output[curb*3+2] = (4*target[curt*3+2]-(target[(curt-wt)*3+2]+target[(curt+wt)*3+2]+target[(curt-1)*3+2]+target[(curt+1)*3+2])
                                +(pre_output[(curb-wb)*3+2]+pre_output[(curb+wb)*3+2]+pre_output[(curb-1)*3+2]+pre_output[(curb+1)*3+2]))/4;
        }
    }
}

void PoissonImageCloning(
	const float *background,
	const float *target,
	const float *mask,
	float *output,
	const int wb, const int hb, const int wt, const int ht,
	const int oy, const int ox
)
{
	float *temp_output;
    cudaMalloc((void**)&temp_output, wb*hb*sizeof(float)*3);  

    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
    cudaMemcpy(temp_output,output, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    
    for(int i=0;i<7000;i++){
        JacobiIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            temp_output, target, mask, output,
            wb, hb, wt, ht, oy, ox
        );
        cudaMemcpy(temp_output,output, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }
}
