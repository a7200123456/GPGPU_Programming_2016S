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

__global__ void Downscale(
    const float *input,
    float *output,
    const int wb, const int hb
)
{
    const int yb = blockIdx.y * blockDim.y + threadIdx.y;
    const int xb = blockIdx.x * blockDim.x + threadIdx.x;
    const int curb = wb*yb+xb;
    if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
        output[curb*3+0] = input[2*curb*3+0];
        output[curb*3+1] = input[2*curb*3+1];
        output[curb*3+2] = input[2*curb*3+2];
        // let see github work
    }
}

__global__ void Upscale(
    const float *input,
    float *output,
    const int wb, const int hb
)
{
    const int yb = blockIdx.y * blockDim.y + threadIdx.y;
    const int xb = blockIdx.x * blockDim.x + threadIdx.x;
    const int curb = wb*yb+xb;
    if (0 <= yb and yb < hb and 0 <= xb and xb < wb) {
        output[2*curb*3+0] = input[curb*3+0];    output[(2*curb+wb)*3+0] = input[curb*3+0];
        output[(2*curb+1)*3+0] = input[curb*3+0];output[(2*curb+wb+1)*3+0] = input[curb*3+0];
        output[2*curb*3+1] = input[curb*3+1];    output[(2*curb+wb)*3+1] = input[curb*3+1];
        output[(2*curb+1)*3+1] = input[curb*3+1];output[(2*curb+wb+1)*3+1] = input[curb*3+1];
        output[2*curb*3+2] = input[curb*3+2];    output[(2*curb+wb)*3+2] = input[curb*3+2];
        output[(2*curb+1)*3+2] = input[curb*3+2];output[(2*curb+wb+1)*3+2] = input[curb*3+2];
        // let see github work
    }
}

__global__ void UpscaleClone(
    const float *input,
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
            output[curb*3+0] = input[curb*3+0];
            output[curb*3+1] = input[curb*3+1];
            output[curb*3+2] = input[curb*3+2];
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
    }
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
    float *halfoutput;
    float *halftar;
    float *halfmsk;
    float *temp_halfimg;
    float *temp_halfoutput;
    float *quadoutput;
    float *quadtar;
    float *quadmsk;
    float *temp_quadimg;
    float *temp_quadoutput;
    cudaMalloc((void**)&temp_output, wb*hb*sizeof(float)*3);  
    cudaMalloc((void**)&halfoutput, wb/2*hb/2*sizeof(float)*3);  
    cudaMalloc((void**)&halftar, wt/2*ht/2*sizeof(float)*3);  
    cudaMalloc((void**)&halfmsk, wt/2*ht/2*sizeof(float)*3);  
    cudaMalloc((void**)&temp_halfimg, wb/2*hb/2*sizeof(float)*3); 
    cudaMalloc((void**)&temp_halfoutput, wb/2*hb/2*sizeof(float)*3);  
    cudaMalloc((void**)&quadoutput, wb/4*hb/4*sizeof(float)*3);  
    cudaMalloc((void**)&quadtar, wt/4*ht/4*sizeof(float)*3);  
    cudaMalloc((void**)&quadmsk, wt/4*ht/4*sizeof(float)*3);  
    cudaMalloc((void**)&temp_quadimg, wb/4*hb/4*sizeof(float)*3); 
    cudaMalloc((void**)&temp_quadoutput, wb/4*hb/4*sizeof(float)*3);  

    cudaMemcpy(output, background, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
	SimpleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
		background, target, mask, output,
		wb, hb, wt, ht, oy, ox
	);
    
    Downscale<<<dim3(CeilDiv(wb/2,32), CeilDiv(hb/2,16)), dim3(32,16)>>>(output, halfoutput,wb/2, hb/2);
    Downscale<<<dim3(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), dim3(32,16)>>>(target, halftar,wt/2, ht/2);
    Downscale<<<dim3(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), dim3(32,16)>>>(mask  , halfmsk,wt/2, ht/2);

    Downscale<<<dim3(CeilDiv(wb/4,32), CeilDiv(hb/4,16)), dim3(32,16)>>>(halfoutput, quadoutput,wb/4, hb/4);
    Downscale<<<dim3(CeilDiv(wt/4,32), CeilDiv(ht/4,16)), dim3(32,16)>>>(halftar, quadtar,wt/4, ht/4);
    Downscale<<<dim3(CeilDiv(wt/4,32), CeilDiv(ht/4,16)), dim3(32,16)>>>(halfmsk  , quadmsk,wt/4, ht/4);

    cudaMemcpy(temp_quadoutput,quadoutput, wb/4*hb/4*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    for(int i=0;i<2700;i++){
        JacobiIteration<<<dim3(CeilDiv(wt/4,32), CeilDiv(ht/4,16)), dim3(32,16)>>>(
            temp_quadoutput, quadtar, quadmsk, quadoutput,
            wb/4, hb/4, wt/4, ht/4, oy/4, ox/4
        );
        cudaMemcpy(temp_quadoutput,quadoutput, wb/4*hb/4*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }

    Upscale<<<dim3(CeilDiv(wb/4,32), CeilDiv(hb/4,16)), dim3(32,16)>>>(
        quadoutput, temp_halfoutput,wb/4, hb/4
    );
    UpscaleClone<<<dim3(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), dim3(32,16)>>>(
        temp_halfoutput, halfmsk, halfoutput,
        wb/2, hb/2, wt/2, ht/2, oy/2, ox/2
    );

    cudaMemcpy(temp_halfoutput,halfoutput, wb/2*hb/2*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    for(int i=0;i<5400;i++){
        JacobiIteration<<<dim3(CeilDiv(wt/2,32), CeilDiv(ht/2,16)), dim3(32,16)>>>(
            temp_halfoutput, halftar, halfmsk, halfoutput,
            wb/2, hb/2, wt/2, ht/2, oy/2, ox/2
        );
        cudaMemcpy(temp_halfoutput,halfoutput, wb/2*hb/2*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }

    Upscale<<<dim3(CeilDiv(wb/2,32), CeilDiv(hb/2,16)), dim3(32,16)>>>(
        halfoutput, temp_output,wb/2, hb/2
    );
    UpscaleClone<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
        temp_output, mask, output,
        wb, hb, wt, ht, oy, ox
    );

    cudaMemcpy(temp_output,output, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice); 
    for(int i=0;i<10800;i++){
        JacobiIteration<<<dim3(CeilDiv(wt,32), CeilDiv(ht,16)), dim3(32,16)>>>(
            temp_output, target, mask, output,
            wb, hb, wt, ht, oy, ox
        );
        cudaMemcpy(temp_output,output, wb*hb*sizeof(float)*3, cudaMemcpyDeviceToDevice);
    }


    cudaFree(temp_output); 
    cudaFree(halfoutput ); 
    cudaFree(halftar ); 
    cudaFree(halfmsk ); 
    cudaFree(temp_halfimg ); 
    cudaFree(temp_halfoutput ); 
    cudaFree(quadoutput ); 
    cudaFree(quadtar ); 
    cudaFree(quadmsk ); 
    cudaFree(temp_quadimg ); 
    cudaFree(temp_quadoutput ); 
}
