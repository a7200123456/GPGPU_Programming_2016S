#include "counting.h"
#include <cstdio>
#include <cassert>
#include <thrust/scan.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h> // add 
#include <thrust/execution_policy.h>

__device__ __host__ int CeilDiv(int a, int b) { return (a-1)/b + 1; }
__device__ __host__ int CeilAlign(int a, int b) { return CeilDiv(a, b) * b; }

__global__ void init_tree(const char *text, thrust::device_ptr<int> seg_tree, int text_size) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < text_size and text[idx] != '\n') 
		seg_tree[idx] = 1;
  else
    seg_tree[idx] = 0;
}

//__global__ void build_tree(char *input_gpu,  int fsize, int offset) {
//	int idx = blockIdx.x * blockDim.x + threadIdx.x + offset;
//}

void CountPosition(const char *text, int *pos, int text_size)
{
		int* temp;
    thrust::device_ptr<int> seg_tree = thrust::device_malloc<int>(text_size*2);
    //init_tree<<<(text_size/512+1), 512>>>(text, seg_tree, text_size);
    thrust::fill(seg_tree,seg_tree+text_size, (int)0 );
    temp = thrust::raw_pointer_cast(seg_tree);
    //cudaMemcpy(temp, seg_tree, sizeof(int)*text_size, cudaMemcpyDeviceToHost);
    printf("check point\n");
    for(int i=0; i< text_size; i++){
      printf("%d", temp[i] );
    }
    //build_tree_layer<<<(text_size/512+1), 512>>>(text, fsize, offset*i/64);
}

int ExtractHead(const int *pos, int *head, int text_size)
{
	int *buffer;
	int nhead;
	cudaMalloc(&buffer, sizeof(int)*text_size*2); // this is enough
	thrust::device_ptr<const int> pos_d(pos);
	thrust::device_ptr<int> head_d(head), flag_d(buffer), cumsum_d(buffer+text_size);

	// TODO

	cudaFree(buffer);
	return nhead;
}

void Part3(char *text, int *pos, int *head, int text_size, int n_head)
{
}
