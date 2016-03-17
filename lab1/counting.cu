#include "counting.h"
#include <cstdio>
#include <iostream>
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

__global__ void build_tree(thrust::device_ptr<int> seg_tree ,int num,int nodes, int start, int last_start) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //use share maybe TODO
  if (idx < nodes and seg_tree[last_start+2*idx] != 0 and seg_tree[last_start+2*idx+1] != 0) 
		seg_tree[start+idx] = num;
  else
    seg_tree[start+idx] = 0;
}

__global__ void count_p(int *pos, thrust::device_ptr<int> seg_tree ,int num,int nodes, int start, int last_start) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
  //use share maybe TODO
  if (idx < nodes and seg_tree[last_start+2*idx] != 0 and seg_tree[last_start+2*idx+1] != 0) 
		seg_tree[start+idx] = num;
  else
    seg_tree[start+idx] = 0;
}

void CountPosition(const char *text, int *pos, int text_size)
{
		//int* temp_d;
    //int* temp_h = nullptr;
    int last_start_position = 0;
    int start_position = text_size;
    
    thrust::device_ptr<int> seg_tree = thrust::device_malloc<int>(text_size*2);
    init_tree<<<(text_size/512+1), 512>>>(text, seg_tree, text_size);
    for(int i = 1; i <=8; i++ ){
      build_tree<<<text_size/(512*pow(2,i))+1, 512>>>(seg_tree, pow(2,i),text_size/pow(2,i),start_position,last_start_position);
      last_start_position = start_position;
      start_position += (text_size/pow(2,i));
    }
    init_tree<<<(text_size/512+1), 512>>>(seg_tree, pos,text_size);
    //thrust::fill(seg_tree.begin(),seg_tree.begin()+text_size, (int)0 );
    //temp_d = thrust::raw_pointer_cast(seg_tree);
    thrust::device_vector<int> temp_d(seg_tree+text_size,seg_tree+text_size+text_size/2+text_size/4+text_size/8);  
    //cudaMemcpy(h_a, d_a, sizeof(StructA), cudaMemcpyDeviceToHost);
    //cudaMemcpy(temp_h, temp_d, sizeof(int)*text_size, cudaMemcpyDeviceToHost);
    printf("check point\n");
    for(int i=0; i< 100; i++){
      std::cout << temp_d[i];
    }
    std::cout << std::endl;
    for(int i=text_size/2; i< 100+text_size/2; i++){
      std::cout << temp_d[i];
    }
    std::cout << std::endl;
    for(int i=text_size/2+text_size/4; i< 100+text_size/2+text_size/4; i++){
      std::cout << temp_d[i];
    }
    std::cout << std::endl;
    for(int i=text_size/2+text_size/4+text_size/8; i< 100+text_size/2+text_size/4+text_size/8; i++){
      std::cout << temp_d[i];
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
