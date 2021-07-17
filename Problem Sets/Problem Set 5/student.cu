/* Udacity HW5
   Histogramming for Speed

   The goal of this assignment is compute a histogram
   as fast as possible.  We have simplified the problem as much as
   possible to allow you to focus solely on the histogramming algorithm.

   The input values that you need to histogram are already the exact
   bins that need to be updated.  This is unlike in HW3 where you needed
   to compute the range of the data and then do:
   bin = (val - valMin) / valRange to determine the bin.

   Here the bin is just:
   bin = val

   so the serial histogram calculation looks like:
   for (i = 0; i < numElems; ++i)
     histo[val[i]]++;

   That's it!  Your job is to make it run as fast as possible!

   The values are normally distributed - you may take
   advantage of this fact in your implementation.

*/

#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
// #include <thrust/copy.h>
// #include <thrust/random.h>
// #include <thrust/inner_product.h>
#include <thrust/binary_search.h>
#include <thrust/adjacent_difference.h>
// #include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>

#include "utils.h"

#define TPB 1024

__global__
void yourHisto(const unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  //TODO fill in this kernel to calculate the histogram
  //as quickly as possible

  //Although we provide only one kernel skeleton,
  //feel free to use more if it will help you
  //write faster code
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  if(threadId >= numElems)
      return ;
  int bin = vals[threadId];
  atomicAdd(&histo[bin], 1);
}

void thrustDenseHistogram(unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  thrust::device_ptr<unsigned int> dev_vals(vals);
  thrust::device_ptr<unsigned int> dev_histo(histo);
  thrust::sort(dev_vals, dev_vals+ numElems);
  
  thrust::counting_iterator<unsigned int> search_begin(0);
  thrust::upper_bound(dev_vals, dev_vals+numElems, 
                      search_begin, search_begin+numBins, 
                      dev_histo);
  thrust::adjacent_difference(dev_histo, dev_histo+numBins, dev_histo);
}

__global__ 
void histoKernelStrategy1(unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  int section_size = (numElems - 1) / (blockDim.x*gridDim.x) + 1;
  int start = threadId*section_size;
  for(int k = 0; k < section_size;k++) {
    if(start + k < numElems) {
      int pos = vals[start + k];
      atomicAdd(&histo[pos], 1);
    }
  } 
}

__global__ 
void histoKernelStrategy2(unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  for(unsigned int i = threadId; i < numElems; i += blockDim.x*gridDim.x){
    int pos = vals[i];
    atomicAdd(&histo[pos],1);
  }
}

__global__ 
void histoKernelStrategy3(unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;

  extern __shared__ unsigned int s_histo[];
  
  for(unsigned int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x){
    s_histo[binIdx] = 0u;
  }

  __syncthreads();

  for(unsigned int i = threadId; i < numElems; i += blockDim.x*gridDim.x){
    int pos = vals[i];
    atomicAdd(&s_histo[pos],1);
  }

  __syncthreads();

  for(unsigned int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
    atomicAdd(&histo[binIdx],s_histo[binIdx]);
  }
}

__global__ 
void histoKernelStrategy4(unsigned int* const vals, //INPUT
               unsigned int* const histo,      //OUPUT
               const unsigned int numElems,
               const unsigned int numBins)
{
  int threadId = threadIdx.x + blockDim.x * blockIdx.x;
  extern __shared__ unsigned int s_histo[];

  for(unsigned int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x){
    s_histo[binIdx] = 0u;
  }

  __syncthreads();
  
  int prev_index = -1;
  unsigned int accumulator = 1;

  for(unsigned int i = threadId; i<numElems; i+= blockDim.x*gridDim.x) {
    int pos = vals[i];
    if(pos != prev_index) {
      atomicAdd(&s_histo[pos], accumulator);
      accumulator = 1;
      prev_index = pos;
    }
    else {
      accumulator++;
    }
  }

  __syncthreads();

  for(unsigned int binIdx = threadIdx.x; binIdx < numBins; binIdx += blockDim.x) {
    atomicAdd(&histo[binIdx],s_histo[binIdx]);
  }
}


void computeHistogram(unsigned int* const d_vals, //INPUT
                      unsigned int* const d_histo,      //OUTPUT
                      const unsigned int numBins,
                      const unsigned int numElems)
{
  //TODO Launch the yourHisto kernel
  printf("numElems: %d, numBins: %d\n", numElems, numBins);
  const int blocks = (numElems + TPB - 1) / TPB;
  //if you want to use/launch more than one kernel,
  //feel free
  // yourHisto<<<blocks,TPB>>>(d_vals, d_histo, numElems, numBins);
  // histoKernelStrategy2<<<blocks,TPB>>>(d_vals, d_histo, numElems, numBins);
  histoKernelStrategy4<<<blocks,TPB, TPB * sizeof(unsigned int)>>>(d_vals, d_histo, numElems, numBins);

  // thrustDenseHistogram(d_vals, d_histo, numElems, numBins);
  cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
}
