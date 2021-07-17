//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>

#define TPB 1024
/* Red Eye Removal
   ===============
   
   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
__global__ void histogram(unsigned int *const d_inputVals,
                          unsigned int *const d_binHistogram,
                          const unsigned int bit,
                          const int numBins,
                          const int numElems)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numElems)
      return;
   unsigned int mask = (numBins - 1) << bit;
   unsigned int bin = (d_inputVals[threadId] & mask) >> bit;
   atomicAdd(&d_binHistogram[bin], 1);
}

__global__ void getNegativeMask(const unsigned int *const vals_src,
                                const unsigned int bit,
                                unsigned int *const d_negativeMask,
                                const int numElems)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numElems)
      return;
   unsigned int mask = 1 << bit;
   unsigned int bin = (vals_src[threadId] & mask) >> bit;
   d_negativeMask[threadId] = !bin;
}

__global__ void move_position(unsigned int *const vals_src,
                              unsigned int *const pos_src,
                              unsigned int *const vals_dst,
                              unsigned int *const pos_dst,
                              const unsigned int *const d_binHistogram,
                              const unsigned int *const d_binScan,
                              const unsigned int bit,
                              const int numElems
                              )
{
   // extern __shared__ unsigned int s_binScan;
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numElems)
      return;
   unsigned int mask = 1 << bit;
   unsigned int bin = (vals_src[threadId] & mask) >> bit;
   int des_id = bin ? threadId - d_binScan[threadId] + d_binHistogram[0] : d_binScan[threadId];
   vals_dst[des_id] = vals_src[threadId];
   pos_dst[des_id] = pos_src[threadId];
}

void your_sort(unsigned int *const d_inputVals,
               unsigned int *const d_inputPos,
               unsigned int *const d_outputVals,
               unsigned int *const d_outputPos,
               const size_t numElems)
{
   unsigned int *d_binHistogram, *d_binScan, *d_negativeMask;
   const int numBits = 1;
   const int numBins = 1 << numBits;
   cudaMalloc((void **)&d_binHistogram, sizeof(unsigned int) * numBins);
   cudaMalloc((void **)&d_negativeMask, sizeof(unsigned int) * numElems);
   cudaMalloc((void **)&d_binScan, sizeof(unsigned int) * numElems);

   unsigned int *vals_src = d_inputVals;
   unsigned int *pos_src = d_inputPos;
   unsigned int *vals_dst = d_outputVals;
   unsigned int *pos_dst = d_outputPos;

   const int blocks = (numElems + TPB - 1) / TPB;
   for (unsigned int i = 0; i < 8 * sizeof(unsigned int); i += numBits)
   {
      cudaMemset(d_binHistogram, 0, sizeof(unsigned int)* numBins);
      cudaMemset(d_negativeMask, 0, sizeof(unsigned int)* numElems);
      cudaMemset(d_binScan, 0, sizeof(unsigned int)* numElems);

      // step 1: histogram
      histogram<<<blocks, TPB>>>(vals_src, d_binHistogram, i, numBins, numElems);

      // step 2: exclusive prefix sum, to get offset of first 1
      getNegativeMask<<<blocks, TPB>>>(vals_src, i, d_negativeMask, numElems);
      
      thrust::device_ptr<unsigned int> dev_negativeMask(d_negativeMask);
      thrust::device_ptr<unsigned int> dev_binScan(d_binScan);
      thrust::exclusive_scan(dev_negativeMask, dev_negativeMask + numElems, dev_binScan);
      
      // unsigned int z, o, t;
      // cudaMemcpy(&z, &d_binHistogram[0], sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // cudaMemcpy(&o, &d_binHistogram[1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // cudaMemcpy(&t, &d_binScan[numElems-1], sizeof(unsigned int), cudaMemcpyDeviceToHost);
      // printf("numElems: %d, 0: %d, 1: %d, scan ending: %d\n", numElems, z, o, t);

      // step 3: Determine relative offset of each digit
      move_position<<<blocks, TPB>>>(vals_src, pos_src, vals_dst, pos_dst, d_binHistogram, d_binScan, i, numElems);
      
      cudaMemcpy(vals_src, vals_dst, sizeof(unsigned int)* numElems, cudaMemcpyDeviceToDevice);
      cudaMemcpy(pos_src, pos_dst, sizeof(unsigned int)* numElems, cudaMemcpyDeviceToDevice);
   }
   cudaFree(d_binHistogram);
   cudaFree(d_negativeMask);
   cudaFree(d_binScan);
}
