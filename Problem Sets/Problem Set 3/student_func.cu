/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.  

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.
  

  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/

#include "utils.h"

#define TPB 1024 // ThreadsPerBlock

__global__ void reduce_kernel(
    const float *const input, float *output,
    const int arraySize, bool is_max)
{
   //todo
   extern __shared__ float sdata[];
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   int tid = threadIdx.x;

   if (threadId >= arraySize)
      return;

   sdata[tid] = input[threadId];
   __syncthreads();

   for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1)
   {
      if (tid < s)
      {
         sdata[tid] = is_max ? max(sdata[tid], sdata[tid + s]) : min(sdata[tid], sdata[tid + s]);
      }
      __syncthreads();
   }

   if (tid == 0)
   {
      output[blockIdx.x] = sdata[0];
   }
}

/**
 * bin_index = (lum[i] - lumMin) / lumRange * numBins
 */
__global__ void histogram(const float *const d_logLuminance,
                          unsigned int *const d_cdf,
                          const float lumMin,
                          const float lumRange,
                          const size_t numBins,
                          const int arraySize)
{
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= arraySize)
      return;
   float ratio = (d_logLuminance[threadId] - lumMin) / lumRange;
   int bin_index = ratio * numBins;
   atomicAdd(&d_cdf[bin_index], 1);
}

__global__ void prefix_sum(unsigned int *const d_cdf, const size_t numBins)
{
   // hillis steele scan (inclusive scan)
   extern __shared__ unsigned int s_cdf[];
   int threadId = threadIdx.x + blockDim.x * blockIdx.x;
   if (threadId >= numBins)
      return;

   s_cdf[threadId] = d_cdf[threadId];
   __syncthreads();

   for (unsigned int offset = 1; offset <= numBins / 2; offset <<= 1)
   {
      int left = threadId - offset;
      int left_val = 0;
      if (left >= 0)
         left_val = s_cdf[left];
      __syncthreads();
      if(left >= 0)
         s_cdf[threadId] += left_val;
      __syncthreads();
   }
   // inclusive to exclusive
   if (threadId == 0)
      d_cdf[threadId] = 0;
   else
      d_cdf[threadId] = s_cdf[threadId - 1];
}

void your_histogram_and_prefixsum(const float *const d_logLuminance,
                                  unsigned int *const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
   //TODO
   /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

   // step 1 & 2: max , min & range
   const int blocks = (numRows * numCols + TPB - 1) / TPB;
   printf("blocks: %d, numRows: %zu, numCols: %zu\n", blocks, numRows, numCols);

   float *min_intermediate, *max_intermediate;
   cudaMalloc((void **)&min_intermediate, blocks * sizeof(float));
   cudaMalloc((void **)&max_intermediate, blocks * sizeof(float));

   reduce_kernel<<<blocks, TPB, TPB * sizeof(float)>>>(d_logLuminance, min_intermediate, numRows * numCols, false);

   reduce_kernel<<<blocks, TPB, TPB * sizeof(float)>>>(d_logLuminance, max_intermediate, numRows * numCols, true);

   float *min_final, *max_final;
   cudaMalloc((void **)&min_final, 1 * sizeof(float));
   cudaMalloc((void **)&max_final, 1 * sizeof(float));

   reduce_kernel<<<1, blocks, blocks * sizeof(float)>>>(min_intermediate, min_final, blocks, false);
   reduce_kernel<<<1, blocks, blocks * sizeof(float)>>>(max_intermediate, max_final, blocks, true);
   cudaMemcpy(&min_logLum, &min_final[0], sizeof(float), cudaMemcpyDeviceToHost);
   cudaMemcpy(&max_logLum, &max_final[0], sizeof(float), cudaMemcpyDeviceToHost);

   printf("CUDA2 min: %f, max: %f\n", min_logLum, max_logLum);
   // REF min: -4.000000, max: 2.317289 for ../memorial_raw_large.png
   cudaFree(min_intermediate);
   cudaFree(max_intermediate);
   cudaFree(min_final);
   cudaFree(max_final);

   // step 3: histogram
   histogram<<<blocks, TPB>>>(d_logLuminance, d_cdf, min_logLum, max_logLum - min_logLum, numBins, numRows * numCols);

   // step 4: prefix sum
   prefix_sum<<<1, TPB, TPB * sizeof(unsigned int)>>>(d_cdf, numBins);
}

void thrust_histogram_and_prefixsum(const float *const d_logLuminance,
                                    unsigned int *const d_cdf,
                                    float &min_logLum,
                                    float &max_logLum,
                                    const size_t numRows,
                                    const size_t numCols,
                                    const size_t numBins)
{
   // todo
}