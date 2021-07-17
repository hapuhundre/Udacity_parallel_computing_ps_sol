//Udacity HW 6
//Poisson Blending

/* Background
   ==========

   The goal for this assignment is to take one image (the source) and
   paste it into another image (the destination) attempting to match the
   two images so that the pasting is non-obvious. This is
   known as a "seamless clone".

   The basic ideas are as follows:

   1) Figure out the interior and border of the source image
   2) Use the values of the border pixels in the destination image 
      as boundary conditions for solving a Poisson equation that tells
      us how to blend the images.
   
      No pixels from the destination except pixels on the border
      are used to compute the match.

   Solving the Poisson Equation
   ============================

   There are multiple ways to solve this equation - we choose an iterative
   method - specifically the Jacobi method. Iterative methods start with
   a guess of the solution and then iterate to try and improve the guess
   until it stops changing.  If the problem was well-suited for the method
   then it will stop and where it stops will be the solution.

   The Jacobi method is the simplest iterative method and converges slowly - 
   that is we need a lot of iterations to get to the answer, but it is the
   easiest method to write.

   Jacobi Iterations
   =================

   Our initial guess is going to be the source image itself.  This is a pretty
   good guess for what the blended image will look like and it means that
   we won't have to do as many iterations compared to if we had started far
   from the final solution.

   ImageGuess_prev (Floating point)
   ImageGuess_next (Floating point)

   DestinationImg
   SourceImg

   Follow these steps to implement one iteration:

   1) For every pixel p in the interior, compute two sums over the four neighboring pixels:
      Sum1: If the neighbor is in the interior then += ImageGuess_prev[neighbor]
             else if the neighbor in on the border then += DestinationImg[neighbor]

      Sum2: += SourceImg[p] - SourceImg[neighbor]   (for all four neighbors)

   2) Calculate the new pixel value:
      float newVal= (Sum1 + Sum2) / 4.f  <------ Notice that the result is FLOATING POINT
      ImageGuess_next[p] = min(255, max(0, newVal)); //clamp to [0, 255]


    In this assignment we will do 800 iterations.
   */

#include "utils.h"
#include <algorithm>
#include <thrust/host_vector.h>

#define TPB 1024

__global__ void init_mask(const uchar4 *const h_sourceImg,
                          unsigned char *mask,
                          const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   mask[i] = (h_sourceImg[i].x + h_sourceImg[i].y + h_sourceImg[i].z < 3 * 255) ? 1 : 0;
}

__global__ void init_rgb(const uchar4 *const h_sourceImg,
                          unsigned char *red_src,
                          unsigned char *green_src,
                          unsigned char *blue_src,
                          const uchar4 * const h_destImg, 
                          unsigned char *red_dst,
                          unsigned char *green_dst,
                          unsigned char *blue_dst,
                          const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   red_src[i] = h_sourceImg[i].x;
   blue_src[i] = h_sourceImg[i].y;
   green_src[i] = h_sourceImg[i].z;
   red_dst[i] = h_destImg[i].x;
   blue_dst[i] = h_destImg[i].y;
   green_dst[i] = h_destImg[i].z;
}

__global__ void compute_regions(const unsigned char *const mask,
                                unsigned char *strictInteriorPixels,
                                unsigned char *borderPixels,
                                const size_t numColsSource,
                                const size_t numRowsSource)
{
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;

   if (c < 1 || c >= (numColsSource - 1) || r < 1 || r >= (numRowsSource - 1))
      return;

   if (mask[r * numColsSource + c])
   {
      if (mask[(r - 1) * numColsSource + c] && mask[(r + 1) * numColsSource + c] &&
          mask[r * numColsSource + c - 1] && mask[r * numColsSource + c + 1])
      {
         strictInteriorPixels[r * numColsSource + c] = 1;
         borderPixels[r * numColsSource + c] = 0;
      }
      else
      {
         strictInteriorPixels[r * numColsSource + c] = 0;
         borderPixels[r * numColsSource + c] = 1;
      }
   }
   else
   {
      strictInteriorPixels[r * numColsSource + c] = 0;
      borderPixels[r * numColsSource + c] = 0;
   }
}

__global__
void compute_g(const unsigned char* const src,
               const unsigned char* const strictInteriorPixels,
               float *g, const size_t numColsSource,
               const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   if(strictInteriorPixels[i])
   {
      float sum = 4.f * src[i];

      sum -= (float)src[i-1] + (float)src[i+1];
      sum -= (float)src[i+numColsSource] + (float)src[i-numColsSource];

      g[i] = sum;
   }
}

__global__
void init_buffer(const unsigned char* const src,
                 float *buffer_1, float *buffer_2,
                 const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   buffer_1[i] = src[i];
   buffer_2[i] = src[i];
}

__global__
void compute_iteration(const unsigned char* const dst,
                       const unsigned char* const strictInteriorPixels,
                       const unsigned char* const borderPixels,
                       const size_t numColsSource,
                       const float* const f,
                       const float* const g, 
                       float* const f_next,
                       const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   if(strictInteriorPixels[i])
   {
      float blendedSum = 0.f;
      float borderSum = 0.f;

      if(strictInteriorPixels[i-1]) 
      {
         blendedSum += f[i-1];
      }
      else
      {
         borderSum += dst[i-1];
      }

      if(strictInteriorPixels[i+1])
      {
         blendedSum += f[i+1];
      }
      else
      {
         borderSum += dst[i+1];
      }

      if(strictInteriorPixels[i+numColsSource])
      {
         blendedSum += f[i+numColsSource];
      }
      else
      {
         borderSum += dst[i+numColsSource];
      }

      if(strictInteriorPixels[i-numColsSource])
      {
         blendedSum += f[i-numColsSource];
      }
      else
      {
         borderSum += dst[i-numColsSource];
      }

      float f_next_val = (blendedSum + borderSum + g[i]) / 4.f;
      f_next[i] = min(255.f, max(0.f, f_next_val));
   }
}

__global__
void swap_blended(float *const blender1, 
                  float *const blender2, const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   float tmp = blender1[i];
   blender1[i] = blender2[i];
   blender2[i] = tmp;
}

__global__
void copy_blended(uchar4 *const d_blendedImg,
                  const float *const blendedRed, 
                  const float *const blendedGreen,
                  const float *const blendedBlue, 
                  const unsigned char* const strictInteriorPixels,
                  const size_t srcSize)
{
   int i = threadIdx.x + blockDim.x * blockIdx.x;
   if (i >= srcSize)
      return;
   if(strictInteriorPixels[i])
   {
      d_blendedImg[i].x = (char)blendedRed[i];
      d_blendedImg[i].y = (char)blendedBlue[i];
      d_blendedImg[i].z = (char)blendedGreen[i];
   }
}



void your_blend(const uchar4 *const h_sourceImg, //IN
                const size_t numRowsSource, const size_t numColsSource,
                const uchar4 *const h_destImg, //IN
                uchar4 *const h_blendedImg)    //OUT
{

   /* To Recap here are the steps you need to implement
  
     1) Compute a mask of the pixels from the source image to be copied
        The pixels that shouldn't be copied are completely white, they
        have R=255, G=255, B=255.  Any other pixels SHOULD be copied.

     2) Compute the interior and border regions of the mask.  An interior
        pixel has all 4 neighbors also inside the mask.  A border pixel is
        in the mask itself, but has at least one neighbor that isn't.

     3) Separate out the incoming image into three separate channels

     4) Create two float(!) buffers for each color channel that will
        act as our guesses.  Initialize them to the respective color
        channel of the source image since that will act as our intial guess.

     5) For each color channel perform the Jacobi iteration described 
        above 800 times.

     6) Create the output image by replacing all the interior pixels
        in the destination image with the result of the Jacobi iterations.
        Just cast the floating point values to unsigned chars since we have
        already made sure to clamp them to the correct range.

      Since this is final assignment we provide little boilerplate code to
      help you.  Notice that all the input/output pointers are HOST pointers.

      You will have to allocate all of your own GPU memory and perform your own
      memcopies to get data in and out of the GPU memory.

      Remember to wrap all of your calls with checkCudaErrors() to catch any
      thing that might go wrong.  After each kernel call do:

      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      to catch any errors that happened while executing the kernel.
  */
   const size_t srcSize = numRowsSource * numColsSource;
   const int blocks1D = (srcSize + TPB - 1) / TPB;
   dim3 blockDim(16, 16, 1);
   dim3 gridDim(
       (numColsSource + blockDim.x - 1) / blockDim.x,
       (numRowsSource + blockDim.y - 1) / blockDim.y);
   
   // step 0: image from host to device
   uchar4 * d_sourceImg, *d_destImg, *d_blendedImg;
   cudaMalloc((void **)&d_sourceImg, sizeof(uchar4) * srcSize);
   cudaMalloc((void **)&d_destImg, sizeof(uchar4) * srcSize);
   cudaMalloc((void **)&d_blendedImg, sizeof(uchar4) * srcSize);

   cudaMemcpy(d_sourceImg, h_sourceImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice);
   cudaMemcpy(d_destImg, h_destImg, sizeof(uchar4) * srcSize, cudaMemcpyHostToDevice);

   // step 1: get a mask
   unsigned char *mask, *borderPixels, *strictInteriorPixels;
   cudaMalloc((void **)&mask, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&borderPixels, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&strictInteriorPixels, sizeof(unsigned char) * srcSize);

   init_mask<<<blocks1D, TPB>>>(d_sourceImg, mask, srcSize);

   // step 2: regions strictInteriorPixels & borderPixels // todo interiorPixelList
   compute_regions<<<blockDim, gridDim>>>(
      mask, strictInteriorPixels, borderPixels, numColsSource, numRowsSource);
   
   // step 3: rgb src & rgb dst, and g terms
   unsigned char *red_src, *blue_src, *green_src;
   unsigned char *red_dst, *blue_dst, *green_dst;
   cudaMalloc((void **)&red_src, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&blue_src, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&green_src, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&red_dst, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&blue_dst, sizeof(unsigned char) * srcSize);
   cudaMalloc((void **)&green_dst, sizeof(unsigned char) * srcSize);
   init_rgb<<<blocks1D, TPB>>>(
      d_sourceImg, red_src, green_src, blue_src,
      d_destImg, red_dst, green_dst, blue_dst, srcSize);

   // computeG
   float *g_red, *g_green, *g_blue;
   cudaMalloc((void **)&g_red, sizeof(float) * srcSize);
   cudaMalloc((void **)&g_green, sizeof(float) * srcSize);
   cudaMalloc((void **)&g_blue, sizeof(float) * srcSize);
   cudaMemset(g_red, 0.f, sizeof(float) * srcSize);
   cudaMemset(g_green, 0.f, sizeof(float) * srcSize);
   cudaMemset(g_blue, 0.f, sizeof(float) * srcSize);
   
   compute_g<<<blocks1D, TPB>>>(red_src, strictInteriorPixels, g_red, numColsSource, srcSize);
   compute_g<<<blocks1D, TPB>>>(blue_src, strictInteriorPixels, g_blue, numColsSource, srcSize);
   compute_g<<<blocks1D, TPB>>>(green_src, strictInteriorPixels, g_green, numColsSource, srcSize);

   // step 4: init two buffers of rgb
   float *blendedValsRed_1, *blendedValsRed_2;
   float *blendedValsBlue_1, *blendedValsBlue_2;
   float *blendedValsGreen_1, *blendedValsGreen_2;
   cudaMalloc((void **)&blendedValsRed_1, sizeof(float) * srcSize);
   cudaMalloc((void **)&blendedValsRed_2, sizeof(float) * srcSize);
   cudaMalloc((void **)&blendedValsBlue_1, sizeof(float) * srcSize);
   cudaMalloc((void **)&blendedValsBlue_2, sizeof(float) * srcSize);
   cudaMalloc((void **)&blendedValsGreen_1, sizeof(float) * srcSize);
   cudaMalloc((void **)&blendedValsGreen_2, sizeof(float) * srcSize);

   init_buffer<<<blocks1D, TPB>>>(red_src, blendedValsRed_1, blendedValsRed_2, srcSize);
   init_buffer<<<blocks1D, TPB>>>(green_src, blendedValsGreen_1, blendedValsGreen_2, srcSize);
   init_buffer<<<blocks1D, TPB>>>(blue_src, blendedValsBlue_1, blendedValsBlue_2, srcSize);

   // step 5: solve 
   const size_t numIterations = 800;
   for(size_t i = 0; i < numIterations; ++i)
   {
      compute_iteration<<<blocks1D, TPB>>>(
         red_dst,strictInteriorPixels, borderPixels, numColsSource,
         blendedValsRed_1, g_red, blendedValsRed_2, srcSize);
      // cudaMemcpy(blendedValsRed_1, blendedValsRed_2, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);
      std::swap(blendedValsRed_1, blendedValsRed_2);

      compute_iteration<<<blocks1D, TPB>>>(
         green_dst,strictInteriorPixels, borderPixels, numColsSource,
         blendedValsGreen_1, g_green, blendedValsGreen_2, srcSize);
      // cudaMemcpy(blendedValsGreen_1, blendedValsGreen_2, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);
      std::swap(blendedValsGreen_1, blendedValsGreen_2);

      compute_iteration<<<blocks1D, TPB>>>(
         blue_dst,strictInteriorPixels, borderPixels, numColsSource,
         blendedValsBlue_1, g_blue, blendedValsBlue_2, srcSize);
      // cudaMemcpy(blendedValsBlue_1, blendedValsBlue_2, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);
      std::swap(blendedValsBlue_1, blendedValsBlue_2);
   }
   cudaMemcpy(blendedValsRed_2, blendedValsRed_1, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);
   cudaMemcpy(blendedValsGreen_2, blendedValsGreen_1, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);
   cudaMemcpy(blendedValsBlue_2, blendedValsBlue_1, sizeof(float)*srcSize, cudaMemcpyDeviceToDevice);

   // cudaMemcpy()
   cudaMemcpy(d_blendedImg, d_destImg, sizeof(uchar4) * srcSize, cudaMemcpyDeviceToDevice);
   copy_blended<<<blocks1D, TPB>>>(
      d_blendedImg, blendedValsRed_2, blendedValsGreen_2, blendedValsBlue_2, strictInteriorPixels, srcSize);
   cudaMemcpy(h_blendedImg, d_blendedImg, sizeof(uchar4) * srcSize,cudaMemcpyDeviceToHost);
   // 
   cudaFree(mask);
   cudaFree(borderPixels);
   cudaFree(strictInteriorPixels);
   cudaFree(red_src); cudaFree(green_src); cudaFree(blue_src);
   cudaFree(red_dst); cudaFree(green_dst); cudaFree(blue_dst);
   cudaFree(g_red); cudaFree(g_green); cudaFree(g_blue);
   cudaFree(blendedValsRed_1); cudaFree(blendedValsRed_2);
   cudaFree(blendedValsGreen_1); cudaFree(blendedValsGreen_2);
   cudaFree(blendedValsBlue_1); cudaFree(blendedValsBlue_2);
   cudaFree(d_sourceImg); cudaFree(d_destImg); cudaFree(d_blendedImg);
}
