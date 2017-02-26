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
#include <stdio.h>

#include <thrust/functional.h>
#include <thrust/sort.h>

enum ReduceOptions
{
  kMaximize,
  kMinimize
};

__global__
void reduce_minm_maxm(float* d_out, const float* const d_in, ReduceOptions option) 
{

    extern __shared__ float sdata[];
    
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int local_tid = threadIdx.x;

    sdata[local_tid] = d_in[global_tid];        // copy into shared memory from global memory
    __syncthreads();
    
    //Reduce within block
    for( unsigned int delta_size = blockDim.x/2; delta_size > 0; delta_size >>= 1) {
        if( local_tid < delta_size) {
            if(option == kMaximize) {
                sdata[local_tid] = max(sdata[local_tid], sdata[local_tid + delta_size]);
            } else if ( option == kMinimize) {
                sdata[local_tid] = min(sdata[local_tid], sdata[local_tid + delta_size]);
            }
        }
        __syncthreads();
    }

    // first thread in every block holds the value
    if(local_tid == 0 ) {
        d_out[blockIdx.x] = sdata[local_tid];
    }
}

__global__
void atomic_histogram(unsigned int* d_out, const float* const d_in, const float lumMin, const float lumRange, const size_t numBins) 
{
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int bin_value = ((d_in[global_tid] - lumMin)/lumRange)*numBins;

    atomicAdd(&d_out[bin_value], 1);
}

__global__
void blelloch_scan(unsigned int* d_in, const size_t n) 
{
    extern __shared__ unsigned int s[];

    unsigned int tid = threadIdx.x;
    s[tid] = d_in[tid];
    __syncthreads();

    //reduce
    for(unsigned int step = 1; step < n; step*=2) {
        if( (tid+1)%(2*step) == 0 && tid >= step) {
            s[tid] += s[tid-step];
        }
        __syncthreads();
    }

    s[n-1] = 0;

    //downsweep
    for( unsigned int step = n/2; step > 0; step/=2 ) {
        if( (tid+1)%(2*step) == 0 && tid >= step) {
            unsigned int temp = s[tid];
            s[tid] += s[tid-step];
            s[tid-step] = temp;
        }
        __syncthreads();
    }

    d_in[tid] = s[tid];
}

void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum */

  //Use Parallel Reduce to compute minm and maxm for array d_logLum
    //printf("%d  %d %d\n", numRows, numCols, numBins);   //384  256 1024

    int block_size = 1 << 8;
    int grid_size = ceil((float)numCols * numRows / block_size);

    float* d_temp;
    checkCudaErrors(cudaMalloc(&d_temp, sizeof(float) * grid_size));

    //Max    
    reduce_minm_maxm<<<grid_size, block_size, block_size*sizeof(float)>>>(d_temp, d_logLuminance, kMaximize);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    
    float* d_max_logLum;
    checkCudaErrors(cudaMalloc(&d_max_logLum, sizeof(float)));
    reduce_minm_maxm<<<1, grid_size, grid_size*sizeof(float)>>>(d_max_logLum, d_temp, kMaximize);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&max_logLum, d_max_logLum, sizeof(float), cudaMemcpyDeviceToHost));

    //Min    
    reduce_minm_maxm<<<grid_size, block_size, block_size*sizeof(float)>>>(d_temp, d_logLuminance, kMinimize);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    
    float* d_min_logLum;
    checkCudaErrors(cudaMalloc(&d_min_logLum, sizeof(float)));
    reduce_minm_maxm<<<1, grid_size, grid_size*sizeof(float)>>>(d_min_logLum, d_temp, kMinimize);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaMemcpy(&min_logLum, d_min_logLum, sizeof(float), cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaFree(d_temp));         //cleanup

    //printf("%.6f  %.6f\n", min_logLum, max_logLum); //max_logLum: 2.265088  min_logLum: -3.109206 
  /*2) subtract them to find the range */

    float lumRange = max_logLum - min_logLum;

    /*3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins  */

    checkCudaErrors(cudaMemset(d_cdf, 0, sizeof(unsigned int)* numBins));
    atomic_histogram<<<grid_size, block_size>>>(d_cdf, d_logLuminance, min_logLum, lumRange, numBins);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());

    /*4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */

    blelloch_scan<<<1, numBins, sizeof(unsigned int)*numBins>>>(d_cdf, numBins);
    cudaDeviceSynchronize();  checkCudaErrors(cudaGetLastError());

}

