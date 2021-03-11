// #include "ped_model.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <omp.h>

__global__
void fadeHeat(int *d_heatmap, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size*size)
    {
        d_heatmap[index] = (int)round(d_heatmap[index] * 0.80);
    }
}

__global__
void heatIntensify(int *d_heatmap, int *x, int *y, int agent_size, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x; 
    if (index >= 0 && index < agent_size)
    {
      atomicAdd(&d_heatmap[y[index]*size + x[index]], 150);
    }	
}

__global__
void setMaxHeat(int *d_heatmap, int size)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size*size)
    {
      if (d_heatmap[index] >= 255)
      {
          d_heatmap[index] = 255;
      }
    }
}


__global__
void scaleHeatmap(int *d_heatmap, int *d_scaledHeatmap, int size, int cellSize)
{
  int scaledSize = size*cellSize;
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size*size)
    {
      for (int cellY = 0; cellY < cellSize; cellY++)
	{
          for (int cellX = 0; cellX < cellSize; cellX++)
	    {
	      int scaledIndex = (blockIdx.x * cellSize + cellY) * blockDim.x *cellSize + (threadIdx.x * cellSize + cellX);
	      d_scaledHeatmap[scaledIndex] = d_heatmap[index];
	      
	      # if __CUDA_ARCH__>=200
	      //printf("cellY: %d, cellX: %d\n", cellY, cellX);
	      #endif
	    }
	}
    }
}

// Updates the heatmap according to the agent positions
void updateHeatFade(int *heatmap, int SIZE)
{
  int THREADSPERBLOCK = 256;
  // std::cout << "Fade\n";
  int *d_heatmap;
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  // printf("after malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  // printf("after HostToDevice: %s\n", cudaGetErrorString(cudaGetLastError()));
  fadeHeat<<<((SIZE*SIZE)/THREADSPERBLOCK)+1,THREADSPERBLOCK>>>(d_heatmap, SIZE);
  // printf("after fadeHeat: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  // printf("after DeviceToHost: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFree(d_heatmap);
  // printf("after Free: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void updateHeatIntensity(int *heatmap, int *x, int *y, int agent_size, int SIZE)
{
  int THREADSPERBLOCK = 32;
  int *d_heatmap, *d_x, *d_y;
  // printf("before malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  cudaMalloc((void **)&d_x, agent_size*sizeof(int));
  cudaMalloc((void **)&d_y, agent_size*sizeof(int));
  // printf("after malloc: %s\n", cudaGetErrorString(cudaGetLastError()));

  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, agent_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, agent_size*sizeof(int), cudaMemcpyHostToDevice);
  // printf("after memcpy: %s\n", cudaGetErrorString(cudaGetLastError()));
  heatIntensify<<<((agent_size)/THREADSPERBLOCK)+1,THREADSPERBLOCK>>>(d_heatmap, d_x, d_y, agent_size, SIZE);
  // printf("after function: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, agent_size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, agent_size*sizeof(int), cudaMemcpyDeviceToHost);
  // printf("after memcpy: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFree(d_heatmap);
  cudaFree(d_x);
  cudaFree(d_y);
  // printf("after free: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void updateSetMaxHeat(int *heatmap, int SIZE)
{
  int THREADSPERBLOCK = 256;
  int *d_heatmap;
  // printf("before malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  // printf("after malloc: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  // printf("after memcpy: %s\n", cudaGetErrorString(cudaGetLastError()));
  setMaxHeat<<<((SIZE*SIZE)/THREADSPERBLOCK)+1,THREADSPERBLOCK>>>(d_heatmap, SIZE);
  // printf("after function: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  // printf("after memcpy: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFree(d_heatmap);
  // printf("after free: %s\n", cudaGetErrorString(cudaGetLastError()));
}

void updateScaledHeatmap(int *heatmap, int *scaledHeatmap, int SIZE, int cellSize)
{
  int THREADSPERBLOCK = 256;
  int *d_heatmap;
  int *d_scaledHeatmap;
  // printf("before malloc1: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  // printf("after malloc1: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMalloc((void **)&d_scaledHeatmap, SIZE*SIZE*cellSize*cellSize*sizeof(int));
  // printf("after malloc2: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  // printf("after memcpy1: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(d_scaledHeatmap, scaledHeatmap, SIZE*SIZE*cellSize*cellSize*sizeof(int), cudaMemcpyHostToDevice);
  // printf("after memcpy2: %s\n", cudaGetErrorString(cudaGetLastError()));
  scaleHeatmap<<<(SIZE*SIZE)/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_heatmap, d_scaledHeatmap, SIZE, cellSize);
  // printf("after scaleHeatmap: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(scaledHeatmap, d_scaledHeatmap, SIZE*SIZE*cellSize*cellSize*sizeof(int), cudaMemcpyDeviceToHost);
  printf("after memecpy: %s\n", cudaGetErrorString(cudaGetLastError()));
  cudaFree(d_heatmap);  
  cudaFree(d_scaledHeatmap);  
  printf("after free: %s\n", cudaGetErrorString(cudaGetLastError()));
}

// 	// Weights for blur filter
// 	const int w[5][5] = {
// 		{ 1, 4, 7, 4, 1 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 7, 26, 41, 26, 7 },
// 		{ 4, 16, 26, 16, 4 },
// 		{ 1, 4, 7, 4, 1 }
// 	};

// #define WEIGHTSUM 273
// 	// Apply gaussian blurfilter		       
// 	for (int i = 2; i < SCALED_SIZE - 2; i++)
// 	{
// 		for (int j = 2; j < SCALED_SIZE - 2; j++)
// 		{
// 			int sum = 0;
// 			for (int k = -2; k < 3; k++)
// 			{
// 				for (int l = -2; l < 3; l++)
// 				{
// 					sum += w[2 + k][2 + l] * scaled_heatmap[i + k][j + l];
// 				}
// 			}
// 			int value = sum / WEIGHTSUM;
// 			blurred_heatmap[i][j] = 0x00FF0000 | value << 24;
// 		}
// 	}
// }
