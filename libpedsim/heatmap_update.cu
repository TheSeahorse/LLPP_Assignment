#include "heatmap_update.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <omp.h>
#include <chrono> 

int* heatmap;
int* scaledHeatmap;
int* blurredHeatmap;
int* d_heatmap;
int* d_blurredHeatmap;
int* d_scaledHeatmap;
int* d_x;
int* d_y;



void initHeatmaps(int SIZE, int scaledSize, int agent_size, int *heatmap)
{
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  cudaMalloc((void **)&d_scaledHeatmap, scaledSize*scaledSize*sizeof(int));
  cudaMalloc((void **)&d_blurredHeatmap, scaledSize*scaledSize*sizeof(int));
  cudaMalloc((void **)&d_x, agent_size*sizeof(int));
  cudaMalloc((void **)&d_y, agent_size*sizeof(int));

  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);

  cudaMemset(d_heatmap, 0, SIZE*SIZE * sizeof(int));
  cudaMemset(d_scaledHeatmap, 0, scaledSize*scaledSize * sizeof(int));
  cudaMemset(d_blurredHeatmap, 0, scaledSize*scaledSize * sizeof(int));
}

void freeHeatmaps()
{
  cudaFree(d_heatmap);
  cudaFree(d_blurredHeatmap);  
  cudaFree(d_scaledHeatmap);  
  cudaFree(d_x);
  cudaFree(d_y);
}

__global__
void updateHeat(int *d_heatmap, int *x, int *y, int agent_size, int size)
{
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int index = row * size + col;
  if(col > size || row > size)
    {
      return;
    }
  else
    {
      d_heatmap[row*size + col] = (int)round(d_heatmap[row*size + col] * 0.80);
      if (index >= 0 && index < agent_size)
        {
          atomicAdd(&d_heatmap[y[index]*size + x[index]], 40);
        }	
      if (d_heatmap[index] >= 255)
        {
	  d_heatmap[index] = 255;
        }   
    }
}


__global__
void scaleHeatmap(int *d_heatmap, int *d_scaledHeatmap, int size, int cellSize)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < size*size)
    {
      for (int cellY = 0; cellY < cellSize; cellY++)
	    {
        int scaledBlock = (blockIdx.x * cellSize + cellY) * blockDim.x * cellSize;
        for (int cellX = 0; cellX < cellSize; cellX++)
        {
          int scaledThread = threadIdx.x * cellSize + cellX;
          d_scaledHeatmap[scaledBlock + scaledThread] = d_heatmap[index];
        }
	    }
    }
}



__global__
void blurHeatmap(int *d_scaledHeatmap, int *d_blurredHeatmap, int scaledSize)
{
  const int tileSize = 32;
#define WEIGHTSUM 273
  const int w[5][5] = {
    { 1, 4, 7, 4, 1 },
    { 4, 16, 26, 16, 4 },
    { 7, 26, 41, 26, 7 },
    { 4, 16, 26, 16, 4 },
    { 1, 4, 7, 4, 1 }
  };
  int ty, tx; ty = threadIdx.y; tx = threadIdx.x;
  int col; col = blockIdx.x*blockDim.x + tx;
  int row; row = blockIdx.y*blockDim.y + ty;
  if (blockIdx.y*blockDim.y + ty > scaledSize || blockIdx.x*blockDim.x + tx > scaledSize) return;
  __shared__ int sTile[tileSize][tileSize];

  int sum = 0;
  sTile[threadIdx.y][threadIdx.x] = d_scaledHeatmap[row*scaledSize+col];
  __syncthreads();
  for (int k = -2; k < 3; k++)
    {
      for (int l = -2; l < 3; l++)
	{   
          if (threadIdx.x <= 29 && threadIdx.x > 1 
	      && threadIdx.y <= 29 && threadIdx.y > 1) {   
            sum += w[2 + k][2 + l] * sTile[threadIdx.y + k][threadIdx.x + l];
            
          }
	  else
	    {
	      if(row > 2 && row < scaledSize-2 && col > 2 && col < scaledSize-2)
		{
		  sum += w[2 + k][2 + l] * d_scaledHeatmap[(row*scaledSize + scaledSize*k) + (col + l)]; 
		}
	    }
	}
    }
      
  int value = sum / WEIGHTSUM;  
  d_blurredHeatmap[row * scaledSize + col] = 0x00FF0000 | value << 24;
}


  void cudaUpdateHeatmap(int *heatmap, int *x, int *y, int agent_size, int *scaledHeatmap, int SIZE, int cellSize, int *blurredHeatmap, int scaledSize)
{
  int THREADSPERBLOCK = 1024;
  int tileSize = 32;
  
  dim3 dimBlock(tileSize, tileSize);
  dim3 dimGrid(tileSize, tileSize);
  dim3 blurDimGrid(scaledSize/tileSize, scaledSize/tileSize);

  cudaMemcpyAsync(d_x, x, agent_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(d_y, y, agent_size*sizeof(int), cudaMemcpyHostToDevice);
  
  updateHeat<<<dimGrid, dimBlock>>>(d_heatmap, d_x, d_y, agent_size, SIZE);

  scaleHeatmap<<<(SIZE*SIZE)/THREADSPERBLOCK,THREADSPERBLOCK>>>(d_heatmap, d_scaledHeatmap, SIZE, cellSize);

  blurHeatmap<<<blurDimGrid, dimBlock>>>(d_scaledHeatmap, d_blurredHeatmap, scaledSize);

  cudaMemcpyAsync(blurredHeatmap, d_blurredHeatmap, scaledSize*scaledSize*sizeof(int), cudaMemcpyDeviceToHost);
}
