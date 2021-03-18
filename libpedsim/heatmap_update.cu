#include "heatmap_update.cuh"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <omp.h>

int* d_heatmap;
int* d_blurred_heatmap;
int* d_scaled_heatmap;
int** d_hm;
int** d_bhm;
int** d_shm;
float* d_destinationsX;
float* d_destinationsY;

__global__
void updateHeatmap(int **d_hm, float *destinationX, float *destinationY, int agentSize, int size)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  if (row > 1024 || col > 1024) return;
  d_hm[row][col] = (int)round(d_hm[row][col] * 0.8);

  if (col < agentSize && row == col) 
    {
      int x = destinationX[row];
      int y = destinationY[col];

      if (!(x < 0 
	    || x >= size 
	    || y < 0 
	    || y >= size))
	{
	  d_hm[y][x] += 40;
	}
    }	
  d_hm[row][col] = d_hm[row][col] < 255 ? d_hm[row][col] : 255;
}


__global__
void scaleHeatmap(int **d_hm, int **d_shm, int cellSize)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  if (row > 1024 || col > 1024) return;

  int value = d_hm[row][col];
  for (int cellY = 0; cellY < cellSize; cellY++)
    {
      for (int cellX = 0; cellX < cellSize; cellX++)
	{
	  d_shm[row * cellSize + cellY][col * cellSize + cellX] = value;
	}
    }
}


__global__
void blurHeatmap(int **d_shm, int **d_bhm, int scaledSize)
{
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;
  __shared__ int s[32][32];

  if (row > scaledSize || col > scaledSize) return;


  // Weights for blur filter
  const int w[5][5] = {
    { 1, 4, 7, 4, 1 },
    { 4, 16, 26, 16, 4 },
    { 7, 26, 41, 26, 7 },
    { 4, 16, 26, 16, 4 },
    { 1, 4, 7, 4, 1 }
  };

#define WEIGHTSUM 273
  int sum = 0;
  s[threadIdx.y][threadIdx.x] = d_shm[row][col];
  __syncthreads();
  for (int k = -2; k < 3; k++)
    {
      for (int l = -2; l < 3; l++)
	{
	  if (row <= 2 
	      || row >= scaledSize - 2 
	      || col <= 2 
	      || col >= scaledSize - 2) 
	    {
	      // pass
	    }
	  else if (threadIdx.x >= 29 
		   || threadIdx.x <= 1 
		   || threadIdx.y >= 29 
		   || threadIdx.y <= 1) 
	    {
	      sum += w[2 + k][2 + l] * d_shm[row + k][col + l];
	    }
	  else 
	    {
	      sum += w[2 + k][2 + l] * s[threadIdx.y + k][threadIdx.x + l];
	    }
	}
    }
  int value = sum / WEIGHTSUM;
  d_bhm[row][col] = 0x00FF0000 | value << 24;
}


__global__ 
void setup(int **heatmap, int **scaled_heatmap, int **blurred_heatmap, int *hm, int *shm, int *bhm, int size, int scaledSize) 
{
  for (int i = 0; i < size; i++)
    {
      heatmap[i] = hm + size * i;
    }
  for (int i = 0; i < scaledSize; i++)
    {
      scaled_heatmap[i] = shm + scaledSize * i;
      blurred_heatmap[i] = bhm + scaledSize * i;
    }
}

void allocate(int **heatmap, int **blurred_heatmap, int size, int scaledSize)
{
  cudaHostAlloc((void **)blurred_heatmap, scaledSize*scaledSize*sizeof(int), cudaHostAllocDefault);

  cudaMalloc(&d_heatmap, size*size * sizeof(int));
  cudaMalloc(&d_blurred_heatmap, scaledSize*scaledSize * sizeof(int));
  cudaMalloc(&d_scaled_heatmap, scaledSize*scaledSize * sizeof(int));

  cudaMalloc(&d_hm, size * sizeof(int*));
  cudaMalloc(&d_bhm, scaledSize * sizeof(int*));
  cudaMalloc(&d_shm, scaledSize * sizeof(int*));

  cudaMalloc(&d_destinationsX, size * sizeof(float));
  cudaMalloc(&d_destinationsY, size * sizeof(float));

  cudaMemset(d_heatmap, 0, size*size * sizeof(int));
  cudaMemset(d_blurred_heatmap, 0, scaledSize*scaledSize * sizeof(int));
  cudaMemset(d_scaled_heatmap, 0, scaledSize*scaledSize * sizeof(int));

  cudaMemcpy(d_heatmap, *heatmap, size*size * sizeof(int), cudaMemcpyHostToDevice);
}

void update_heatmap(int **heatmap, int **scaled_heatmap, int **blurred_heatmap, int *hm, int *shm, int *bhm, float *destinationsX, float *destinationsY, int size, int scaledSize, int agentSize)
{
  dim3 dimBlock(32, 32);
  dim3 dimGrid(32, 32);
  dim3 dimGridBlur(160, 160);
  
  cudaMemcpy(d_destinationsX, destinationsX, sizeof(int)*agentSize, cudaMemcpyHostToDevice);
  cudaMemcpy(d_destinationsY, destinationsY, sizeof(int)*agentSize, cudaMemcpyHostToDevice);

  setup <<<1,1>>> (d_hm, d_shm, d_bhm, d_heatmap, d_scaled_heatmap, d_blurred_heatmap, size, scaledSize);

  updateHeatmap <<< dimGrid, dimBlock>>>(d_hm, d_destinationsX, d_destinationsY, agentSize, size);

  scaleHeatmap <<< dimGrid, dimBlock >> >(d_hm, d_shm, 5);

  blurHeatmap <<<dimGridBlur ,dimBlock>>> (d_shm, d_bhm, scaledSize);

  cudaMemcpyAsync(*blurred_heatmap, d_blurred_heatmap, scaledSize*scaledSize * sizeof(int), cudaMemcpyDeviceToHost);
}

void free_cuda()
{
  cudaFree(d_heatmap);
  cudaFree(d_scaled_heatmap);
  cudaFree(d_blurred_heatmap);
  cudaFree(d_destinationsX);
  cudaFree(d_destinationsY);
}
