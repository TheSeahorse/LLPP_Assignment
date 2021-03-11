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
    
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < size*size)
    {
        d_heatmap[row*size+col] = (int)round(d_heatmap[row*size+col] * 0.80);
    }
}

__global__
void heatIntensify(int *d_heatmap, int *x, int *y, int agent_size, int size)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x; 
    if (col >= 0 && col < agent_size)
    {
      atomicAdd(&d_heatmap[y[col]*size + x[col]], 100);
    }	
}

__global__
void setMaxHeat(int *d_heatmap, int size)
{
    int row = blockIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (d_heatmap[row*size+col] >= 255)
    {
        d_heatmap[row*size+col] = 255;
    }
}

// __global__
// void scaleHeatmap(int *d_heatmap, int *d_scaledHeatmap, int size, int cellSize)
// {
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;
//     for (int cellY = 0; cellY < cellSize; cellY++)
//     {
//         for (int cellX = 0; cellX < cellSize; cellX++)
//         {
//             d_scaledHeatmap[row * cellSize + cellY][col * cellSize + cellX] = d_heatmap[row*size+col];
//         }
//     }
// }

int THREADSPERBLOCK = 256;
// Updates the heatmap according to the agent positions
void updateHeatFade(int *heatmap, int SIZE)
{
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
  THREADSPERBLOCK = 32;
  int *d_heatmap, *d_x, *d_y;
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  cudaMalloc((void **)&d_x, agent_size*sizeof(int));
  cudaMalloc((void **)&d_y, agent_size*sizeof(int));

  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_x, x, agent_size*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, y, agent_size*sizeof(int), cudaMemcpyHostToDevice);

  heatIntensify<<<((agent_size)/THREADSPERBLOCK)+1,THREADSPERBLOCK>>>(d_heatmap, d_x, d_y, agent_size, SIZE);

  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(x, d_x, agent_size*sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(y, d_y, agent_size*sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_heatmap);
}

void updateSetMaxHeat(int *heatmap, int SIZE)
{
  int *d_heatmap;
  cudaMalloc((void **)&d_heatmap, SIZE*SIZE*sizeof(int));
  cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
  setMaxHeat<<<((SIZE*SIZE)/THREADSPERBLOCK)+1,THREADSPERBLOCK>>>(d_heatmap, SIZE);
  cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
  cudaFree(d_heatmap);
}

// void updateScaledHeatmap(int *heatmap, int *scaledHeatmap, int SIZE, int cellSize)
// {
//     std::cout << "setMax\n";
//     int *d_heatmap;
//     int *d_scaledHeatmap;
//     cudaMalloc(&d_heatmap, SIZE*SIZE*sizeof(int));
//     cudaMalloc(&d_scaledHeatmap, SIZE*SIZE*sizeof(int));
// 	cudaMemcpy(d_heatmap, heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_scaledHeatmap, scaledHeatmap, SIZE*SIZE*sizeof(int), cudaMemcpyHostToDevice);
// 	scaleHeatmap<<<4,256>>>(d_heatmap, d_scaledHeatmap, SIZE, cellSize);
// 	cudaMemcpy(heatmap, d_heatmap, SIZE*SIZE*sizeof(int), cudaMemcpyDeviceToHost);
//     cudaFree(d_heatmap);  
// }
	// for (int x = 0; x < SIZE; x++)
	// {
	// 	for (int y = 0; y < SIZE; y++)
	// 	{
	// 		// heat fades
	// 		heatmap[y][x] = (int)round(heatmap[y][x] * 0.80);
	// 	}
	// }

// 	// Count how many agents want to go to each location
	// for (int i = 0; i < agents.size(); i++)
	// {
	// 	Ped::Tagent* agent = agents[i];
	// 	int x = agent->getDesiredX();
	// 	int y = agent->getDesiredY();

	// 	if (x < 0 || x >= SIZE || y < 0 || y >= SIZE)
	// 	{
	// 		continue;
	// 	}

	// 	else// intensify heat for better color results
	// 	{
	// 		heatmap[y][x] += 40;
	// 	}
		

// 	}

// 	for (int x = 0; x < SIZE; x++)
// 	{
// 		for (int y = 0; y < SIZE; y++)
// 		{
// 			heatmap[y][x] = heatmap[y][x] < 255 ? heatmap[y][x] : 255;
// 		}
// 	}

// 	// Scale the data for visual representation
// 	for (int y = 0; y < SIZE; y++)
// 	{
// 		for (int x = 0; x < SIZE; x++)
// 		{
// 			int value = heatmap[y][x];
// 			for (int cellY = 0; cellY < CELLSIZE; cellY++)
// 			{
// 				for (int cellX = 0; cellX < CELLSIZE; cellX++)
// 				{
// 					scaled_heatmap[y * CELLSIZE + cellY][x * CELLSIZE + cellX] = value;
// 				}
// 			}
// 		}
// 	}

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
