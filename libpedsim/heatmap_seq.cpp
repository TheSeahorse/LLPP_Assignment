// Created for Low Level Parallel Programming 2017
//
// Implements the heatmap functionality. 
//
#include "ped_model.h"
#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <cuda.h>
#include "device_launch_parameters.h"
#include <cstdlib>
#include <iostream>
#include <cmath>
#include <omp.h>
#include "heatmap_update.cuh"
using namespace std;

// Memory leak check with msvc++
#include <stdlib.h>

// Sets up the heatmap
void Ped::Model::setupHeatmapSeq()
{
  __shared__ hm = (int*)calloc(SIZE*SIZE, sizeof(int));
  __shared__ shm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
  __shared__ bhm = (int*)malloc(SCALED_SIZE*SCALED_SIZE*sizeof(int));
  
  cudaHostAlloc((void **)&hm, SIZE*SIZE*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&shm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaHostAllocDefault);
  cudaHostAlloc((void **)&bhm, SCALED_SIZE*SCALED_SIZE*sizeof(int), cudaHostAllocDefault);
  // tickTaskBased(num_threads);
  heatmap = (int**)malloc(SIZE*sizeof(int*));
  scaled_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));
  blurred_heatmap = (int**)malloc(SCALED_SIZE*sizeof(int*));

  // #pragma omp parallel for
  // for (int i = 0; i < SIZE; i++)
  // {
  // 	heatmap[i] = hm + SIZE*i;
  // }

#pragma omp parallel for
  for (int i = 0; i < SCALED_SIZE; i++)
    {
      if(i < SIZE)
	{
	  heatmap[i] = hm + SIZE*i;
	}
      scaled_heatmap[i] = shm + SCALED_SIZE*i;
      blurred_heatmap[i] = bhm + SCALED_SIZE*i;
    }
}

// Updates the heatmap according to the agent positions
void Ped::Model::updateHeatmapSeq()
{   
  // printf("Starting heatmap\n");
  updateHeatFade(hm, SIZE);

  int x[agents.size()];
  int y[agents.size()];
  for (int i = 0; i < agents.size(); i++)
    {
      Ped::Tagent* agent = agents[i];
      x[i] = agent->getDesiredX();
      y[i] = agent->getDesiredY();
    }

  // test(hm, x, y, agents.size(), shm, SIZE, CELLSIZE, bhm, SCALED_SIZE, agents);

  updateHeatIntensity(hm, x, y, agents.size(), SIZE);
  updateSetMaxHeat(hm, SIZE);
  updateScaledHeatmap(hm, shm, SIZE, CELLSIZE);
  updateBlurredHeatmap(shm, bhm, SCALED_SIZE);
  cudaDeviceSynchronize();
  // printf("Finished heatmap\n");
}

int Ped::Model::getHeatmapSize() const {
  return SCALED_SIZE;
}
