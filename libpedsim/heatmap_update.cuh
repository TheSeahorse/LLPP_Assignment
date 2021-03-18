// #include "ped_model.h"

void allocate(int **heatmap, int **blurred_heatmap, int size, int scaledSize);
void update_heatmap(int **heatmap, int **scaled_heatmap, int **blurred_heatmap, int *hm, int *shm, int *bhm, float* destinationsX, float* destinationsY, int size, int scaledSize, int agentSize);
void free_cuda();