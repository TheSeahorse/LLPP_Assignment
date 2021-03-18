// #include "ped_model.h"

// void updateHeatFade(int *heatmap, int size);
// void updateHeatIntensity(int *heatmap, int *xArray, int *yArray, int agent_size, int size);
// void updateSetMaxHeat(int *heatmap, int size);
// void updateScaledHeatmap(int *heatmap, int *scaledHeatmap, int size, int cellSize);
// void updateBlurredHeatmap(int *scaledHeatmap, int *blurredHeatmap, int scaledSize);
void initHeatmaps(int size, int scaledSize, int agentSize, int *heatmap);
void freeHeatmaps();
void cudaUpdateHeatmap(int *heatmap, int *x, int *y, int agent_size, int *scaledHeatmap, int size, int cellSize, int *blurredHeatmap, int scaledSize);