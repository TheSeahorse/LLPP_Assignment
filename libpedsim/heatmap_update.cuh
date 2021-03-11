void updateHeatFade(int *heatmap, int size);
void updateHeatIntensity(int *heatmap, int *x, int *y, int agent_size, int size);
void updateSetMaxHeat(int *heatmap, int size);
void updateScaledHeatmap(int *heatmap, int *scaledHeatmap, int size, int cellSize);
void updateBlurredHeatmap(int *scaledHeatmap, int *blurredHeatmap, int scaledSize);