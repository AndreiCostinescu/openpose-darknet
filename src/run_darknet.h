#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#include <darknet.h>
#include <images/image.h>

void initNet(char *dataFile, char *cfgFile, char *weightFile, int benchmarkLayers,
             int *inW = nullptr, int *inH = nullptr, int *outW = nullptr, int *outH = nullptr, network **_net = nullptr,
             image ***_alphabet = nullptr, char ***_names = nullptr, layer *_lastDetectionLayer = nullptr);

float *runNet(float *inData);

void releaseNet();

#endif // RUN_DARKNET_H
