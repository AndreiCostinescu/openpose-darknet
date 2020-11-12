#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#include <darknet/darknet.h>
#include <darknet/images/image.h>
#include <darknet/utils/list.h>
#include <iostream>
#include <vector>

class DarknetNet {
public:
    DarknetNet(char *cfgFile, char *weightFile, char *dataFile = nullptr);

    void initNet(int benchmarkLayers);

    float *runNet(float *inData);

    void releaseNet();

    int inW{}, inH{}, outW{}, outH{};
    network *net{};
    image **alphabet{};
    char **names{}, *data, *cfg, *weight;
    std::vector<int> detectionLayerIndices;
    layer lastDetectionLayer{};
    list *options{};
};

#endif // RUN_DARKNET_H
