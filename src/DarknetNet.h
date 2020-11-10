#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

#include <darknet.h>
#include <images/image.h>
#include <iostream>
#include <vector>

class DarknetNet {
public:
    DarknetNet();

    void initNet(int benchmarkLayers, int *inW = nullptr, int *inH = nullptr, int *outW = nullptr, int *outH = nullptr,
                 network **_net = nullptr, image ***_alphabet = nullptr, char ***_names = nullptr,
                 layer *_lastDetectionLayer = nullptr);

    float *runNet(float *inData);

    void releaseNet();

private:
    network *net;
    image **alphabet;
    char **names, *data, *cfg, *weight;
    std::vector<int> detectionLayerIndices;
    layer lastDetectionLayer;
    list *options;
};

#endif // RUN_DARKNET_H
