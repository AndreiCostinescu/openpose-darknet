#ifndef RUN_DARKNET_H
#define RUN_DARKNET_H

extern "C" {

void initNet(char *cfgFile, char *weightFile, int *inW, int *inH, int *outW, int *outH);

float *runNet(float *inData);

};

#endif // RUN_DARKNET_H
