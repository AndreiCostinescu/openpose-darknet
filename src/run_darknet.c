#include <darknet.h>
#include <network.h>

static network *net;

void initNet(char *cfgFile, char *weightFile, int *inW, int *inH, int *outW, int *outH) {
    net = load_network_verbose(cfgFile, weightFile, 0, 0);
    set_batch_network(net, 1);
    *inW = net->w;
    *inH = net->h;
    *outW = net->layers[net->n - 2].out_w;
    *outH = net->layers[net->n - 2].out_h;
}

float *runNet(float *inData) {
    network_predict(*net, inData);
    return net->output;
}
