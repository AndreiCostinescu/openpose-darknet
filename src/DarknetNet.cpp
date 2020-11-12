#include <cstdio>
#include <cstdlib>
#include <DarknetNet.h>
#include <darknet/network.h>
#include <darknet/utils/option_list.h>

using namespace std;

DarknetNet::DarknetNet(char *cfgFile, char *weightFile, char *dataFile) :
        cfg(cfgFile), weight(weightFile), data(dataFile)  {}

void DarknetNet::initNet(int benchmarkLayers) {
    this->options = read_data_cfg(this->data);
    char *name_list = option_find_str(this->options, (char *) "names", (char *) "data/names.list");
    int names_size = 0;
    this->names = get_labels_custom(name_list, &names_size);  // get_labels(name_list);

    this->alphabet = load_alphabet();

    this->net = load_network_custom_verbose(this->cfg, this->weight, 0, 1, 0);
    set_batch_network(this->net, 1);

    this->net->benchmark_layers = benchmarkLayers;
    calculate_binary_weights(*this->net);

    for (int layerNumber = 0; layerNumber < this->net->n; ++layerNumber) {
        layer _layer = this->net->layers[layerNumber];
        if (_layer.type == YOLO || _layer.type == GAUSSIAN_YOLO || _layer.type == REGION) {
            this->detectionLayerIndices.push_back(layerNumber);
            this->lastDetectionLayer = _layer;
            printf("Detection layer: %d - type = %d\n", layerNumber, this->lastDetectionLayer.type);
        }
    }

    if (this->lastDetectionLayer.classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
               name_list, names_size, this->lastDetectionLayer.classes, this->cfg);
        if (this->lastDetectionLayer.classes > names_size) {
            getchar();
        }
    }
    srand(2222222);

    this->inW = this->net->w;
    this->inH = this->net->h;
    this->outW = this->net->layers[this->net->n - 2].out_w;
    this->outH = this->net->layers[this->net->n - 2].out_h;
}

float *DarknetNet::runNet(float *inData) {
    network_predict(*this->net, inData);
    return this->net->output;
}

void DarknetNet::releaseNet() {
    free_ptrs((void **) this->names, this->lastDetectionLayer.classes);
    free_list_contents_kvp(this->options);
    free_list(this->options);

    int i, j;
    for (j = 0; j < 8; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(this->alphabet[j][i]);
        }
        free(this->alphabet[j]);
    }
    free(this->alphabet);

    free_network(*this->net);
}