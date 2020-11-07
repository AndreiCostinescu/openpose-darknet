#include <network.h>
#include <utils/option_list.h>
#include <utils/data.h>
#include <images/image.h>
#include <images/image_opencv.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace std;

static network *net;
static image **alphabet;
static char **names;
static vector<int> detectionLayerIndices;
static layer lastDetectionLayer;
static list *options;

void initNet(char *dataFile, char *cfgFile, char *weightFile, int benchmarkLayers,
             int *inW, int *inH, int *outW, int *outH, network **_net, image ***_alphabet,
             char ***_names, layer *_lastDetectionLayer) {
    options = read_data_cfg(dataFile);
    char *name_list = option_find_str(options, "names", "data/names.list");
    int names_size = 0;
    names = get_labels_custom(name_list, &names_size);  // get_labels(name_list);
    if (_names != nullptr) {
        *_names = names;
    }

    alphabet = load_alphabet();
    if (_alphabet != nullptr) {
        *_alphabet = alphabet;
    }

    net = load_network_custom_verbose(cfgFile, weightFile, 0, 1, 0);
    if (_net != nullptr) {
        *_net = net;
    }
    set_batch_network(net, 1);

    net->benchmark_layers = benchmarkLayers;
    calculate_binary_weights(*net);

    for (int layerNumber = 0; layerNumber < net->n; ++layerNumber) {
        layer _layer = net->layers[layerNumber];
        if (_layer.type == YOLO || _layer.type == GAUSSIAN_YOLO || _layer.type == REGION) {
            detectionLayerIndices.push_back(layerNumber);
            lastDetectionLayer = _layer;
            printf("Detection layer: %d - type = %d\n", layerNumber, lastDetectionLayer.type);
        }
    }
    if (_lastDetectionLayer != nullptr) {
        *_lastDetectionLayer = lastDetectionLayer;
    }

    if (lastDetectionLayer.classes != names_size) {
        printf("\n Error: in the file %s number of names %d that isn't equal to classes=%d in the file %s \n",
               name_list, names_size, lastDetectionLayer.classes, cfgFile);
        if (lastDetectionLayer.classes > names_size) {
            getchar();
        }
    }
    srand(2222222);

    if (inW != nullptr) {
        *inW = net->w;
    }
    if (inH != nullptr) {
        *inH = net->h;
    }
    if (outW != nullptr) {
        *outW = net->layers[net->n - 2].out_w;
    }
    if (outH != nullptr) {
        *outH = net->layers[net->n - 2].out_h;
    }
}

float *runNet(float *inData) {
    network_predict(*net, inData);
    return net->output;
}

void releaseNet() {
    free_ptrs((void **) names, lastDetectionLayer.classes);
    free_list_contents_kvp(options);
    free_list(options);

    int i, j;
    for (j = 0; j < 8; ++j) {
        for (i = 32; i < 127; ++i) {
            free_image(alphabet[j][i]);
        }
        free(alphabet[j]);
    }
    free(alphabet);

    free_network(*net);
}