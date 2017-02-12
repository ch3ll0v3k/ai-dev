// #include "floatfann.h"
#include <fann.h>
#include <floatfann.h>


// inp.288-out.1-lay.6-hid.324
int main(int argc, char const *argv[]) {

    if ( argc < 2 ) {
        printf(" Name of new network: missing\n");
        exit(-1);

    }

    char net_name[255];
    // sprintf( net_name, "./net/%s.net", argv[1] );
    sprintf( net_name, "%s", argv[1] );

    const unsigned int num_input = 288;
    const unsigned int num_output = 1;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 324;
    const float desired_error = (const float) 0.0000001;
    const unsigned int max_epochs = 500000;
    const unsigned int epochs_between_reports = 1;

    struct fann *ann = fann_create_standard(num_layers, num_input, num_neurons_hidden, num_output);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);
    // fann_train_on_file(ann, "xor.data", max_epochs, epochs_between_reports, desired_error);
    fann_save(ann, net_name);
    fann_destroy(ann);

    return 0;

}


