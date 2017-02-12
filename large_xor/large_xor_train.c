#include <stdio.h>
#include "fann.h"

int FANN_API test_callback(
    struct fann *ann, struct fann_train_data *train, unsigned int max_epochs,
    unsigned int reports_each, float desired_error, unsigned int epochs
) {

    printf("Epochs     %8d. MSE: %.5f. Desired-MSE: %.5f\n", epochs, fann_get_MSE(ann), desired_error);
    return 0;

}

int main() {

    printf("// ----------------------------------------------------\n");
    fann_type *calc_out;
    const unsigned int num_layers   = 3;
    const unsigned int num_input    = 10;
    const unsigned int num_hidden   = 6;
    const unsigned int num_output   = 1;

    const float desired_error       = (const float) 0;
    const unsigned int max_epochs   = 1000;
    const unsigned int reports_each = 1;

    struct fann *ann;
    struct fann_train_data *data;

    unsigned int i = 0;
    unsigned int decimal_point;

    float AVG = 0;

    printf("// ----------------------------------------------------\n");
    printf("Creating network.\n");
    ann = fann_create_standard(num_layers, num_input, num_hidden, num_output);

    data = fann_read_train_from_file("./dataset/large_xor.data");

    fann_set_activation_steepness_hidden(ann, 1);
    fann_set_activation_steepness_output(ann, 1);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
    fann_set_bit_fail_limit(ann, 0.01f);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_init_weights(ann, data);

    printf("// ----------------------------------------------------\n");
    printf("Training network.\n");
    fann_train_on_data(ann, data, max_epochs, reports_each, desired_error);

    // return 0;
    printf("// ----------------------------------------------------\n");
    printf("Testing network. %f\n", fann_test_data(ann, data));

    unsigned int train_data_length = fann_length_train_data(data);

    for (i = 0; i < train_data_length; i++) {
        calc_out = fann_run(ann, data->input[i]);


        float _avg = fann_abs(calc_out[0] - data->output[i][0]);

        printf( "Test: [%f should be %f] diff: [%f]\n", calc_out[0], data->output[i][0], _avg );


        AVG += _avg;

        /*
        printf(
            "Test: (%f,%f,%f,%f,%f,%f,%f,%f,%f,%f) -> %f, should be %f, difference=%f\n",
            data->input[i][0], data->input[i][1], data->input[i][2], data->input[i][3],
            data->input[i][4], data->input[i][5], data->input[i][6], data->input[i][7],
            data->input[i][8], data->input[i][9],
            calc_out[0], data->output[i][0], fann_abs(calc_out[0] - data->output[i][0])
        );
        */



    }

    printf("// ----------------------------------------------------\n");
    printf("AVG result: [%f] : RAW:[%f] of length:[%d]\n", (AVG / train_data_length), AVG, train_data_length );


    printf("// ----------------------------------------------------\n");
    printf("Saving network.\n");

    printf("Saving :[xor_float.net]\n");
    fann_save(ann, "./net/xor_float.net");

    decimal_point = fann_save_to_fixed(ann, "./net/xor_fixed.net");
    printf("Saving :[xor_fixed.net]\n");
    fann_save_train_to_fixed(data, "./dataset/xor_fixed.data", decimal_point);

    printf("Cleaning up.\n");
    fann_destroy_train(data);
    fann_destroy(ann);

    printf("// ----------------------------------------------------\n");
    return 0;
}
