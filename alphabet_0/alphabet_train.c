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
    const unsigned int num_input    = 900;
    const unsigned int num_output   = 26;

    const unsigned int num_hidden   = (num_input + num_output) / 2 +1;

    const float desired_error       = (const float) 0;
    const unsigned int max_epochs   = 1;
    const unsigned int reports_each = 1;

    struct fann *ann;
    struct fann_train_data *data;

    unsigned int i = 0;
    unsigned int decimal_point;

    char dataset_fixed[] = "./dataset/alphabet_fixed.data";
    char dataset_fload[] = "./dataset/alphabet_float.data";
    char net_fixed[] = "./net/alphabet_fixed.net";
    char net_float[] = "./net/alphabet_float.net";

    float AVG = 0;

    printf("// ----------------------------------------------------\n");

    char MK_NEW = 0;


    if ( MK_NEW ) {
        printf("Creating network.\n");
        ann = fann_create_standard(num_layers, num_input, num_hidden, num_output);

    } else {

        printf("Reading network.\n");
        ann = fann_create_from_file( net_float );
        // ann = fann_create_from_file("./net/alphabet_fixed.net");

    }

    if ( !ann ) {
        printf("Error configuring net: [%d]\n", ann);
        return 0;

    }

    printf("done ...\n\n");

    printf("// ----------------------------------------------------\n");
    printf("Reading dataset: [%s]\n", dataset_fload);

    // data = fann_read_train_from_file( dataset_fixed );
    data = fann_read_train_from_file( dataset_fload );

    printf("done ...\n\n");

    printf("Config: [%s]\n");
    fann_set_activation_steepness_hidden(ann, 1);
    fann_set_activation_steepness_output(ann, 1);

    fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
    fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

    fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
    fann_set_bit_fail_limit(ann, 0.01f);
    fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    fann_init_weights(ann, data);
    printf("done ...\n\n");

    printf("// ----------------------------------------------------\n");
    printf("Training network.\n");
    fann_train_on_data(ann, data, max_epochs, reports_each, desired_error);
    printf("done ...\n\n");

    // return 0;
    printf("// ----------------------------------------------------\n");
    printf("Testing network. %f\n", fann_test_data(ann, data));

    unsigned int train_data_length = fann_length_train_data(data);

    train_data_length = 10;

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

    printf("// ::::::::::::::::::::::::: \n");
    printf("AVG result: [%f] : RAW:[%f] of length:[%d]\n", (AVG / train_data_length), AVG, train_data_length );

    printf("done ...\n\n");

    printf("// ----------------------------------------------------\n");
    printf("Saving network.\n");

    printf("Saving :[%s]\n", net_float);
    fann_save(ann, net_float);
    printf("done ...\n\n");


    if ( 0 ) {
        printf("Saving :[%s]\n", net_fixed);
        decimal_point = fann_save_to_fixed(ann, net_fixed);
        fann_save_train_to_fixed(data, dataset_fixed, decimal_point);
        printf("done ...\n\n");

    }



    printf("Cleaning up.\n");
    fann_destroy_train(data);
    fann_destroy(ann);
    printf("done ...\n\n");

    printf("// ----------------------------------------------------\n");
    printf("END ...\n\n");
    return 0;
}
