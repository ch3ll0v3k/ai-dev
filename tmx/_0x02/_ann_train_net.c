#include <stdio.h>
#include <floatfann.h>
#include <unistd.h>

// ====================================================================
// http://leenissen.dk/fann/html/files/fann_train-h.html#fann_get_learning_momentum
// ====================================================================
void getToMinError();

// ====================================================================
struct fann *ann;

unsigned int fst = 0;

char net_name[255];
char data_name[255];

// -------------------------------------
float dest_error = 0.02;
float ABS_error = 0.00002;
float momentum = 0.1;


// -------------------------------------

int main(int argc, char const *argv[]) {

    // -------------------------------------------------
    if ( argc < 3 ) {
        printf(" Name of network || Name of train set: missing\n");
        exit(-1);

    }

    sprintf( net_name, "./net/%s.net", argv[1] );
    sprintf( data_name, "./data/%s.data", argv[2] );

    ann = fann_create_from_file( net_name );

    // momentum = fann_get_learning_momentum( ann ); // printf( "momentum: [%f]\n", momentum );
    fann_set_learning_momentum( ann, momentum );

    // * float fann_get_MSE( *ann );
    // * unsigned int fann_get_bit_fail( *ann );


    dest_error = 0.001;
    // -------------------------------------------------
    while ( dest_error > ABS_error ) {
        getToMinError();

    }

    fann_destroy(ann);
    // -------------------------------------------------
    return 0;
}

// ====================================================================
void getToMinError() {

    // -------------------------------------------------
    printf(" --------------------------------------------------------- \n");
    printf(" ANN: New session: DEST-ERROR(%f), \n\n", dest_error);

    const unsigned int num_input = 288;
    const unsigned int num_output = 1;
    const unsigned int num_layers = 3;
    const unsigned int num_neurons_hidden = 324;
    float desired_error = dest_error; // 0.003; //0.0000001;
    const unsigned int max_epochs = 5;
    const unsigned int epochs_between_reports = 1;


    fann_train_on_file(ann, data_name, max_epochs, epochs_between_reports, desired_error);

    // -------------------------------------------------
    // fann_type input[2] = { -1, 1 };
    // fann_type *calc_out = fann_run(ann, input);

    // printf(" #%s: (%f,%f) -> %f\n", net_name, input[0], input[1], calc_out[0]);

    if ( fst ) {
        fann_save(ann, net_name);
        dest_error /= 2;
    } else {
        fst = 1;
    }

    printf(" ::::: [%f] :::::\n", dest_error);
    printf(" Cooling down [10] sec ... \n");
    sleep(10);

    // -------------------------------------------------

}

// ====================================================================
