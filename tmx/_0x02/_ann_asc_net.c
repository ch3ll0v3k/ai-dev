#include <stdio.h>
#include <floatfann.h>

int main(int argc, char const *argv[]) {

    if ( argc < 2 ) {
        printf(" Name of network: missing\n");
        exit(-1);

    }

    char net_name[255];
    // sprintf( net_name, "./net/%s.net", argv[1] );
    sprintf( net_name, "%s", argv[1] );

    /*
    fann_type *calc_out;
    fann_type input[2];

    struct fann *ann = fann_create_from_file( net_name );

    input[0] = -1;
    input[1] = 1;
    calc_out = fann_run(ann, input);
    printf(" #%s: (%f,%f) -> %f\n", net_name, input[0], input[1], calc_out[0]);

    fann_destroy(ann);
    */

    printf(" Make it first , njeeeee ... \n" );

    return 0;
}