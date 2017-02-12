#include "network.h"

// ====================================================================
double _F( double _in ) {

    // "fast sigmoid" function
    // f(x) = x / (1 + abs(x))

    if ( _in > 0 )
        return 1 / ( 1+exp( -_in) );

    return 1 / ( 1+exp( _in) );
    // return tanh( in );

}


// ====================================================================
double dataIn[] = { 0.5,0.3,  1};
double OUT_W[2] = { 4, -2 };
double ALL_W[2][4] = {
    { 2,  -3, 3,  2 }, { -1, -1, 2, -3 },
};

uint32_t useRand_W = 1;

NetWork NN;

int main(int argc, char *argv[]) {

    // ------------------------------------------------------------
    // * double d = 1.1;
    // * double d_0 = d / (1 + abs(d));
    // * double d_1 = _F( d );

    // * printf("d_0: %f\n", d_0);
    // * printf("d_1: %f\n", d_1);
    // * return 0;

    // ------------------------------------------------------------
    uint32_t INP_num  = 2; // i_w = [ 2, -3, 3,  2 ];
    uint32_t HID_num  = 2; // h_w = [ -1, -1, 2, -3 ];
    uint32_t LAYs_HID = 2;
    uint32_t OUT_num  = 1; // o_w = [ 4, -2 ];

    // ------------------------------------------------------------
    if ( NN_init( INP_num, HID_num, LAYs_HID, OUT_num )) {
        // if ( NN_init( 12, 14, 43, OUT_num )) {

        // TODO: Input from >>>
        // double data[2] = { 0 };

        NN_iter( dataIn );

    }

    printf(" [ main:END] \n");
    return 0;
    // ------------------------------------------------------------
}

// ====================================================================
double* NN_iter( double* dataIn ) {

    // ------------------------------------------------------------
    printf( "%s\n", line );
    printf( " NN_iter->START: \n" );

    double *result = (double *) malloc( sizeof(double) * NN.num_o );

    if ( !result)
        _break( "ERROR: (!result)" );

    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " INPUT >> HIDDEN \n" );

    for (int h_u = 0; h_u < NN.num_h; h_u++) {

        NN.H_layers[0].units[ h_u ].sum = 0.0;

        for (int i_u = 0; i_u < NN.num_i; i_u++) {

            if ( !useRand_W ) {
                double loc_sum = NN.H_layers[0].units[ h_u ].w[ i_u ] * dataIn[ i_u ];
                NN.H_layers[0].units[ h_u ].sum += loc_sum;
                printf("     IN:[%f] loc_sum:[%f]\n", dataIn[ i_u ], loc_sum);

            } else {
                double loc_sum = NN.H_layers[0].units[ h_u ].w[ i_u ] * getRand(0);
                NN.H_layers[0].units[ h_u ].sum += loc_sum;
                printf("     IN:[%f] loc_sum:[%f]\n", getRand(0), loc_sum);

            }

        }

        double func_res = _F( NN.H_layers[0].units[ h_u ].sum );
        NN.H_layers[0].units[ h_u ].res = func_res;

        printf("         sum(%f), f(%f)\n",
               NN.H_layers[0].units[ h_u ].sum,
               NN.H_layers[0].units[ h_u ].res );


    }

    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " HIDDEN >> HIDDEN: \n" );

    uint32_t C_LAY = 1;

    while ( C_LAY < NN.num_l ) {

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            NN.H_layers[ C_LAY ].units[ h_u ].sum = 0.0;

            for (int i_u = 0; i_u < NN.num_h; i_u++) {

                double loc_sum =
                    NN.H_layers[ C_LAY ].units[ h_u ].w[ i_u ] * NN.H_layers[ C_LAY -1 ].units[ i_u ].res;

                NN.H_layers[ C_LAY ].units[ h_u ].sum += loc_sum;

                printf("     IN:[%f] sum:[%f] loc_sum:[%f]\n",
                       NN.H_layers[ C_LAY -1 ].units[ i_u ].res,
                       NN.H_layers[ C_LAY ].units[ h_u ].sum,
                       loc_sum);

            }

            double func_res = _F( NN.H_layers[ C_LAY ].units[ h_u ].sum );
            NN.H_layers[ C_LAY ].units[ h_u ].res = func_res;

            printf("         h_u: sum(%f), f(%f)\n", NN.H_layers[ C_LAY ].units[ h_u ].sum, func_res );


        }

        C_LAY++;

    }

    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " HIDDEN >> OUTPUT: \n" );


    for (int o_u = 0; o_u < NN.num_o; o_u++) {

        NN.O_layer.units[ o_u ].sum = 0.0;

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            double loc_sum = NN.H_layers[ C_LAY-1 ].units[ h_u ].res * NN.O_layer.units[ o_u ].w[ h_u ];

            NN.O_layer.units[ o_u ].sum += loc_sum;

            printf("     IN:[%f] loc_sum:[%f]\n",
                   NN.H_layers[ C_LAY-1 ].units[ h_u ].res, loc_sum);

        }

        double func_res = _F( NN.O_layer.units[ o_u ].sum );
        NN.O_layer.units[ o_u ].res = func_res;

    }

    double out_diff = dataIn[2] - NN.O_layer.units[ 0 ].res;
    double net_error = (out_diff * 2) * NN.O_layer.units[ 0 ].res;


    printf( "%s\n", line );

    printf(" NET: sum(%f), f(%f), diff(%f), error(%f) \n",
           NN.O_layer.units[ 0 ].sum, NN.O_layer.units[ 0 ].res, out_diff, net_error);

    printf( " NN_iter->END: \n\n" );

    NN_saveNetowk();

    // ::::::::::::::::::::::::::::::::::::::::::::::

    return result;
    // ------------------------------------------------------------
}

// ====================================================================
uint32_t NN_init( uint32_t inp, uint32_t hid, uint32_t lays, uint32_t out ) {

    // ------------------------------------------------------------
    printf( "%s\n", line );
    printf( " NN_init->START: \n" );

    NN.name         = "TestNet.nnf";

    NN.num_i        = inp; // Num of Data inputs
    NN.num_h        = hid; // Num of Neurons in hidden layer
    NN.num_l        = lays; // Num of hidden layers
    NN.num_o        = out; // Num of Output Neurons
    NN.f            = ( uint32_t*) &h_than;

    NN.i            = 0;
    NN.c            = 0;
    NN.rate         = 0;
    NN.eps_past     = 0;
    NN.min_error    = 0;
    NN.cur_error    = 0;
    NN.H_layers     = ( Layer *) malloc( sizeof( Layer ) * ( 1 + NN.num_l) );

    if ( !NN.H_layers)
        _break( "ERROR: (!NN.H_layers)" );

    // ::::::::::: [INPUT] && [HIDDEN] ::::::::::::
    for (int l_i = 0; l_i < NN.num_l; l_i++) {

        printf( " ------------------------------- \n" );
        printf( " INIT: NN.H_layers[%d] \n", l_i );

        Layer layer;

        NN.H_layers[l_i]         = layer;
        NN.H_layers[l_i].i       = l_i;
        NN.H_layers[l_i].l_sum   = 0;
        NN.H_layers[l_i].l_error = 0;
        NN.H_layers[l_i].units   = (Unit *) malloc( sizeof(Unit) * NN.num_h );

        if ( !NN.H_layers[l_i].units )
            _break( "ERROR: (!NN.H_layers[l_i].units)" );

        uint32_t w_index = 0;

        for (int u_i = 0; u_i < NN.num_h; u_i++) {

            printf( "\n     INIT: NN.H_layers[%d].units[%d] \n", l_i, u_i );

            Unit unit;
            NN.H_layers[l_i].units[u_i]       = unit;
            NN.H_layers[l_i].units[u_i].i     = u_i;
            NN.H_layers[l_i].units[u_i].sum   = 0;
            NN.H_layers[l_i].units[u_i].delta = 0;
            NN.H_layers[l_i].units[u_i].error = 0;
            NN.H_layers[l_i].units[u_i].f     = (uint32_t*) h_than;

            // -------------------------------------------
            uint32_t num_of_w = NN.num_h;
            if ( l_i == 0 )
                num_of_w = NN.num_i; // INPUT

            // -------------------------------------------
            NN.H_layers[l_i].units[u_i].w  = (double *) malloc( sizeof(double) * num_of_w );
            if ( !NN.H_layers[l_i].units[u_i].w ) _break( "ERROR: (!NN.H_layers[l_i].units[u_i].w)" );

            for (int nw_i = 0; nw_i < num_of_w; nw_i++) {

                double r_w = getRand(0);
                if ( !useRand_W )
                    r_w = ALL_W[ l_i ][ w_index++ ];

                printf( "         INIT: NN.H_layers[%d].units[%d].w[%d] = %f; \n", l_i, u_i, nw_i, r_w );
                NN.H_layers[l_i].units[u_i].w[nw_i] = r_w;

            }

            // -------------------------------------------

        }

    }

    // ::::::::::: [OUTPUT] ::::::::::::
    NN.O_layer.units = ( Unit *) malloc( sizeof( Unit ) * NN.num_o );

    if ( !NN.O_layer.units )
        _break( "ERROR: (!NN.O_layer.units)" );

    printf( " ------------------------------- \n" );
    printf( " INIT: NN.O_layers\n" );

    for (int o_u= 0;  o_u< NN.num_o; o_u++) {

        Unit unit;
        NN.O_layer.units[ o_u ] = unit;

        NN.O_layer.units[ o_u ].w = (double *) malloc( sizeof(double) * NN.num_h );

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            double r_w = getRand(0);
            if ( !useRand_W )
                r_w = OUT_W[ h_u ]; // Weight To the Hidden unit

            NN.O_layer.units[ o_u ].w[ h_u ] = r_w;

            printf( "     INIT: NN.O_layer.units[ %d ].w[ %d ] = %f; \n", o_u, h_u, r_w );


        }

    }

    // ------------------------------------------------------------
    // free( NN.H_layers[l_i].units[u_i].w **** )
    // free( NN.H_layers[l_i].units **** )
    // free( NN.H_layers **** )

    return 1;
    // ------------------------------------------------------------

}

// ====================================================================
uint32_t NN_saveNetowk() {

    // ------------------------------------------------------------
    printf( "%s\n", line );
    printf( " NN_saveNetowk->START: \n" );


    FILE *pF = fopen( NN.name, "w+" );
    // fputs( NN.name, pF ); fputs( "\n", pF );
    // fclose( pF );


    char tmp[10240];

    sprintf( tmp, "# net:[%s]\n", NN.name  ); fputs( tmp, pF );
    sprintf( tmp, "inp:%d\n", NN.num_i ); fputs( tmp, pF );
    sprintf( tmp, "hid:%d\n", NN.num_h ); fputs( tmp, pF );
    sprintf( tmp, "lays:%d\n", NN.num_l ); fputs( tmp, pF );
    sprintf( tmp, "out:%d\n", NN.num_o ); fputs( tmp, pF );

    sprintf( tmp, "# ---------------------------------\n"); fputs( tmp, pF );


    // FILE *pF = fopen( "./fopen.file", "w+" );
    // char buff[1024];
    // while ( !feof( pF ) )
    //     fgets( buff, 1024, pF );

    // fclose( pF );

    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " INPUT >> HIDDEN \n" );

    for (int h_u = 0; h_u < NN.num_h; h_u++) {
        for (int i_u = 0; i_u < NN.num_i; i_u++) {

            printf( "%f, ", NN.H_layers[0].units[ h_u ].w[ i_u ] );
            sprintf( tmp, "%f,", NN.H_layers[0].units[ h_u ].w[ i_u ] ); fputs( tmp, pF );

        }
    }

    sprintf( tmp, "\n# ---------------------------------\n"); fputs( tmp, pF );
    printf( "\n" );
    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " HIDDEN >> HIDDEN: \n" );

    uint32_t C_LAY = 1;

    while ( C_LAY < NN.num_l ) {
        sprintf( tmp, "# L: [%d]\n", C_LAY); fputs( tmp, pF );

        for (int h_u = 0; h_u < NN.num_h; h_u++) {
            for (int i_u = 0; i_u < NN.num_h; i_u++) {
                printf( "%f, ", NN.H_layers[ C_LAY ].units[ h_u ].w[ i_u ] );
                sprintf( tmp, "%f,", NN.H_layers[ C_LAY ].units[ h_u ].w[ i_u ] ); fputs( tmp, pF );

            }
        }

        sprintf( tmp, "\n"); fputs( tmp, pF );
        C_LAY++;

    }

    sprintf( tmp, "\n# ---------------------------------\n"); fputs( tmp, pF );
    printf( "\n" );
    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " HIDDEN >> OUTPUT: \n" );


    for (int o_u = 0; o_u < NN.num_o; o_u++) {
        for (int h_u = 0; h_u < NN.num_h; h_u++) {
            printf( "%f, ", NN.O_layer.units[ o_u ].w[ h_u ] );
            sprintf( tmp, "%f,", NN.O_layer.units[ o_u ].w[ h_u ] ); fputs( tmp, pF );

        }

    }

    sprintf( tmp, "\n# ---------------------------------\n"); fputs( tmp, pF );
    printf( "\n" );
    // ------------------------------------------------------------

}

// ====================================================================
double getRand( uint32_t pos) {

    uint16_t dev = 10;

    // gettimeofday(&tv, NULL);
    // printf("%lu\n", tv.tv_usec ); // >> [550882, 977996, 830397]

    double d = (((double)rand())/(double)RAND_MAX);
    if (pos)
        return d / dev;

    if (
        (d > 0.1 && d < 0.3 )
        ||
        (d > 0.5 && d < 0.7 )
        ||
        (d > 0.9 && d < 0.9999 )
    )
        return d * -1;

    return d / dev;

}

// ====================================================================
double h_than( double _in) {

    double out;

    if ( _in > 0 )
        out = 1 / ( 1+exp( -_in) );
    else
        out = 1 / ( 1+exp( _in) );

    // double out = tanh( n );
    // printf("%Lf\n", out);

    return tanh( out );

}

void _break( char* data ) {
    printf("%s\n", data);
    exit( 2 );

}

// ====================================================================



