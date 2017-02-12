#include "network.h"

// ====================================================================
double COEF = 1.0;
double dataIn[] = { 0.5,0.3,  1};

double OUT_W[2] = {
    4, //  4 | 4  + ( 4 *  COEF *  b_err:(0.101388) ),
    -2, // -2 | -2 + (-2 *  COEF * b_err:(-0.050694) )
};


double ALL_W[2][4] = {
    { 2,  -3, 3,  2 }, { -1, -1, 2, -3 },
};

uint32_t useRand_W = 0;

NetWork NN;

int main(int argc, char *argv[]) {

    // ------------------------------------------------------------
    uint32_t INP_num  = 2; // i_w = [ 2, -3, 3,  2 ];
    uint32_t HID_num  = 2; // h_w = [ -1, -1, 2, -3 ];
    uint32_t LAYs_HID = 2;
    uint32_t OUT_num  = 1; // o_w = [ 4, -2 ];

    // ------------------------------------------------------------
    // if ( NN_init( 12, 14, 43, OUT_num )) {
    if ( NN_init( INP_num, HID_num, LAYs_HID, OUT_num )) {

        // TODO: Input from >>>
        // double data[2] = { 0 };

        uint32_t max_iters = 1;
        for (int i = 0; i < max_iters; i++) {

            if ( NN_iter( dataIn ) ) {

                printf( "%s\n", line );
                printf( " NN.res(%f), NN.error(diff)(%f) \n", NN.res, NN.error );


                if ( !NN_backPropogation()) _break(" ERROR: [+]\n");

            }

        }

    }

    printf(" [ main:END] \n");
    return 0;
    // ------------------------------------------------------------
}

// ====================================================================
uint32_t NN_backPropogation() {

    // NN_saveNetowk();
    double COEF = 1.0;

    // NOTE: Can be multi Deltas
    double _DELTA = 0;

    // ::::::::::::::::::::::::::::::::::::::::::::::
    printf( " ------------------------------- \n" );
    printf( " OUTPUT::delta \n" );

    for (int o_u = 0; o_u < NN.num_o; o_u++) {

        // delta[ o¹ ] = f'(res) * error // Unit.* [ .sum, .res, .delta, ._error ]
        double sum = NN.O_layer.units[ o_u ].sum;
        double res = NN.O_layer.units[ o_u ].res;
        double _error = NN.O_layer.units[ o_u ]._error;
        double res_dash = _F_Back( res );
        NN.O_layer.units[ o_u ].delta = res_dash * _error; // * NN.net_error if OUTPUT == 1

        printf(" (Out:i:%d) sum(%f), res(%f), _error(%f), res_dash(%f), delta(%f) \n",
               o_u, sum, res, _error, res_dash, NN.O_layer.units[ o_u ].delta );

    }

    for (int h_u = 0; h_u < NN.num_h; h_u++) {
        NN.H_layers[ NN.num_l-1 ].units[ h_u ]._error = NN.O_layer.units[ 0 ].delta;
        NN.H_layers[ NN.num_l-1 ].units[ h_u ].delta = NN.O_layer.units[ 0 ].delta;
    }

    // ::::::::::::::::::::::::::::::::::::::::::::::

    printf( " ------------------------------- \n" );
    printf( " HIDDEN::delta \n" );

    uint32_t C_LAY = NN.num_l-1;


    while ( C_LAY > 0 ) {

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            double sum = NN.H_layers[ C_LAY ].units[ h_u ].sum;
            double res = NN.H_layers[ C_LAY ].units[ h_u ].res;
            double _error = 0;

            for (int h_u2 = 0; h_u2 < NN.num_h; h_u2++) {


                // printf("W: [%f]\n", NN.H_layers[ C_LAY ].units[ h_u2 ].w[ h_u ]);
                // printf("D: [%f]\n", NN.H_layers[ C_LAY ].units[ h_u2 ].delta);

                _error +=
                    NN.H_layers[ C_LAY ].units[ h_u2 ].w[ h_u ] // weight from prev lay unit( h_u)
                    *
                    NN.H_layers[ C_LAY ].units[ h_u2 ].delta ;

            }

            // double w = NN.H_layers[ C_LAY ].units[ h_u ].w[ 0 ];

            double res_dash = _F_Back( res );
            NN.H_layers[ C_LAY ].units[ h_u ].delta = res_dash * (_error ); // * NN.net_error if OUTPUT == 1

            printf(" (Hid:i:%d) sum(%f), res(%f), _error(%f), res_dash(%f), delta(%f) \n",
                   h_u, sum, res, _error, res_dash, NN.H_layers[ C_LAY ].units[ h_u ].delta );










            // for next (prev) layer, current delta IS ERROR value
            NN.H_layers[ C_LAY ].units[ h_u ]._error = NN.H_layers[ C_LAY ].units[ h_u ].delta;


        }

        C_LAY--;

    }

    /*
    printf(" *** HID[0] ***\n");
    for (int h_u = 0; h_u < NN.num_h; h_u++) {

        double sum = NN.H_layers[ 0 ].units[ h_u ].sum;
        double res = NN.H_layers[ 0 ].units[ h_u ].res;
        double _error = NN.H_layers[ 0 ].units[ h_u ]._error;

        double res_dash = _F_Back( res );
        NN.H_layers[ C_LAY ].units[ h_u ].delta = res_dash * _error; // * NN.net_error if OUTPUT == 1

        printf(" (Hid:i:%d) sum(%f), res(%f), _error(%f), res_dash(%f), delta(%f) \n",
               h_u, sum, res, _error, res_dash, NN.H_layers[ 0 ].units[ h_u ].delta );


    }
    */

    // ::::::::::::::::::::::::::::::::::::::::::::::
    /*
    uint32_t C_LAY = NN.num_l-1;

    for (int o_u = 0; o_u < NN.num_o; o_u++) {

        for (int w_u = 0; w_u < NN.num_h; w_u++) {

            // Unit.*.sum;
            // Unit.*.res;
            // Unit.*.delta;
            // Unit.*._error;

            double sum = NN.O_layer.units[ o_u ].sum;
            double res = NN.O_layer.units[ o_u ].res;
            double _error = NN.O_layer.units[ o_u ]._error;

            // delta[ o¹ ] = f'(res) * error
            NN.O_layer.units[ o_u ].delta = _F_Back( res ) * _error; // * NN.net_error if OUTPUT == 1

        }

    }
    */

    // ::::::::::::::::::::::::::::::::::::::::::::::
    /*
    printf( " ------------------------------- \n" );
    printf( " HIDDEN << HIDDEN: \n" );


    // C_LAY--;

    while ( C_LAY > 0 ) {

    for (int h_u = 0; h_u < NN.num_h; h_u++) {

    double neuron_out = NN.H_layers[ C_LAY ].units[ h_u ].res;
    neuron_out *= ( 1 - neuron_out ) * NN.H_layers[ C_LAY ].units[ h_u ].error_out;



    // printf(" PREV: Layer: res:[%f]\n", NN.H_layers[ C_LAY-1 ].units[ h_u ].res);

    // NN.H_layers[ C_LAY ].units[ h_u ].res * neuron_out *
    printf(" H-Out << [%f] err:[%f]\n", NN.H_layers[ C_LAY ].units[ h_u ].res, neuron_out);
    // printf(" -[%f]- \n", NN.H_layers[ C_LAY ].units[ h_u ].error_in);

    }

    // }

    C_LAY--;

    }

    */

    // ::::::::::::::::::::::::::::::::::::::::::::::
    return 1;

}


// ====================================================================
uint32_t NN_iter( double* dataIn ) {

    // ------------------------------------------------------------
    printf( "%s\n", line );
    printf( " NN_iter->START: \n" );

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

    // *^* NN.sum = 0;

    for (int o_u = 0; o_u < NN.num_o; o_u++) {

        NN.O_layer.units[ o_u ].sum = 0.0;

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            double loc_sum = NN.H_layers[ C_LAY-1 ].units[ h_u ].res * NN.O_layer.units[ o_u ].w[ h_u ];
            NN.O_layer.units[ o_u ].sum += loc_sum;
            printf("     IN:[%f] loc_sum:[%f]\n", NN.H_layers[ C_LAY-1 ].units[ h_u ].res, loc_sum);

        }

        double func_res = _F( NN.O_layer.units[ o_u ].sum );
        NN.O_layer.units[ o_u ].res = func_res;

        // FIXME: >> TODO
        NN.O_layer.units[ o_u ]._error = dataIn[2] - func_res;

    }

    // TODO: Multi OUTPUT cycle, local error for each output
    NN.error = dataIn[2] - NN.O_layer.units[ 0 ].res;
    // *^* NN.sum  += NN.O_layer.units[ 0 ].sum;
    NN.res   = NN.O_layer.units[ 0 ].res;

    printf( " NN_iter->END: \n\n" );
    // ::::::::::::::::::::::::::::::::::::::::::::::

    return 1;
    // ------------------------------------------------------------
}

// ====================================================================
uint32_t NN_init( uint32_t inp, uint32_t hid, uint32_t lays, uint32_t out ) {

    // ------------------------------------------------------------
    uint32_t mkLog = 0;
    // ------------------------------------------------------------
    if (mkLog) printf( "%s\n", line );
    if (mkLog) printf( " NN_init->START: \n" );

    NN.name         = "TestNet.nnf";

    NN.num_i        = inp; // Num of Data inputs
    NN.num_h        = hid; // Num of Neurons in hidden layer
    NN.num_l        = lays; // Num of hidden layers
    NN.num_o        = out; // Num of Output Neurons

    NN.rate         = 0.02;
    NN.error        = 0;
    NN.res          = 0;

    NN.H_layers     = ( Layer *) malloc( sizeof( Layer ) * ( 1 + NN.num_l) );

    if ( !NN.H_layers)
        _break( "ERROR: (!NN.H_layers)" );

    // ::::::::::: [INPUT] && [HIDDEN] ::::::::::::
    for (int l_i = 0; l_i < NN.num_l; l_i++) {

        if (mkLog) printf( " ------------------------------- \n" );
        if (mkLog) printf( " INIT: NN.H_layers[%d] \n", l_i );

        Layer layer;

        NN.H_layers[l_i]         = layer;
        NN.H_layers[l_i].i       = l_i;
        NN.H_layers[l_i].units   = (Unit *) malloc( sizeof(Unit) * NN.num_h );

        if ( !NN.H_layers[l_i].units )
            _break( "ERROR: (!NN.H_layers[l_i].units)" );

        uint32_t w_index = 0;

        for (int u_i = 0; u_i < NN.num_h; u_i++) {

            if (mkLog) printf( "\n     INIT: NN.H_layers[%d].units[%d] \n", l_i, u_i );

            Unit unit;
            NN.H_layers[l_i].units[u_i]       = unit;
            NN.H_layers[l_i].units[u_i].i     = u_i;
            NN.H_layers[l_i].units[u_i].sum   = 0;
            NN.H_layers[l_i].units[u_i].res   = 0;
            NN.H_layers[l_i].units[u_i].delta = 0;

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

                if (mkLog) printf( "         INIT: NN.H_layers[%d].units[%d].w[%d] = %f; \n", l_i, u_i, nw_i, r_w );
                NN.H_layers[l_i].units[u_i].w[nw_i] = r_w;

            }

            // -------------------------------------------

        }

    }

    // ::::::::::: [OUTPUT] ::::::::::::
    NN.O_layer.units = ( Unit *) malloc( sizeof( Unit ) * NN.num_o );

    if ( !NN.O_layer.units )
        _break( "ERROR: (!NN.O_layer.units)" );

    if (mkLog) printf( " ------------------------------- \n" );
    if (mkLog) printf( " INIT: NN.O_layers\n" );

    for (int o_u= 0;  o_u< NN.num_o; o_u++) {

        Unit unit;
        NN.O_layer.units[ o_u ] = unit;
        NN.O_layer.units[ o_u ].w = (double *) malloc( sizeof(double) * NN.num_h );

        for (int h_u = 0; h_u < NN.num_h; h_u++) {

            double r_w = getRand(0);
            if ( !useRand_W )
                r_w = OUT_W[ h_u ]; // Weight To the Hidden unit

            NN.O_layer.units[ o_u ].w[ h_u ] = r_w;

            if (mkLog) printf( "     INIT: NN.O_layer.units[ %d ].w[ %d ] = %f; \n", o_u, h_u, r_w );


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
    /*
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
    */
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
double _F( double _in ) { // == f(v)

    if ( _in > 0 )
        return 1 / ( 1+exp( -_in) );

    return 1 / ( 1+exp( _in) );
    // return tanh( in );

}

// ====================================================================
double _F_Back( double _in ) { // == f'(v)
    double res = _F( _in );
    return ( res * ( 1 - res ) );

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



