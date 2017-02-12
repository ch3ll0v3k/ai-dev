#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "network.h"

// ====================================================================
#define BREAK_VALUE 0.1

uint32_t Il = 3;
uint32_t Ol = 1;
uint32_t ITER = 0; // количество итераций
double KIK_pin = 0.1;

uint8_t use_non_float_activation = 0;

double in[3] = { 0 }; // создаём входы
double W[3]  = { 0 }; // весовые коэффициенты
double out; // храним выход сети

uint8_t learn_data_in = 8;

double tableOfLearn[8][4] = {
    {0,0,0, 1},
    {1,0,0, 0.66},
    {0,1,0, 0.66},
    {0,0,1, 0.66},
    {1,1,0, 0.33},
    {0,1,1, 0.33},
    {1,0,1, 0.33},
    {1,1,1, 0},
};

// ====================================================================
double W_0 = 0;
double W_1 = 0;
double W_2 = 0;

uint32_t init_network() {

    time_t t;
    srand((unsigned) time(&t));

    uint32_t i=0;
    for ( ; i < Il ; i++ ) {
        double d = (rand() % 50)/90.2;
        W[ i ] = d;
        // printf("%f\n", d);

    }

    // exit(0);
    // W[ 0 ] = 0.041;
    // W[ 1 ] = 0.083;
    // W[ 2 ] = ?;

    W_0 = W[ 0 ];
    W_1 = W[ 1 ];
    W_2 = W[ 2 ];


    return 1;
}

// ====================================================================
void summator() {

    out = 0; // обнуляем выход

    uint32_t ii = 0;
    for ( ; ii < Il; ii++ )
        out += in[ii] * W[ii]; // вход * вес, суммируем.

    if ( use_non_float_activation ) {

        // функция активации
        if ( out > BREAK_VALUE ) {
            // * printf("F-out: [*] [%2.3f], ", out );
            out = 1;

        } else {
            // * printf("F-out: [ ] [%2.3f], ", out );
            out = 0;

        }

    }


}

double mF[2][4] = { 0 };

void train() {
    double gError = 0; // создаём счётчик ошибок
    double gErrorG = 0; // создаём счётчик ошибок

    uint8_t INV = 0;
    uint8_t times = 0;
    uint8_t break_after_times = 2;

    do {
        gError = 0; // обнуляем счётчик
        double error = 0;

        INV = INV ? 0 : 1;

        // printf("ITER: [%d] ", ITER );
        printf(" --- \n");

        uint32_t i = 0;
        for ( ; i < learn_data_in; i++ ) {

            // -------------------------------------------
            // копируем в входы обучающие входы
            in[0] = tableOfLearn[i][0];
            in[1] = tableOfLearn[i][1];
            in[2] = tableOfLearn[i][2];

            // -------------------------------------------
            summator();

            // -------------------------------------------
            error = tableOfLearn[i][3] - out; // получаем ошибку

            mF[INV][ i ] = error;

            gError += mk_abs(error); // суммируем ошибку в модуле

            //printf(" gE:[%f] e:[%f] in(%1.3f, %1.3f) W(%1.3f, %1.3f) out(%1.3f) R-out(%1.3f) \n", gError, error, in[0], in[1], W[0], W[1], out, tableOfLearn[i][2] );
            // usleep(250000);
            // usleep(125000);

            uint32_t j=0;
            for ( ; j < Il /*in.length;*/; j++ ) {

                // W[j] += 0.1 * error * in[j]; // старый вес + скорость * ошибку * i-ый вход
                W[j] += BREAK_VALUE * error * in[j]; // старый вес + скорость * ошибку * i-ый вход

                /*
                if ( error > 0.5 ) {
                    W[j] += 0.1 * error * in[j]; // старый вес + скорость * ошибку * i-ый вход

                } else {
                    W[j] += 1.1 * error * in[j]; // старый вес + скорость * ошибку * i-ый вход

                }
                */

            }
            // -------------------------------------------
        }

        if ( gErrorG < gError ) {

            W[0] *= (1+KIK_pin);
            W[1] *= (1+KIK_pin);
            W[2] *= (1+KIK_pin);

        } else {
            //W[0] *= (1-KIK_pin);
            //W[1] *= (1-KIK_pin);
            //W[2] *= (1-KIK_pin);

        }

        gErrorG = gError;

        printf(" gE:[%f] e:[%f] in(%1.3f, %1.3f, %1.3f) W(%1.3f, %1.3f, %1.3f) out(%1.3f) R-out(%1.3f) \n",
               gError, error, in[0],in[1],in[2], W[0],W[1],W[2], out, tableOfLearn[i][3] );
        // printf(" N-out:[%f] gE:[%f] e:[%f]\n", out, gError, error);
        ITER++; // увеличиваем на 1 итерации

    } while (gError > 0.0001 ); // пока gError не равно 0, выполняем код
    //} while (gError !=0 ); // пока gError не равно 0, выполняем код

}

// ====================================================================
void test() {

    train();

    // use_non_float_activation = 0;

    printf("---------------------------------------------------\n");
    uint32_t i=0;
    for ( ; i < learn_data_in; i++ ) {

        in[0] = tableOfLearn[i][0];
        in[1] = tableOfLearn[i][1];

        summator();
        printf(" [%c] in(%1.3f, %1.3f) W(%1.3f, %1.3f) out(%1.3f) R-out(%1.3f) \n",
               (out == tableOfLearn[i][2] ? '*' : ' '), in[0], in[1], W[0], W[1], out, tableOfLearn[i][2] );

    }

}

// ====================================================================
uint32_t main(uint32_t argc, int8_t *argv[]) {

    if ( init_network()) {
        test();

    }

    return 0;
}


// ====================================================================
double mk_abs( double d ) {

    if ( d > 0 ) return d;

    double t=0;
    while ( d < 0.00000001 ) {
        t+=0.00001;
        d+=0.00001;

    }

    return t;

}

