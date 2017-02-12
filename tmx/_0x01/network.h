#ifndef _NETWORK_H_
#define _NETWORK_H_

// ----------------------------------------------
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>

// ----------------------------------------------
#include <time.h>

#include <sys/time.h>
struct timeval tv;
// struct timespec ts;

// gettimeofday(&tv, NULL);
// tv.tv_usec;

// ----------------------------------------------
typedef unsigned long long uint64_t;
typedef long long int64_t;

typedef unsigned int uint32_t;
typedef int int32_t;

typedef unsigned short uint16_t;
typedef short int16_t;

// typedef unsigned char uint8_t;
// typedef char int8_t;

// ----------------------------------------------
typedef struct {

    int32_t i;
    double sum;
    double res;
    double delta;
    double _error;
    double *w;

} __attribute__((packed)) Unit;


typedef struct {

    int32_t i;
    Unit *units;

} __attribute__((packed)) Layer;


typedef struct {
    char *name;
    int32_t num_i; // Num of Data inputs
    int32_t num_h; // Num of Neurons in hidden layer
    int32_t num_l; // Num of hidden layers
    int32_t num_o; // Num of Output Neurons
    double rate;
    double error;
    double res;

    Layer *H_layers;
    Layer O_layer;

} __attribute__((packed)) NetWork;

// ----------------------------------------------
uint32_t NN_init( uint32_t inp, uint32_t hid, uint32_t lays, uint32_t out );
uint32_t NN_iter( double *dataIn );
uint32_t NN_saveNetowk();
uint32_t NN_backPropogation();


double getRand( uint32_t pos);
double h_than( double n);
void _break( char* data );


double _F( double _in );        // f(x)
double _F_Back( double _in );   // f'(x)


// ----------------------------------------------
char line[] = {" ======================================================================== "};


// ----------------------------------------------

#endif
